"""参数遍历：对若干参数的候选值做网格组合，多进程并行训练，汇总比较各组合在验证集上的表现。

用进程而不是线程并行：小型 MLP 每个 batch 的耗时主要是 Python 代码，线程受 GIL 限制几乎不能同时运行。
数据只准备一次（读取、划分、标准化），写到临时文件，各进程启动时读入，之后依次训练分配给它的组合。

进程池是自己管理的（见 _Pool 和 sweep_worker.py），而不是 ProcessPoolExecutor：后者在任一进程意外退出时
会结束所有进程、让全部组合失败，共享的锁和队列还可能让停止操作卡住。这里一个进程退出只影响它正在训练的组合
（重试一次），同时并行的进程数减一；停止时最多等待 STOP_TIMEOUT 秒，超时强制结束进程。

结果目录，例如 runs/20261027-160000_sweep/：
  sweep.json       基础配置、遍历的参数及候选值、各组合的参数
  results.csv      每个组合一行：参数取值、状态、最佳 epoch、训练 / 验证 / 测试集指标、耗时（每完成一组就更新）
  001_bs32_lr0.001_h1-64_h2-32/ …   每个组合一个普通的训练结果文件夹（见 runs.py）
"""

from __future__ import annotations

import itertools
import json
import math
import multiprocessing as mp
import os
import pickle
import re
import sys
import tempfile
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing import connection
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .config import ExperimentConfig, OptimConfig
from .data import SPLITS, PreparedData, prepare_data
from .models import get_spec
from .models.base import KIND_CHOICE, KIND_FLOAT, KIND_INT, KIND_INT_LIST
from .runs import new_run_dir, save_run
from .sweep_worker import last_line, worker_main
from .trainer import TrainResult, fit

SWEEP_FILE = "sweep.json"
RESULTS_FILE = "results.csv"

STATUS_PENDING = "等待"
STATUS_RUNNING = "训练中"
STATUS_DONE = "完成"
STATUS_STOPPED = "已停止"
STATUS_FAILED = "失败"
STATUS_CANCELLED = "未运行"
FINAL_STATUSES = (STATUS_DONE, STATUS_STOPPED, STATUS_FAILED, STATUS_CANCELLED)

STOP_TIMEOUT = 30.0  # 停止后等待正在训练的组合保存结果的最长时间（秒），超时强制结束进程
MAX_RESTARTS = 3  # 所有进程都已退出、还有组合没训练时，最多重新启动进程的次数
CRASH_HINT = (
    "常见原因是内存不足，可以调小“并行进程数”。Windows 上出现 WinError 1455（页面文件太小）时，"
    "也可以增大虚拟内存，或改装只用 CPU 的 PyTorch（体积小得多）"
)


def default_workers() -> int:
    """默认并行进程数：约等于物理核数（逻辑核数的一半），最多 8 个，避免占用过多内存。"""
    return max(1, min((os.cpu_count() or 1) // 2, 8))

# 用来比较各组合的指标；交叉熵越低越好，其余越高越好
RANK_METRICS = {
    "val_accuracy": "验证集准确率",
    "val_f1": "验证集 F1",
    "val_auc": "验证集 AUC",
    "val_ce": "验证集交叉熵",
}
LOWER_IS_BETTER = {"val_ce"}
METRIC_KEYS = ("ce", "accuracy", "precision", "recall", "f1", "auc")

# 可遍历的训练参数：配置项 → (界面名称, 组合名称中的缩写, 取值示例；为空时按当前值生成)
TRAIN_AXES = {
    "batch_size": ("batch size", "bs", ""),
    "learning_rate": ("learning rate", "lr", "0.001, 0.0003"),
    "l2_lambda": ("L2 系数 λ", "l2", "0, 0.0001, 0.001"),
    "epochs": ("epoch 数", "ep", ""),
}
# 命令行 --sweep 的别名
ALIASES = {
    "bs": "train.batch_size",
    "batch-size": "train.batch_size",
    "lr": "train.learning_rate",
    "learning-rate": "train.learning_rate",
    "l2": "train.l2_lambda",
}


@dataclass(frozen=True)
class SweepAxis:
    key: str  # train.batch_size / model.hidden_sizes[0] / model.<参数名>
    label: str
    short: str
    base: Any  # 基础配置中的取值
    parse_one: Callable[[str], Any]
    example: str  # 界面中的取值示例

    def parse(self, text: str) -> list:
        """“32, 64, 128” → [32, 64, 128]；逗号、分号、空格分隔，重复值只保留一个。"""
        values: list = []
        for part in re.split(r"[,，;；\s]+", text.strip()):
            if not part:
                continue
            try:
                value = self.parse_one(part)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{self.label}：{exc}" if str(exc) else f"{self.label}：“{part}” 无效") from None
            if value not in values:
                values.append(value)
        return values


@dataclass
class Trial:
    index: int  # 从 1 开始
    name: str  # 结果子文件夹名
    overrides: dict[str, Any]  # 遍历参数的取值
    config: ExperimentConfig


def available_axes(base: ExperimentConfig) -> list[SweepAxis]:
    """可遍历的参数：训练参数 + 模型的数值 / 选项参数；整数列表参数（如隐藏层节点数）按每个位置分别遍历。"""
    axes = []
    defaults = OptimConfig()
    for name, (label, short, example) in TRAIN_AXES.items():
        kind, value = type(getattr(defaults, name)), getattr(base.train, name)
        axes.append(SweepAxis(f"train.{name}", label, short, value, _number_parser(kind), example or _example(value)))
    spec = get_spec(base.model.name)
    params = spec.resolve_params(base.model.params)
    for p in spec.params:
        short = p.short or p.name
        if p.kind == KIND_INT_LIST:
            for i, value in enumerate(params[p.name]):
                parse = _number_parser(int)
                axes.append(SweepAxis(f"model.{p.name}[{i}]", f"第 {i + 1} 层{p.label}", f"{short}{i + 1}-", value,
                                      lambda text, p=p, parse=parse: p.parse([parse(text)])[0],
                                      p.example or _example(value)))
        elif p.kind in (KIND_INT, KIND_FLOAT, KIND_CHOICE):
            value = params[p.name]
            axes.append(SweepAxis(f"model.{p.name}", p.label, f"{short}-", value, p.parse,
                                  p.example or _example(value)))
    return axes


def resolve_axis_key(base: ExperimentConfig, name: str) -> str:
    """命令行参数名 → 配置键。支持 batch_size / lr / l2 / epochs、hidden1、h2、模型参数名或完整键。"""
    keys = {a.key for a in available_axes(base)}
    name = name.strip()
    candidates = [name, ALIASES.get(name, ""), f"train.{name.replace('-', '_')}", f"model.{name}"]
    match = re.fullmatch(r"(?:hidden|h)(\d+)", name)
    if match:
        candidates.append(f"model.hidden_sizes[{int(match.group(1)) - 1}]")
    for key in candidates:
        if key in keys:
            return key
    raise ValueError(f"不能遍历参数 “{name}”，可选：{'、'.join(sorted(keys))}")


def expand_grid(base: ExperimentConfig, axes: dict[str, list]) -> list[Trial]:
    """按 axes（配置键 → 候选值）的笛卡尔积生成各组合；没有候选值的参数使用基础配置。"""
    known = {a.key: a for a in available_axes(base)}
    unknown = [k for k in axes if k not in known]
    if unknown:
        raise ValueError(f"不能遍历的参数：{'、'.join(unknown)}")
    keys = [k for k, values in axes.items() if values]
    combos = list(itertools.product(*(axes[k] for k in keys)))
    width = max(3, len(str(len(combos))))
    trials = []
    for index, combo in enumerate(combos, start=1):
        cfg = ExperimentConfig.from_dict(base.to_dict())
        spec = get_spec(cfg.model.name)
        cfg.model.params = spec.resolve_params(cfg.model.params)
        for key, value in zip(keys, combo):
            _apply(cfg, key, value)
        try:
            cfg.validate()
            spec.resolve_params(cfg.model.params)
        except ValueError as exc:
            raise ValueError(f"组合 {_describe(known, keys, combo)} 无效：{exc}") from None
        label = "_".join(f"{known[k].short}{_fmt(v)}" for k, v in zip(keys, combo)) or "base"
        name = f"{index:0{width}d}_" + re.sub(r'[\\/:*?"<>|\s]+', "_", label)
        trials.append(Trial(index, name, dict(zip(keys, combo)), cfg))
    return trials


def run_sweep(
    base: ExperimentConfig,
    trials: list[Trial],
    workers: int,
    rank_by: str = "val_accuracy",
    on_event: Callable[[tuple], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    log: Callable[[str], None] | None = None,
) -> Path:
    """多进程训练所有组合，返回结果目录。

    on_event 收到：("ready", 结果目录)、("start", 序号)、("epoch", 序号, history 记录)、("done", 序号, 结果行)。
    should_stop 返回 True 后：正在训练的组合在当前 batch 后停止并保存，尚未开始的组合不再运行。
    """
    on_event = on_event or (lambda _event: None)
    should_stop = should_stop or (lambda: False)
    log = log or (lambda _msg: None)
    if not trials:
        raise ValueError("没有要训练的参数组合")
    data = prepare_data(base.data, base.train.seed, log)
    if not len(data.val):
        log("注意：没有验证集，各组合只能按训练集指标比较，建议设置验证集比例")

    sweep_cfg = ExperimentConfig.from_dict(base.to_dict())
    sweep_cfg.run_name = f"sweep_{base.run_name}" if base.run_name else "sweep"
    sweep_dir = new_run_dir(sweep_cfg)
    keys = list(trials[0].overrides)
    saved_base = ExperimentConfig.from_dict(base.to_dict())
    saved_base.data.features = list(data.features)
    saved_base.data.datasets = list(data.train_files)
    saved_base.data.val_datasets, saved_base.data.test_datasets = list(data.val_files), list(data.test_files)
    (sweep_dir / SWEEP_FILE).write_text(
        json.dumps(
            {
                "base": saved_base.to_dict(),
                "axes": {k: _unique([t.overrides[k] for t in trials]) for k in keys},
                "rank_by": rank_by,
                "trials": [{"index": t.index, "name": t.name, "overrides": t.overrides} for t in trials],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    rows = {t.index: {"trial": t.index, "name": t.name, **t.overrides, "status": STATUS_PENDING} for t in trials}
    _write_results(sweep_dir, rows, keys)
    on_event(("ready", sweep_dir))

    workers = max(1, min(workers, len(trials), 61 if sys.platform == "win32" else workers))  # Windows 上限 61
    threads = max(1, (os.cpu_count() or 1) // workers)
    log(f"共 {len(trials)} 组参数，{workers} 个进程并行（每个进程 {threads} 个计算线程）")
    if base.train.device == "auto" and torch.cuda.is_available():
        log("“设备”为自动时，参数遍历在 CPU 上训练：多个进程同时使用 GPU 会占用大量显存和内存，小模型在 CPU 上也很快")
    # 数据经文件传给子进程：直接作为进程参数传递时，父进程要等每个子进程导入完模块才能写入管道，进程只能逐个启动
    fd, data_file = tempfile.mkstemp(prefix="carinout_sweep_", suffix=".pkl")
    with os.fdopen(fd, "wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    try:
        # spawn：Windows 只支持这种方式；Linux 上也避免 fork 带走界面和 torch 的线程状态
        pool = _Pool(mp.get_context("spawn"), data_file, threads, sweep_dir, rows, keys, on_event, log)
        pool.run(trials, workers, should_stop)
    finally:
        os.unlink(data_file)
    _write_results(sweep_dir, rows, keys)
    counts = pd.Series([r["status"] for r in rows.values()]).value_counts()
    summary = "，".join(f"{status} {counts[status]} 组" for status in FINAL_STATUSES if status in counts)
    log(f"参数遍历结束（共 {len(trials)} 组：{summary}），结果保存在 {sweep_dir}")
    return sweep_dir


@dataclass
class _Worker:
    process: Any
    conn: Any  # 与该进程通信的管道
    ready: bool = False  # 已导入模块、读入数据
    trial: Trial | None = None  # 正在训练的组合
    error: str = ""  # 启动失败的原因


class _Pool:
    """训练进程池：分配组合、转发进度、处理进程意外退出和停止。"""

    def __init__(
        self,
        ctx,
        data_file: str,
        threads: int,
        sweep_dir: Path,
        rows: dict[int, dict],
        keys: list[str],
        on_event: Callable[[tuple], None],
        log: Callable[[str], None],
    ) -> None:
        self._ctx = ctx
        self._data_file = data_file
        self._threads = threads
        self._sweep_dir = sweep_dir
        self._rows = rows
        self._keys = keys
        self._on_event = on_event
        self._log = log
        self._stop_flag = ctx.RawValue("b", 0)  # 无锁：进程被强制结束时不会留下未释放的锁
        self._workers: list[_Worker] = []
        self._queue: deque[Trial] = deque()
        self._retried: set[int] = set()
        self._restarts = 0
        self._last_error = ""
        self._hinted = False
        self._stop_time: float | None = None

    def run(self, trials: list[Trial], size: int, should_stop: Callable[[], bool]) -> None:
        self._queue.extend(trials)
        for _ in range(size):
            self._spawn()
        try:
            while self._queue or any(w.trial for w in self._workers):
                if self._stop_time is None and should_stop():
                    self._request_stop()
                if self._stop_time is not None and time.monotonic() - self._stop_time > STOP_TIMEOUT:
                    self._force_stop()
                    break
                self._assign()
                self._poll()
                self._reap()
                if not self._workers and self._queue and not self._restart():
                    break
        finally:
            self._shutdown()

    def _spawn(self) -> None:
        conn, child_conn = self._ctx.Pipe()
        args = (child_conn, self._data_file, self._stop_flag, self._threads)
        process = self._ctx.Process(target=worker_main, args=args, daemon=True)
        process.start()
        child_conn.close()
        self._workers.append(_Worker(process, conn))

    def _assign(self) -> None:
        if self._stop_time is not None:
            return
        for worker in self._workers:
            if worker.ready and worker.trial is None and self._queue:
                worker.trial = self._queue.popleft()
                cfg = worker.trial.config.to_dict()
                if cfg["train"]["device"] == "auto":
                    cfg["train"]["device"] = "cpu"
                try:
                    worker.conn.send((worker.trial.index, cfg, str(self._sweep_dir / worker.trial.name)))
                except OSError:
                    pass  # 进程已退出，由 _reap 处理

    def _poll(self) -> None:
        """等待任一进程发来消息或退出（最多 0.2 秒），读取所有消息。"""
        handles = [w.conn for w in self._workers] + [w.process.sentinel for w in self._workers]
        if handles:
            connection.wait(handles, timeout=0.2)
        for worker in self._workers:
            self._read(worker)

    def _read(self, worker: _Worker) -> None:
        try:
            while worker.conn.poll():
                self._handle(worker, worker.conn.recv())
        except (EOFError, OSError):
            pass  # 进程已退出，由 _reap 处理

    def _handle(self, worker: _Worker, message: tuple) -> None:
        kind = message[0]
        if kind == "ready":
            worker.ready = True
        elif kind == "init_error":
            worker.error = last_line(message[1])
            self._log("训练进程启动失败：\n" + message[1].rstrip())
        elif kind in ("start", "epoch"):
            self._on_event(message)
        elif kind == "result":
            index, row = message[1], message[2]
            worker.trial = None
            self._finish(index, row)

    def _reap(self) -> None:
        """处理意外退出的进程：它正在训练的组合重试一次，再次失败则标记失败；不补充新进程（并行数减一）。"""
        for worker in [w for w in self._workers if not w.process.is_alive()]:
            self._read(worker)  # 读完退出前发出的消息
            self._workers.remove(worker)
            worker.conn.close()
            reason = worker.error or _exit_reason(worker.process.exitcode)
            self._last_error = reason
            trial = worker.trial
            if trial is not None:
                if self._stop_time is None and trial.index not in self._retried:
                    self._retried.add(trial.index)
                    self._queue.appendleft(trial)
                    self._on_event(("retry", trial.index))
                    self._log(f"组合 #{trial.index} 的训练进程意外退出（{reason}），稍后重试")
                else:
                    self._finish(trial.index, {"status": STATUS_FAILED, "error": f"训练进程意外退出：{reason}"})
            if self._queue or any(w.trial for w in self._workers):
                note = "" if self._hinted else "。" + CRASH_HINT
                self._hinted = True
                self._log(f"一个训练进程已退出（{reason}），同时训练的进程数减为 {len(self._workers)}{note}")

    def _restart(self) -> bool:
        """所有进程都已退出但还有组合没训练：重新启动 1 个进程。次数用完时剩余组合标记失败，返回 False。"""
        if self._restarts < MAX_RESTARTS:
            self._restarts += 1
            self._log(f"所有训练进程都已退出，重新启动 1 个进程继续（第 {self._restarts} 次）")
            self._spawn()
            return True
        error = f"训练进程反复退出：{self._last_error}" if self._last_error else "训练进程反复退出"
        while self._queue:
            self._finish(self._queue.popleft().index, {"status": STATUS_FAILED, "error": error})
        self._log(f"{error}，剩余组合不再训练。{CRASH_HINT}")
        return False

    def _request_stop(self) -> None:
        self._stop_time = time.monotonic()
        self._stop_flag.value = 1
        while self._queue:
            self._finish(self._queue.popleft().index, {"status": STATUS_CANCELLED})
        self._log("正在停止：正在训练的组合会保存当前结果，其余组合不再运行")

    def _force_stop(self) -> None:
        for worker in self._workers:
            if worker.trial is not None:
                worker.process.kill()
                update = {"status": STATUS_STOPPED, "error": "停止超时，已强制结束，未保存结果"}
                self._finish(worker.trial.index, update)
                worker.trial = None
        self._log(f"等待超过 {STOP_TIMEOUT:.0f} 秒，已强制结束仍在运行的训练进程")

    def _finish(self, index: int, update: dict) -> None:
        self._rows[index].update(update)
        self._on_event(("done", index, dict(self._rows[index])))
        _write_results(self._sweep_dir, self._rows, self._keys)

    def _shutdown(self) -> None:
        for worker in self._workers:
            try:
                worker.conn.send(None)
            except OSError:
                pass
        deadline = time.monotonic() + 5
        for worker in self._workers:
            worker.process.join(max(0.0, deadline - time.monotonic()))
            if worker.process.is_alive():
                worker.process.kill()
                worker.process.join()
            worker.conn.close()
        self._workers.clear()


def _exit_reason(code: int | None) -> str:
    if code is None:
        return "进程无响应"
    if code < 0:
        return f"被系统结束（信号 {-code}）"
    if code > 255:  # Windows 的异常代码，如 0xC0000005（访问冲突）
        return f"退出码 0x{code & 0xFFFFFFFF:08X}"
    return f"退出码 {code}"


def rank_results(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    """按指标从好到差排序（没有该指标的组合排在最后），并列时验证集交叉熵低的在前。"""
    if results.empty or metric not in results:
        return results
    ascending = metric in LOWER_IS_BETTER
    by = [metric] + (["val_ce"] if metric != "val_ce" and "val_ce" in results else [])
    return results.sort_values(by, ascending=[ascending, True][: len(by)], na_position="last", kind="stable")


def best_trial(results: pd.DataFrame, metric: str) -> pd.Series | None:
    ranked = rank_results(results, metric)
    if ranked.empty or metric not in ranked or pd.isna(ranked[metric].iloc[0]):
        return None
    return ranked.iloc[0]


@dataclass
class SweepRecord:
    path: Path
    base: ExperimentConfig
    axes: dict[str, list]
    rank_by: str
    results: pd.DataFrame

    def trial_config(self, row: pd.Series) -> ExperimentConfig:
        """某个组合的完整配置（读取它自己的 config.json）。"""
        return ExperimentConfig.load(self.path / str(row["name"]) / "config.json")


def load_sweep(path: str | Path) -> SweepRecord:
    path = Path(path)
    if not (path / SWEEP_FILE).is_file():
        raise ValueError(f"{path} 不是参数遍历结果文件夹（缺少 {SWEEP_FILE}）")
    info = json.loads((path / SWEEP_FILE).read_text(encoding="utf-8"))
    results_file = path / RESULTS_FILE
    results = pd.read_csv(results_file, encoding="utf-8-sig") if results_file.is_file() else pd.DataFrame()
    base = ExperimentConfig.from_dict(info["base"])
    return SweepRecord(path, base, info.get("axes", {}), info.get("rank_by", "val_accuracy"), results)


def is_sweep_dir(path: str | Path) -> bool:
    return (Path(path) / SWEEP_FILE).is_file()


# ---------------------------------------------------------------------------- 子进程中运行
def run_trial(
    index: int,
    cfg_dict: dict,
    run_dir: str,
    data: PreparedData,
    send: Callable[[tuple], None],
    should_stop: Callable[[], bool],
) -> dict:
    """训练一个组合并保存结果，返回结果行。send 用于回传 ("start", 序号) 和 ("epoch", 序号, 记录)。"""
    if should_stop():
        return {"status": STATUS_CANCELLED}
    cfg = ExperimentConfig.from_dict(cfg_dict)
    send(("start", index))
    started = time.perf_counter()
    result = fit(cfg, data, lambda record: send(("epoch", index, record)), should_stop)
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    save_run(path, cfg, data, result)
    return {**_summary(result), "seconds": time.perf_counter() - started}


def _summary(result: TrainResult) -> dict:
    row: dict[str, Any] = {
        "status": STATUS_STOPPED if result.stopped else STATUS_DONE,
        "best_epoch": result.best_epoch,
        "epochs_run": len(result.history),
    }
    for split in SPLITS:
        ev = result.evals.get(split)
        for key in METRIC_KEYS:
            row[f"{split}_{key}"] = math.nan if ev is None else ev.loss if key == "ce" else ev.metrics[key]
    return row


# ---------------------------------------------------------------------------- 内部
def _number_parser(kind: type) -> Callable[[str], Any]:
    def parse(text: str):
        try:
            value = float(text)
        except ValueError:
            raise ValueError(f"“{text}” 不是数字") from None
        if kind is int:
            if value != int(value):
                raise ValueError(f"“{text}” 不是整数")
            return int(value)
        return value

    return parse


def _apply(cfg: ExperimentConfig, key: str, value: Any) -> None:
    section, name = key.split(".", 1)
    if section == "train":
        setattr(cfg.train, name, value)
        return
    match = re.fullmatch(r"(\w+)\[(\d+)\]", name)
    if match:
        values = list(cfg.model.params[match.group(1)])
        values[int(match.group(2))] = value
        cfg.model.params[match.group(1)] = values
    else:
        cfg.model.params[name] = value


def _example(value: Any) -> str:
    """按当前值生成取值示例：整数取一半、本身、两倍；小数取十分之一、本身、十倍。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, int):
        return f"{max(1, value // 2)}, {value}, {value * 2}"
    return f"{value / 10:g}, {value:g}, {value * 10:g}" if value else "0"


def _fmt(value: Any) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def _describe(known: dict[str, SweepAxis], keys: list[str], combo: tuple) -> str:
    return "，".join(f"{known[k].label}={_fmt(v)}" for k, v in zip(keys, combo))


def _unique(values: list) -> list:
    return [v for i, v in enumerate(values) if v not in values[:i]]


def _write_results(sweep_dir: Path, rows: dict[int, dict], keys: list[str]) -> None:
    columns = ["trial", "name", *keys, "status", "error", "best_epoch", "epochs_run", "seconds"]
    columns += [f"{split}_{key}" for split in ("val", "test", "train") for key in METRIC_KEYS]
    frame = pd.DataFrame([rows[i] for i in sorted(rows)]).reindex(columns=columns)
    frame.to_csv(sweep_dir / RESULTS_FILE, index=False, encoding="utf-8-sig")
