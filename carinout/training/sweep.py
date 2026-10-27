"""参数遍历：对若干参数的候选值做网格组合，多进程并行训练，汇总比较各组合在验证集上的表现。

用进程而不是线程并行：小型 MLP 每个 batch 的耗时主要是 Python 代码，线程受 GIL 限制几乎不能同时运行。
数据只准备一次（读取、划分、标准化），写到临时文件，各进程启动时读入，之后依次训练分配给它的组合。

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
import queue as queue_module
import re
import signal
import sys
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .config import ExperimentConfig, OptimConfig
from .data import SPLITS, PreparedData, prepare_data
from .models import get_spec
from .models.base import KIND_CHOICE, KIND_FLOAT, KIND_INT, KIND_INT_LIST
from .runs import new_run_dir, save_run
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

# 用来比较各组合的指标；交叉熵越低越好，其余越高越好
RANK_METRICS = {
    "val_accuracy": "验证集准确率",
    "val_f1": "验证集 F1",
    "val_auc": "验证集 AUC",
    "val_ce": "验证集交叉熵",
}
LOWER_IS_BETTER = {"val_ce"}
METRIC_KEYS = ("ce", "accuracy", "precision", "recall", "f1", "auc")

# 可遍历的训练参数：配置项 → (界面名称, 组合名称中的缩写)
TRAIN_AXES = {
    "batch_size": ("batch size", "bs"),
    "learning_rate": ("learning rate", "lr"),
    "l2_lambda": ("L2 系数 λ", "l2"),
    "epochs": ("epoch 数", "ep"),
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
    for name, (label, short) in TRAIN_AXES.items():
        kind = type(getattr(defaults, name))
        axes.append(SweepAxis(f"train.{name}", label, short, getattr(base.train, name), _number_parser(kind)))
    spec = get_spec(base.model.name)
    params = spec.resolve_params(base.model.params)
    for p in spec.params:
        short = p.short or p.name
        if p.kind == KIND_INT_LIST:
            for i, value in enumerate(params[p.name]):
                parse = _number_parser(int)
                axes.append(SweepAxis(f"model.{p.name}[{i}]", f"第 {i + 1} 层{p.label}", f"{short}{i + 1}-", value,
                                      lambda text, p=p, parse=parse: p.parse([parse(text)])[0]))
        elif p.kind in (KIND_INT, KIND_FLOAT, KIND_CHOICE):
            axes.append(SweepAxis(f"model.{p.name}", p.label, f"{short}-", params[p.name], p.parse))
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
    # spawn：Windows 只支持这种方式；Linux 上也避免 fork 带走界面和 torch 的线程状态
    ctx = mp.get_context("spawn")
    events = ctx.Queue()
    stop = ctx.Event()
    # 数据经文件传给子进程：直接放进 initargs 时，父进程要等每个子进程导入完模块才能把数据写进管道，进程只能逐个启动
    fd, data_file = tempfile.mkstemp(prefix="carinout_sweep_", suffix=".pkl")
    with os.fdopen(fd, "wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    try:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=ctx, initializer=_init_worker, initargs=(data_file, events, stop, threads)
        ) as pool:
            futures = {
                pool.submit(_run_trial, t.index, t.config.to_dict(), str(sweep_dir / t.name)): t for t in trials
            }
            pending = set(futures)
            while pending:
                done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                _drain(events, on_event)
                for future in done:
                    rows[futures[future].index].update(_outcome(future))
                    on_event(("done", futures[future].index, dict(rows[futures[future].index])))
                if done:
                    _write_results(sweep_dir, rows, keys)
                if should_stop() and not stop.is_set():
                    stop.set()
                    for future in pending:
                        future.cancel()
                    log("正在停止：正在训练的组合会保存当前结果，其余组合不再运行")
    finally:
        os.unlink(data_file)
    _drain(events, on_event)
    _write_results(sweep_dir, rows, keys)
    counts = pd.Series([r["status"] for r in rows.values()]).value_counts()
    summary = "，".join(f"{status} {counts[status]} 组" for status in FINAL_STATUSES if status in counts)
    log(f"参数遍历结束（共 {len(trials)} 组：{summary}），结果保存在 {sweep_dir}")
    return sweep_dir


def _outcome(future) -> dict:
    if future.cancelled():
        return {"status": STATUS_CANCELLED}
    exc = future.exception()
    if exc is not None:  # 包括子进程意外退出（如内存不足）
        return {"status": STATUS_FAILED, "error": str(exc) or repr(exc)}
    return future.result()


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


# ---------------------------------------------------------------------------- 子进程
_DATA: PreparedData | None = None
_EVENTS = None
_STOP = None


def _init_worker(data_file: str, events, stop, threads: int) -> None:
    global _DATA, _EVENTS, _STOP
    # 终端里按 Ctrl+C 时子进程也会收到，交给主进程通过 stop 统一处理（保存已训练的结果）
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(threads)
    with open(data_file, "rb") as fh:
        _DATA = pickle.load(fh)
    _EVENTS, _STOP = events, stop


def _run_trial(index: int, cfg_dict: dict, run_dir: str) -> dict:
    if _STOP.is_set():
        return {"status": STATUS_CANCELLED}
    cfg = ExperimentConfig.from_dict(cfg_dict)
    _EVENTS.put(("start", index))
    started = time.perf_counter()
    result = fit(cfg, _DATA, lambda record: _EVENTS.put(("epoch", index, record)), _STOP.is_set)
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    save_run(path, cfg, _DATA, result)
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


def _fmt(value: Any) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def _describe(known: dict[str, SweepAxis], keys: list[str], combo: tuple) -> str:
    return "，".join(f"{known[k].label}={_fmt(v)}" for k, v in zip(keys, combo))


def _unique(values: list) -> list:
    return [v for i, v in enumerate(values) if v not in values[:i]]


def _drain(events, on_event: Callable[[tuple], None]) -> None:
    while True:
        try:
            event = events.get_nowait()
        except queue_module.Empty:
            return
        on_event(event)


def _write_results(sweep_dir: Path, rows: dict[int, dict], keys: list[str]) -> None:
    columns = ["trial", "name", *keys, "status", "error", "best_epoch", "epochs_run", "seconds"]
    columns += [f"{split}_{key}" for split in ("val", "test", "train") for key in METRIC_KEYS]
    frame = pd.DataFrame([rows[i] for i in sorted(rows)]).reindex(columns=columns)
    frame.to_csv(sweep_dir / RESULTS_FILE, index=False, encoding="utf-8-sig")
