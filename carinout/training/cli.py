"""命令行训练：python train.py --config 配置.json [覆盖参数...]
参数遍历：python train.py --config 配置.json --sweep bs=32,64 --sweep lr=1e-3,3e-4 --sweep hidden1=32,64 --workers 8

不带任何参数运行 train.py 时打开图形界面；带参数时在命令行训练，结果与界面训练相同（保存在 output_dir 下）。
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from .config import ExperimentConfig
from .data import SPLIT_NAMES, feature_candidates, load_table, resolve_datasets
from .metrics import METRIC_NAMES
from .models import available_models, get_spec
from .sweep import RANK_METRICS


def build_parser() -> argparse.ArgumentParser:
    models = "、".join(f"{s.name}（{s.label}）" for s in available_models())
    p = argparse.ArgumentParser(
        prog="python train.py",
        description="车内外识别模型训练。不带参数运行时打开图形界面。命令行参数会覆盖配置文件中的同名项。",
    )
    p.add_argument("--config", help="训练配置 JSON（可在界面中“保存配置”得到）")
    p.add_argument(
        "--dataset",
        action="append",
        metavar="文件夹",
        help="训练数据，可重复：含 _dataset.csv 的文件夹、其上级文件夹（取各子文件夹）或 csv 文件",
    )
    p.add_argument(
        "--val-dataset",
        action="append",
        metavar="文件夹",
        help="独立验证集，可重复：含 _dataset.csv 的文件夹、其上级文件夹（取各子文件夹）或 csv 文件",
    )
    p.add_argument("--test-dataset", action="append", metavar="文件夹", help="独立测试集，可重复，格式同 --val-dataset")
    p.add_argument("--features", help="输入特征列，逗号分隔；不指定时使用全部非常数的数值列")
    p.add_argument("--list-features", action="store_true", help="列出数据集中可用的特征列后退出")
    p.add_argument("--model", help=f"模型：{models}")
    p.add_argument("--hidden", help="MLP 隐藏层节点数，如 64,32（等同 --param hidden_sizes=64,32）")
    p.add_argument("--param", action="append", default=[], metavar="名称=值", help="模型参数，可重复")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float, help="learning rate")
    p.add_argument("--epochs", type=int)
    p.add_argument("--l2", type=float, help="L2 系数 λ")
    p.add_argument("--optimizer", choices=["adam", "sgd"])
    p.add_argument("--class-weight", choices=["none", "balanced"])
    p.add_argument("--early-stopping", type=int, metavar="N", help="验证集交叉熵 N 个 epoch 无改善即停止，0 不停止")
    p.add_argument("--missing", choices=["drop", "mean", "constant"], help="缺失值处理")
    p.add_argument("--split", choices=["point", "row"], help="按点位整组划分 / 按行随机划分")
    p.add_argument("--val-ratio", type=float)
    p.add_argument("--test-ratio", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"])
    p.add_argument("--output-dir", help="训练结果保存位置（默认 runs）")
    p.add_argument("--name", help="本次训练名称，作为结果文件夹名的后缀")
    g = p.add_argument_group("参数遍历（任一参数给出多个取值时，训练所有组合）")
    g.add_argument(
        "--sweep",
        action="append",
        default=[],
        metavar="参数=值1,值2",
        help="可重复。参数：bs（batch size）、lr、l2、epochs、hidden1 / hidden2 …（第 N 层隐藏层节点数），"
        "如 --sweep bs=32,64 --sweep hidden1=32,64,128",
    )
    g.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="并行训练的进程数（默认 CPU 核数）")
    g.add_argument("--rank-by", choices=list(RANK_METRICS), default="val_accuracy", help="挑选最佳组合的验证集指标")
    return p


def apply_args(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    d, t = cfg.data, cfg.train
    simple = {
        "dataset": (d, "datasets"),
        "val_dataset": (d, "val_datasets"),
        "test_dataset": (d, "test_datasets"),
        "missing": (d, "missing"),
        "split": (d, "split"),
        "val_ratio": (d, "val_ratio"),
        "test_ratio": (d, "test_ratio"),
        "batch_size": (t, "batch_size"),
        "lr": (t, "learning_rate"),
        "epochs": (t, "epochs"),
        "l2": (t, "l2_lambda"),
        "optimizer": (t, "optimizer"),
        "class_weight": (t, "class_weight"),
        "early_stopping": (t, "early_stopping"),
        "seed": (t, "seed"),
        "device": (t, "device"),
        "output_dir": (cfg, "output_dir"),
        "name": (cfg, "run_name"),
    }
    for arg, (target, attr) in simple.items():
        value = getattr(args, arg)
        if value is not None:
            setattr(target, attr, value)
    if args.features is not None:
        d.features = [f.strip() for f in args.features.replace("，", ",").split(",") if f.strip()]
    if args.model and args.model != cfg.model.name:
        cfg.model.name, cfg.model.params = args.model, {}
    params = dict(cfg.model.params)
    if args.hidden:
        params["hidden_sizes"] = args.hidden
    for item in args.param:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--param 应为 名称=值 的形式：{item}")
        params[key.strip()] = value.strip()
    cfg.model.params = get_spec(cfg.model.name).resolve_params(params)
    return cfg


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = ExperimentConfig.load(args.config) if args.config else ExperimentConfig()
        cfg = apply_args(cfg, args)
        if args.list_features:
            return _list_features(cfg)
        cfg.validate()
        return _sweep(cfg, args) if args.sweep else _train(cfg)
    except (OSError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


def _list_features(cfg: ExperimentConfig) -> int:
    if not cfg.data.datasets:
        raise ValueError("请用 --dataset 指定训练数据")
    tables = [load_table(path) for path in resolve_datasets(cfg.data.datasets)]
    for info in feature_candidates(tables, cfg.data.label_column):
        flags = ("  常数" if info.constant else "") + (f"  缺失 {info.missing_ratio:.1%}" if info.missing_ratio else "")
        print(info.name + flags)
    return 0


def _train(cfg: ExperimentConfig) -> int:
    from .data import prepare_data
    from .runs import new_run_dir, per_source_metrics, predictions_frame, save_run
    from .trainer import fit

    data = prepare_data(cfg.data, cfg.train.seed, log=print)
    every = max(1, cfg.train.epochs // 50)

    def on_epoch(r: dict) -> None:
        if r["epoch"] % every and r["epoch"] not in (1, cfg.train.epochs):
            return
        text = (
            f"epoch {r['epoch']:>4}/{cfg.train.epochs}  loss {r['train_loss']:.4f}"
            f"（CE {r['train_ce']:.4f} + L2 {r['l2']:.4f}）  train_acc {r['train_accuracy']:.4f}"
        )
        if not math.isnan(r["val_ce"]):
            text += f"  val_loss {r['val_loss']:.4f}  val_acc {r['val_accuracy']:.4f}  val_auc {r['val_auc']:.4f}"
        print(text, flush=True)

    print(f"特征（{len(data.features)}）：{', '.join(data.features)}")
    with _ctrl_c_stops("停止训练并保存结果") as stopped:
        result = fit(cfg, data, on_epoch, stopped, log=print)
    run_dir = new_run_dir(cfg)
    save_run(run_dir, cfg, data, result)

    print(f"\n最佳 epoch：{result.best_epoch}")
    print(f"{'':<16}{'样本数':>8}{'交叉熵':>10}" + "".join(f"{name:>10}" for name in METRIC_NAMES.values()))
    sources = per_source_metrics(predictions_frame(data, result))
    for name, ev in result.evals.items():
        values = "".join(f"{ev.metrics[k]:>10.4f}" for k in METRIC_NAMES)
        print(f"{SPLIT_NAMES[name]:<16}{ev.metrics['n']:>8}{ev.loss:>10.4f}{values}")
        for source, m in sources.get(name, {}).items():  # 来自多个文件夹时逐个列出
            values = "".join(f"{_num(m[k]):>10}" for k in METRIC_NAMES)
            print(f"  {source:<14}{m['n']:>8}{'':>10}{values}")
    print(f"\n训练结果已保存到 {run_dir}")
    return 0


def _sweep(cfg: ExperimentConfig, args: argparse.Namespace) -> int:
    from .sweep import RESULTS_FILE, available_axes, expand_grid, load_sweep, rank_results, resolve_axis_key, run_sweep

    axes_by_key = {a.key: a for a in available_axes(cfg)}
    axes: dict[str, list] = {}
    for item in args.sweep:
        name, sep, values = item.partition("=")
        if not sep:
            raise ValueError(f"--sweep 应为 参数=值1,值2 的形式：{item}")
        key = resolve_axis_key(cfg, name)
        axes[key] = axes_by_key[key].parse(values)
        if not axes[key]:
            raise ValueError(f"--sweep {name} 没有给出取值")
    trials = expand_grid(cfg, axes)
    print("参数遍历：" + "；".join(f"{axes_by_key[k].label} = {', '.join(map(str, v))}" for k, v in axes.items()))
    finished = 0

    def on_event(event: tuple) -> None:
        nonlocal finished
        if event[0] != "done":
            return
        finished += 1
        row = event[2]
        text = f"[{finished}/{len(trials)}] {row['name']}  {row['status']}"
        if row.get("error"):
            text += f"：{row['error']}"
        elif not math.isnan(row.get("val_ce", math.nan)):
            text += f"  val_acc {row['val_accuracy']:.4f}  val_auc {row['val_auc']:.4f}  val_ce {row['val_ce']:.4f}"
        if "seconds" in row:
            text += f"  ({row['seconds']:.1f}s)"
        print(text, flush=True)

    with _ctrl_c_stops("停止遍历：正在训练的组合会保存当前结果，其余组合不再运行") as stopped:
        path = run_sweep(cfg, trials, args.workers, args.rank_by, on_event, stopped, print)

    ranked = rank_results(load_sweep(path).results, args.rank_by)
    columns = ["name", "status", "val_ce", "val_accuracy", "val_f1", "val_auc", "test_accuracy", "test_auc"]
    print(f"\n按{RANK_METRICS[args.rank_by]}排序的前 10 组：")
    print(ranked[columns].head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n全部结果：{path / RESULTS_FILE}")
    return 0


@contextmanager
def _ctrl_c_stops(action: str) -> Iterator[Callable[[], bool]]:
    """第一次 Ctrl+C：在当前 batch 后停止，照常评估并保存结果；第二次：立即退出。"""
    stop = False

    def on_sigint(_signum, _frame) -> None:
        nonlocal stop
        if stop:
            raise KeyboardInterrupt
        stop = True
        print(f"\n收到 Ctrl+C，{action}（再按一次立即退出）", flush=True)

    previous = signal.signal(signal.SIGINT, on_sigint)
    try:
        yield lambda: stop
    finally:
        signal.signal(signal.SIGINT, previous)


def _num(value) -> str:
    return "—" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.4f}"
