import math
import multiprocessing as mp
import os
import signal
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from carinout.training import plots, sweep
from carinout.training.cli import main as cli_main
from carinout.training.config import DataConfig, ExperimentConfig, ModelConfig, OptimConfig
from carinout.training.runs import load_run
from carinout.training.sweep import (
    RESULTS_FILE,
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_STOPPED,
    available_axes,
    best_trial,
    expand_grid,
    is_sweep_dir,
    load_sweep,
    rank_results,
    resolve_axis_key,
    run_sweep,
)
from synthetic import make_dataset


@pytest.fixture
def base(tmp_path: Path) -> ExperimentConfig:
    dataset = make_dataset(tmp_path / "_dataset.csv", n_inside=6, n_outside=6, rows_per_point=15)
    return ExperimentConfig(
        data=DataConfig(datasets=[str(dataset)]),
        model=ModelConfig("mlp", {"hidden_sizes": [16, 8]}),
        train=OptimConfig(epochs=3, batch_size=32, learning_rate=0.01),
        output_dir=str(tmp_path / "runs"),
    )


def axis(base: ExperimentConfig, key: str):
    return {a.key: a for a in available_axes(base)}[key]


def test_axes_and_aliases(base: ExperimentConfig):
    keys = [a.key for a in available_axes(base)]
    assert keys == ["train.batch_size", "train.learning_rate", "train.l2_lambda", "train.epochs",
                    "train.class_weight", "model.hidden_sizes[0]", "model.hidden_sizes[1]", "model.dropout"]
    assert axis(base, "model.hidden_sizes[1]").base == 8
    for name, key in [("bs", "train.batch_size"), ("batch-size", "train.batch_size"), ("lr", "train.learning_rate"),
                      ("l2", "train.l2_lambda"), ("epochs", "train.epochs"), ("hidden1", "model.hidden_sizes[0]"),
                      ("h2", "model.hidden_sizes[1]"), ("model.hidden_sizes[0]", "model.hidden_sizes[0]"),
                      ("dropout", "model.dropout"), ("cw", "train.class_weight"),
                      ("class-weight", "train.class_weight")]:
        assert resolve_axis_key(base, name) == key
    with pytest.raises(ValueError, match="不能遍历"):
        resolve_axis_key(base, "hidden3")  # 基础配置只有两层隐藏层


def test_parse_values(base: ExperimentConfig):
    assert axis(base, "train.batch_size").parse("32, 64，128 64") == [32, 64, 128]
    assert axis(base, "train.learning_rate").parse("1e-3; 3e-4") == [0.001, 0.0003]
    assert axis(base, "model.hidden_sizes[0]").parse("") == []
    assert axis(base, "model.dropout").parse("0, 0.2，0.5") == [0.0, 0.2, 0.5]
    assert axis(base, "train.class_weight").parse("none, 按样本数平衡, balanced") == ["none", "balanced"]
    for key, text, message in [("train.batch_size", "32.5", "不是整数"), ("train.learning_rate", "abc", "不是数字"),
                               ("model.hidden_sizes[0]", "0", "不能小于 1"), ("model.dropout", "1", "不能大于 0.9"),
                               ("train.class_weight", "auto", "可选")]:
        with pytest.raises(ValueError, match=message):
            axis(base, key).parse(text)


def test_expand_grid(base: ExperimentConfig):
    trials = expand_grid(base, {"train.batch_size": [16, 32], "model.hidden_sizes[0]": [4, 8, 12],
                                "train.learning_rate": []})
    assert len(trials) == 6
    assert [t.name for t in trials[:2]] == ["001_bs16_h1-4", "002_bs16_h1-8"]
    last = trials[-1].config
    assert last.train.batch_size == 32 and last.model.params["hidden_sizes"] == [12, 8]
    assert last.train.learning_rate == base.train.learning_rate
    assert base.model.params == {"hidden_sizes": [16, 8]}  # 基础配置不被修改
    weighted = expand_grid(base, {"train.class_weight": ["none", "balanced"]})
    assert [t.name for t in weighted] == ["001_cw-none", "002_cw-balanced"]
    assert weighted[1].config.train.class_weight == "balanced"
    dropout = expand_grid(base, {"model.dropout": [0.0, 0.3]})
    assert [t.name for t in dropout] == ["001_do-0", "002_do-0.3"]
    assert dropout[1].config.model.params == {"hidden_sizes": [16, 8], "dropout": 0.3}
    (only,) = expand_grid(base, {})
    assert only.name == "001_base" and only.overrides == {}
    with pytest.raises(ValueError, match="batch size=0"):
        expand_grid(base, {"train.batch_size": [0]})


def test_rank_results():
    df = pd.DataFrame({"trial": [1, 2, 3, 4], "val_accuracy": [0.9, math.nan, 0.95, 0.95],
                       "val_ce": [0.3, 0.1, 0.25, 0.2]})
    assert rank_results(df, "val_accuracy")["trial"].tolist() == [4, 3, 1, 2]  # 并列时交叉熵低的在前，NaN 最后
    assert rank_results(df, "val_ce")["trial"].tolist() == [2, 4, 3, 1]
    assert best_trial(df, "val_accuracy")["trial"] == 4
    assert best_trial(df.assign(val_auc=math.nan), "val_auc") is None


def test_run_sweep_in_parallel(base: ExperimentConfig):
    trials = expand_grid(base, {"train.batch_size": [16, 32], "model.hidden_sizes[1]": [4, 8]})
    events = []
    path = run_sweep(base, trials, workers=2, rank_by="val_ce", on_event=events.append)

    assert is_sweep_dir(path) and (path / RESULTS_FILE).is_file()
    kinds = [e[0] for e in events]
    assert kinds[0] == "ready" and kinds.count("start") == 4 and kinds.count("done") == 4
    assert kinds.count("epoch") == 4 * base.train.epochs

    record = load_sweep(path)
    assert record.axes == {"train.batch_size": [16, 32], "model.hidden_sizes[1]": [4, 8]}
    assert record.rank_by == "val_ce" and record.base.data.features  # 保存实际使用的特征
    results = record.results
    assert results["status"].eq(STATUS_DONE).all() and len(results) == 4
    assert results[["val_ce", "val_accuracy", "test_auc", "train_accuracy", "seconds"]].notna().all().all()
    row = results.set_index("trial").loc[4]
    assert (row["train.batch_size"], row["model.hidden_sizes[1]"]) == (32, 8)
    run = load_run(path / row["name"])
    assert run.config.train.batch_size == 32 and run.config.model.params == {"hidden_sizes": [16, 8], "dropout": 0.0}
    assert (path / row["name"] / "loss.png").is_file()
    assert record.trial_config(row).model.params == {"hidden_sizes": [16, 8], "dropout": 0.0}

    # 与单独训练同一组参数的结果一致（各进程中的训练可复现）
    from carinout.training.data import prepare_data
    from carinout.training.trainer import fit

    data = prepare_data(base.data, base.train.seed)
    single = fit(trials[3].config, data)
    assert np.isclose(single.evals["val"].loss, row["val_ce"], atol=1e-5)


def test_stop_and_failure(base: ExperimentConfig):
    base.train.epochs = 200
    trials = expand_grid(base, {"train.batch_size": [8, 16, 32, 64]})
    path = run_sweep(base, trials, workers=1, should_stop=lambda: True)
    statuses = set(load_sweep(path).results["status"])
    assert statuses <= {STATUS_STOPPED, STATUS_CANCELLED} and STATUS_CANCELLED in statuses

    if torch.cuda.is_available():
        pytest.skip("有 GPU 时无法构造设备错误")
    base.train.epochs = 1
    base.train.device = "cuda"
    path = run_sweep(base, expand_grid(base, {"train.batch_size": [16, 32]}), workers=2)
    results = load_sweep(path).results
    assert results["status"].eq(STATUS_FAILED).all() and results["error"].str.contains("CUDA").all()


def test_crashed_worker_only_affects_its_trial(base: ExperimentConfig):
    """一个训练进程意外退出（如内存不足被系统结束）：它的组合重试一次，其他组合照常完成，不会全部失败。"""
    base.train.epochs = 20
    trials = expand_grid(base, {"train.batch_size": [16, 32, 64]})
    logs, killed = [], []

    def on_event(event):
        if event[0] == "epoch" and not killed:  # 两个进程都在训练时杀掉其中一个
            victim = mp.active_children()[0]
            victim.kill()
            killed.append(victim.pid)

    path = run_sweep(base, trials, workers=2, on_event=on_event, log=logs.append)
    assert killed and load_sweep(path).results["status"].eq(STATUS_DONE).all()
    assert any("稍后重试" in m for m in logs) and any("进程数减为 1" in m for m in logs)
    assert not mp.active_children()


def test_worker_startup_failure_is_reported(base: ExperimentConfig, monkeypatch):
    """子进程启动失败时显示真正的原因，有限次重启后结束，不会卡住。"""
    monkeypatch.setattr(sweep.pickle, "dump", lambda obj, fh, **kw: fh.write(b"broken"))
    monkeypatch.setattr(sweep, "MAX_RESTARTS", 1)
    logs = []
    path = run_sweep(base, expand_grid(base, {"train.batch_size": [16, 32]}), workers=2, log=logs.append)
    results = load_sweep(path).results
    assert results["status"].eq(STATUS_FAILED).all()
    assert results["error"].str.contains("UnpicklingError").all()
    assert any("训练进程启动失败" in m for m in logs)


@pytest.mark.skipif(sys.platform == "win32", reason="需要 SIGSTOP")
def test_stop_does_not_hang_on_unresponsive_worker(base: ExperimentConfig, monkeypatch):
    """停止时某个进程没有响应：超时后强制结束，遍历照常返回。"""
    monkeypatch.setattr(sweep, "STOP_TIMEOUT", 3.0)
    base.train.epochs = 500
    state = {"stop": False}

    def on_event(event):
        if event[0] == "epoch" and not state["stop"]:
            os.kill(mp.active_children()[0].pid, signal.SIGSTOP)
            state["stop"] = True

    path = run_sweep(base, expand_grid(base, {"train.batch_size": [16, 32]}), workers=2, on_event=on_event,
                     should_stop=lambda: state["stop"])
    results = load_sweep(path).results
    assert results["status"].eq(STATUS_STOPPED).all()
    assert results["error"].fillna("").str.contains("停止超时").any()
    assert not mp.active_children()


def test_cli_sweep(base: ExperimentConfig, tmp_path: Path, capsys):
    out = tmp_path / "cli"
    val = tmp_path / "val_day"
    val.mkdir()
    make_dataset(val / "_dataset.csv", n_inside=3, n_outside=3, rows_per_point=10, seed=9)
    args = ["--dataset", base.data.datasets[0], "--epochs", "2", "--hidden", "8,4", "--output-dir", str(out),
            "--sweep", "lr=0.01,0.001", "--sweep", "hidden2=2,4", "--sweep", "dropout=0,0.2", "--workers", "2",
            "--rank-by", "val_auc",
            "--val-dataset", str(val)]
    assert cli_main(args) == 0
    (path,) = out.iterdir()
    record = load_sweep(path)
    assert record.base.data.val_datasets == [str(val / "_dataset.csv")]  # 记录实际使用的文件
    results = record.results
    assert results["val_auc"].notna().all()
    assert load_run(path / results["name"].iloc[0]).predictions.query("split == 'val'")["source"].eq("val_day").all()
    combos = sorted(zip(results["train.learning_rate"], results["model.hidden_sizes[1]"], results["model.dropout"]))
    assert combos == [(lr, h, d) for lr in (0.001, 0.01) for h in (2, 4) for d in (0.0, 0.2)]
    assert "按验证集 AUC排序" in capsys.readouterr().out
    assert cli_main(["--dataset", base.data.datasets[0], "--sweep", "hidden5=8"]) == 1
    assert cli_main(["--dataset", base.data.datasets[0], "--sweep", "bs"]) == 1


def test_draw_sweep():
    results = pd.DataFrame({"train.batch_size": [16, 16, 32, 32], "model.hidden_sizes[0]": [4, 8, 4, 8],
                            "val_accuracy": [0.8, 0.85, 0.9, math.nan]})
    axes = [("train.batch_size", "batch size"), ("model.hidden_sizes[0]", "第 1 层")]
    fig = plots.new_figure(8, 3)
    plots.draw_sweep(fig, results, axes, "val_accuracy", "验证集准确率")
    assert len([ax for ax in fig.axes if ax.get_visible()]) == 2
    plots.draw_sweep(fig, results.iloc[:0], axes, "val_accuracy", "验证集准确率")  # 还没有结果
    assert len(fig.axes) == 1
