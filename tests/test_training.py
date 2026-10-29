import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from carinout.training import models
from carinout.training.cli import main as cli_main
from carinout.training.config import DataConfig, ExperimentConfig, ModelConfig, OptimConfig
from carinout.training.data import load_table, prepare_data
from carinout.training.losses import Objective, class_weights, l2_penalty
from carinout.training.metrics import binary_metrics
from carinout.training.models.base import KIND_FLOAT, ModelSpec, ParamSpec
from carinout.training.runs import (
    MODEL_FILE,
    Predictor,
    load_run,
    new_run_dir,
    per_point_accuracy,
    save_run,
)
from carinout.training.trainer import HISTORY_COLUMNS, fit
from synthetic import make_dataset


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    return make_dataset(tmp_path / "_dataset.csv", n_inside=10, n_outside=10, rows_per_point=20)


def config(dataset: Path, tmp_path: Path, **train) -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(datasets=[str(dataset)]),
        model=ModelConfig("mlp", {"hidden_sizes": [16, 8]}),
        train=OptimConfig(**{"epochs": 15, "batch_size": 32, "learning_rate": 0.01, **train}),
        output_dir=str(tmp_path / "runs"),
    )


# ---------------------------------------------------------------------------- 模型
def test_mlp_has_three_linear_layers_with_relu():
    model = models.build_model("mlp", 10, 2)
    layers = list(model.net)
    assert [type(m) for m in layers] == [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear]
    linear = [(m.in_features, m.out_features) for m in layers if isinstance(m, nn.Linear)]
    assert linear == [(10, 64), (64, 32), (32, 2)]
    assert model(torch.zeros(5, 10)).shape == (5, 2)
    custom = models.build_model("mlp", 4, 2, {"hidden_sizes": "128, 16"})
    assert [m.out_features for m in custom.net if isinstance(m, nn.Linear)] == [128, 16, 2]


def test_mlp_dropout():
    model = models.build_model("mlp", 6, 2, {"hidden_sizes": [32, 16], "dropout": 0.5})
    types = [type(m) for m in model.net]
    assert types == [nn.Linear, nn.ReLU, nn.Dropout, nn.Linear, nn.ReLU, nn.Dropout, nn.Linear]
    assert all(m.p == 0.5 for m in model.net if isinstance(m, nn.Dropout))
    x = torch.randn(64, 6)
    model.train()
    assert not torch.equal(model(x), model(x))  # 训练时随机置 0
    model.eval()
    assert torch.equal(model(x), model(x))  # 评估时不生效
    # dropout 为 0 时结构与旧版本相同，旧的参数文件仍能加载
    old_keys = set(models.build_model("mlp", 6, 2, {"hidden_sizes": [32, 16]}).state_dict())
    assert old_keys == {"net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias", "net.4.weight", "net.4.bias"}


@pytest.mark.parametrize("value", ["", "a,2", "0,8", [8.5], 16])
def test_invalid_mlp_params(value):
    with pytest.raises(ValueError):
        models.get_spec("mlp").resolve_params({"hidden_sizes": value})


@pytest.mark.parametrize("value", [-0.1, 0.95, 1, "abc"])
def test_invalid_dropout(value):
    with pytest.raises(ValueError, match="Dropout"):
        models.get_spec("mlp").resolve_params({"dropout": value})


def test_unknown_model_and_param():
    with pytest.raises(ValueError, match="未知的模型"):
        models.get_spec("svm")
    with pytest.raises(ValueError, match="没有参数"):
        models.get_spec("mlp").resolve_params({"batch_norm": True})


def test_registered_model_trains_without_other_changes(dataset: Path, tmp_path: Path):
    """新模型只需注册 ModelSpec，训练、保存、加载流程都不用改。"""

    class Linear(nn.Module):
        def __init__(self, in_features, num_classes, scale):
            super().__init__()
            self.fc = nn.Linear(in_features, num_classes)
            self.scale = scale

        def forward(self, x):
            return self.fc(x) * self.scale

    spec = ModelSpec("linear_test", "线性模型", "测试用", (ParamSpec("scale", "缩放", KIND_FLOAT, 1.0),),
                     lambda i, c, scale: Linear(i, c, scale))
    models.register(spec)
    try:
        cfg = config(dataset, tmp_path, epochs=3)
        cfg.model = ModelConfig("linear_test", {"scale": "2"})
        data = prepare_data(cfg.data, cfg.train.seed)
        result = fit(cfg, data)
        run = new_run_dir(cfg)
        save_run(run, cfg, data, result, figures=False)
        assert load_run(run).config.model.params == {"scale": 2.0}
        assert Predictor.load(run).predict_proba(load_table(dataset)).shape == (400,)
    finally:
        models._REGISTRY.pop("linear_test")


# ---------------------------------------------------------------------------- 损失
def test_l2_penalty_only_weights():
    model = models.build_model("mlp", 3, 2, {"hidden_sizes": [4]})
    expected = sum((m.weight**2).sum() for m in model.net if isinstance(m, nn.Linear))
    assert torch.isclose(l2_penalty(model), expected)
    logits, target = model(torch.randn(6, 3)), torch.tensor([0, 1, 0, 1, 1, 1])
    total, ce, l2 = Objective(0.5)(model, logits, target)
    assert torch.isclose(total, ce + 0.5 * expected) and torch.isclose(l2, 0.5 * expected)
    assert torch.allclose(class_weights([0, 0, 0, 1], 2, "balanced"), torch.tensor([4 / 6, 2.0]))
    assert class_weights([0, 1], 2, "none") is None


# ---------------------------------------------------------------------------- 训练
def test_fit_learns_and_records_history(dataset: Path, tmp_path: Path):
    cfg = config(dataset, tmp_path, l2_lambda=1e-3)
    data = prepare_data(cfg.data, cfg.train.seed)
    seen = []
    result = fit(cfg, data, on_epoch=seen.append)
    assert len(result.history) == len(seen) == 15
    assert list(result.history[0]) == list(HISTORY_COLUMNS)
    first, last = result.history[0], result.history[-1]
    assert last["train_loss"] < first["train_loss"]
    assert np.isclose(last["train_loss"], last["train_ce"] + last["l2"]) and last["l2"] > 0
    assert result.evals["val"].metrics["accuracy"] > 0.9
    best = min(result.history, key=lambda r: r["val_ce"])
    assert result.best_epoch == best["epoch"]
    assert np.isclose(result.evals["val"].loss, best["val_ce"], atol=1e-5)  # 恢复了最佳 epoch 的参数
    assert set(result.evals) == {"train", "val", "test"}


def test_fit_is_reproducible(dataset: Path, tmp_path: Path):
    cfg = config(dataset, tmp_path, epochs=3)
    data = prepare_data(cfg.data, cfg.train.seed)
    a, b = fit(cfg, data), fit(cfg, data)
    assert [r["train_loss"] for r in a.history] == [r["train_loss"] for r in b.history]


def test_stop_and_early_stopping(dataset: Path, tmp_path: Path):
    cfg = config(dataset, tmp_path, epochs=50)
    data = prepare_data(cfg.data, cfg.train.seed)
    history = []
    result = fit(cfg, data, on_epoch=history.append, should_stop=lambda: len(history) >= 2)
    assert result.stopped and len(result.history) == 2 and "test" in result.evals

    cfg = config(dataset, tmp_path, epochs=500, early_stopping=2, learning_rate=0.05)
    result = fit(cfg, data)
    assert result.early_stopped and len(result.history) == result.best_epoch + 2


def test_threshold_and_imbalance(dataset: Path, tmp_path: Path):
    """判定阈值用于验证集指标、预测结果和保存的模型；样本不平衡时日志提示。"""
    cfg = config(dataset, tmp_path, epochs=4, threshold=0.8)
    data = prepare_data(cfg.data, cfg.train.seed)
    logs = []
    result = fit(cfg, data, log=logs.append)
    val = result.evals["val"]
    assert val.metrics == binary_metrics(data.val.y, val.probs, 0.8)
    best = result.history[result.best_epoch - 1]  # 最终模型即最佳 epoch 的参数
    assert best["val_balanced_accuracy"] == pytest.approx(val.metrics["balanced_accuracy"])
    assert best["val_accuracy"] == pytest.approx(val.metrics["accuracy"])
    assert any("判定阈值" in m for m in logs)
    run_dir = new_run_dir(cfg)
    save_run(run_dir, cfg, data, result, figures=False)
    run = load_run(run_dir)
    assert run.metrics["threshold"] == 0.8 and run.config.train.threshold == 0.8
    pred = run.predictions
    assert (pred["pred"] == (pred["prob"] >= 0.8).astype(int)).all()
    assert Predictor.load(run_dir).threshold == 0.8
    # 旧模型文件没有 threshold，按 0.5
    checkpoint = torch.load(run_dir / MODEL_FILE, weights_only=True)
    del checkpoint["threshold"]
    assert Predictor(checkpoint).threshold == 0.5

    imbalanced = make_dataset(tmp_path / "imb.csv", n_inside=4, n_outside=16, rows_per_point=10)
    cfg = config(imbalanced, tmp_path, epochs=1)
    data = prepare_data(cfg.data, cfg.train.seed)
    logs = []
    fit(cfg, data, log=logs.append)
    assert any("样本不平衡" in m and "建议" in m for m in logs)
    cfg.train.class_weight = "balanced"
    logs = []
    fit(cfg, data, log=logs.append)
    assert any("已按样本数加权" in m for m in logs) and any(m.startswith("类别权重：车外") for m in logs)
    with pytest.raises(ValueError, match="判定阈值"):
        config(dataset, tmp_path, threshold=1.0).validate()


def test_without_validation_set(dataset: Path, tmp_path: Path):
    cfg = config(dataset, tmp_path, epochs=3)
    cfg.data.val_ratio = 0
    data = prepare_data(cfg.data, cfg.train.seed)
    result = fit(cfg, data)
    assert np.isnan(result.history[0]["val_ce"]) and "val" not in result.evals
    assert result.best_epoch == min(result.history, key=lambda r: r["train_loss"])["epoch"]


# ---------------------------------------------------------------------------- 结果保存
def test_save_and_load_run(dataset: Path, tmp_path: Path):
    cfg = config(dataset, tmp_path, epochs=5)
    cfg.model.params["dropout"] = 0.2
    cfg.data.missing = "mean"
    cfg.run_name = "h16 8"
    data = prepare_data(cfg.data, cfg.train.seed)
    result = fit(cfg, data)
    run_dir = new_run_dir(cfg)
    assert run_dir.name.endswith("_h16_8")
    save_run(run_dir, cfg, data, result)
    for name in ("config.json", "history.csv", "metrics.json", "predictions.csv", MODEL_FILE,
                 "loss.png", "metrics.png", "evaluation.png"):
        assert (run_dir / name).is_file(), name

    run = load_run(run_dir)
    assert run.config.data.features == data.features  # 保存的是实际使用的特征
    assert run.config.model.params == {"hidden_sizes": [16, 8], "dropout": 0.2}
    assert len(run.history) == 5 and run.metrics["best_epoch"] == result.best_epoch
    assert run.metrics["splits"]["test"]["n"] == len(data.test)
    assert len(run.predictions) == len(data.train) + len(data.val) + len(data.test)
    assert run.predictions["point"].map(type).eq(str).all()  # 点位按文本读回
    json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))  # 合法 JSON（NaN 写为 null）

    # 用 Predictor 推理与训练时的预测一致
    table = load_table(dataset)
    test = run.predictions[run.predictions["split"] == "test"]
    rows = table.merge(test[["scene", "point", "ts", "prob"]], on=["scene", "point", "ts"])
    prob = Predictor.load(run_dir).predict_proba(rows)
    assert np.allclose(prob, rows["prob"].to_numpy(), atol=1e-5)

    points = per_point_accuracy(run.predictions)
    assert {"split", "scene", "point", "label", "samples", "accuracy"} <= set(points.columns)
    assert points["accuracy"].is_monotonic_increasing


def test_run_with_validation_folders(tmp_path: Path):
    main = make_dataset(tmp_path / "train.csv", n_inside=8, n_outside=8, rows_per_point=15, seed=1)
    merged = tmp_path / "merged"
    for i, name in enumerate(["1024采集", "1025采集"]):
        (merged / name).mkdir(parents=True)
        make_dataset(merged / name / "_dataset.csv", n_inside=3, n_outside=3, rows_per_point=15, seed=i + 2)
    cfg = config(main, tmp_path, epochs=3)
    cfg.data.val_datasets = [str(merged)]  # 上级文件夹
    data = prepare_data(cfg.data, cfg.train.seed)
    result = fit(cfg, data)
    run_dir = new_run_dir(cfg)
    save_run(run_dir, cfg, data, result, figures=False)

    run = load_run(run_dir)
    assert run.config.data.val_datasets == [str(merged / n / "_dataset.csv") for n in ("1024采集", "1025采集")]
    sources = run.metrics["sources"]
    assert list(sources) == ["val"] and set(sources["val"]) == {"1024采集", "1025采集"}
    assert sum(m["n"] for m in sources["val"].values()) == run.metrics["splits"]["val"]["n"] == 180
    points = per_point_accuracy(run.predictions)
    assert set(points[points["split"] == "val"]["source"]) == {"1024采集", "1025采集"}
    assert set(run.predictions["source"]) == {"train", "1024采集", "1025采集"}

    # 旧版本保存的 predictions.csv 没有 source 列，也能读取和统计
    old = run.predictions.drop(columns="source")
    old.to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    assert "source" not in per_point_accuracy(load_run(run_dir).predictions)


def test_predictor_marks_missing_rows_when_trained_with_drop(dataset: Path, tmp_path: Path):
    cfg = config(dataset, tmp_path, epochs=2)
    data = prepare_data(cfg.data, cfg.train.seed)
    run_dir = new_run_dir(cfg)
    save_run(run_dir, cfg, data, fit(cfg, data), figures=False)
    df = load_table(dataset).head(3).copy()
    df.loc[1, data.features[0]] = ""
    prob = Predictor.load(run_dir / MODEL_FILE).predict_proba(df)
    assert np.isnan(prob[1]) and not np.isnan(prob[[0, 2]]).any()


# ---------------------------------------------------------------------------- 配置与命令行
def test_config_round_trip_and_validation(tmp_path: Path):
    cfg = ExperimentConfig()
    cfg.data.features = ["rssi_FL"]
    cfg.model.params = {"hidden_sizes": [8, 4]}
    cfg.save(tmp_path / "c.json")
    assert ExperimentConfig.load(tmp_path / "c.json") == cfg

    loaded = ExperimentConfig.from_dict({"train": {"learning_rate": "1e-3", "batch_size": 64.0}})
    assert loaded.train.learning_rate == 0.001 and loaded.train.batch_size == 64
    with pytest.raises(ValueError, match="未知的配置项：train.batchsize"):
        ExperimentConfig.from_dict({"train": {"batchsize": 64}})
    with pytest.raises(ValueError, match="类型不正确"):
        ExperimentConfig.from_dict({"train": {"batch_size": 1.5}})
    bad = ExperimentConfig()
    bad.data.val_ratio, bad.data.test_ratio = 0.6, 0.5
    with pytest.raises(ValueError, match="之和"):
        bad.validate()
    bad.data.val_datasets = ["D:/data/merged/1024采集"]  # 有独立验证集时验证集比例不生效
    bad.validate()

    # 旧版配置的训练数据 dataset、独立测试集 test_dataset 只有一个
    legacy = ExperimentConfig.from_dict({"data": {"dataset": "D:/a.csv", "test_dataset": "D:/b.csv"}})
    assert legacy.data.datasets == ["D:/a.csv"]
    assert legacy.data.test_datasets == ["D:/b.csv"] and legacy.data.val_datasets == []
    assert ExperimentConfig.from_dict({"data": {"test_dataset": ""}}).data.test_datasets == []


def test_cli(dataset: Path, tmp_path: Path, capsys):
    out = tmp_path / "cli_runs"
    args = ["--dataset", str(dataset), "--hidden", "8,4", "--batch-size", "16", "--lr", "0.01",
            "--epochs", "3", "--l2", "0.001", "--features", "rssi_FL,rssi_IF", "--output-dir", str(out)]
    assert cli_main(args + ["--dropout", "0.3"]) == 0
    (run_dir,) = out.iterdir()
    saved = load_run(run_dir).config
    assert saved.model.params == {"hidden_sizes": [8, 4], "dropout": 0.3}
    assert saved.train.batch_size == 16 and saved.train.learning_rate == 0.01 and saved.train.l2_lambda == 0.001
    assert saved.data.features == ["rssi_FL", "rssi_IF"]

    # 配置文件 + 命令行覆盖
    saved.save(tmp_path / "c.json")
    assert cli_main(["--config", str(tmp_path / "c.json"), "--epochs", "1", "--name", "again"]) == 0
    assert any(p.name.endswith("_again") for p in out.iterdir())

    # 多个训练数据文件夹
    for name in ("t1", "t2"):
        (tmp_path / name).mkdir()
        make_dataset(tmp_path / name / "_dataset.csv", n_inside=4, n_outside=4, rows_per_point=10, seed=len(name) + 5)
    multi_out = tmp_path / "multi_runs"
    assert cli_main(["--dataset", str(tmp_path / "t1"), "--dataset", str(tmp_path / "t2"), "--epochs", "1",
                     "--output-dir", str(multi_out)]) == 0
    (run_dir,) = multi_out.iterdir()
    assert [Path(p).parent.name for p in load_run(run_dir).config.data.datasets] == ["t1", "t2"]

    # 多个独立验证集
    for name in ("d1", "d2"):
        (tmp_path / name).mkdir()
        make_dataset(tmp_path / name / "_dataset.csv", n_inside=3, n_outside=3, rows_per_point=10, seed=len(name))
    val_out = tmp_path / "val_runs"
    assert cli_main(["--dataset", str(dataset), "--epochs", "1", "--output-dir", str(val_out),
                     "--val-dataset", str(tmp_path / "d1"), "--val-dataset", str(tmp_path / "d2")]) == 0
    (run_dir,) = val_out.iterdir()
    assert len(load_run(run_dir).config.data.val_datasets) == 2
    assert "  d1" in capsys.readouterr().out  # 按文件夹分别列出指标

    assert cli_main(["--dataset", str(dataset), "--list-features"]) == 0
    assert "rssi_FL" in capsys.readouterr().out
    assert cli_main(["--dataset", str(dataset), "--batch-size", "0"]) == 1
    assert "batch size" in capsys.readouterr().err


def test_dataset_written_by_merger_is_readable(tmp_path: Path):
    """数据集构建工具的总表（含 ts_锚点 列、补空的列）可以直接训练。"""
    rows = []
    for k, label in enumerate([1] * 6 + [0] * 6):
        for t in range(10):
            rssi = -50 - 15 * label + np.sin(t) * 2
            rows.append({"ts": t, "ts_FL": t, "rssi_FL": rssi, "errcode_FL": "0x00000000", "ts_IF": t,
                         "rssi_IF": -rssi - 100, "errcode_IF": 0, "scene": "内" if label else "外",
                         "point": str(k), "label": label})
    df = pd.DataFrame(rows)
    df.loc[3, "rssi_IF"] = None
    df.to_csv(tmp_path / "_dataset.csv", index=False, encoding="utf-8-sig")
    data = prepare_data(DataConfig(datasets=[str(tmp_path / "_dataset.csv")]), seed=0)
    assert data.features == ["rssi_FL", "rssi_IF"]
