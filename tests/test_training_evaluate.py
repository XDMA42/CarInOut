import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from carinout.training.cli import main as cli_main
from carinout.training.config import DataConfig, ExperimentConfig, ModelConfig, OptimConfig
from carinout.training.data import load_table, prepare_data
from carinout.training.evaluate import (
    EVALUATION_FILE,
    evaluate_model,
    is_evaluation_dir,
    load_evaluation,
    save_evaluation_figure,
)
from carinout.training.runs import Predictor, load_run, new_run_dir, per_scene_accuracy, save_run
from carinout.training.trainer import fit
from synthetic import make_dataset


def day(root: Path, name: str, seed: int, points: int = 4, **kwargs) -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    make_dataset(folder / "_dataset.csv", n_inside=points, n_outside=points, rows_per_point=12, seed=seed,
                 sub_scenes=True, **kwargs)
    return folder


@pytest.fixture
def trained(tmp_path: Path) -> tuple[Path, Path]:
    """(训练结果文件夹, merged 文件夹)；merged 中 day1 用于训练，day2 ~ day4 未参与训练。"""
    merged = tmp_path / "merged"
    day(merged, "day1", seed=1, points=8)
    for i, name in enumerate(["day2", "day3", "day4"]):
        day(merged, name, seed=i + 2)
    cfg = ExperimentConfig(
        data=DataConfig(datasets=[str(merged / "day1")], missing="mean"),
        model=ModelConfig("mlp", {"hidden_sizes": [16, 8]}),
        train=OptimConfig(epochs=5, batch_size=32, learning_rate=0.01),
        output_dir=str(tmp_path / "runs"),
    )
    data = prepare_data(cfg.data, cfg.train.seed)
    run_dir = new_run_dir(cfg)
    save_run(run_dir, cfg, data, fit(cfg, data), figures=False)
    return run_dir, merged


def test_evaluate_validation_and_test_folders(trained):
    run_dir, merged = trained
    logs = []
    val, test = [str(merged / "day2"), str(merged / "day3")], [str(merged / "day4")]
    record = evaluate_model(run_dir, val, test, log=logs.append)

    assert is_evaluation_dir(record.path) and record.path.parent == run_dir
    assert {p.name for p in record.path.iterdir()} == {EVALUATION_FILE, "predictions.csv"}
    save_evaluation_figure(record)
    assert (record.path / "evaluation.png").is_file()

    m = record.metrics
    assert set(m["splits"]) == {"val", "test"}
    assert m["splits"]["val"]["n"] == 2 * 8 * 12 and m["splits"]["test"]["n"] == 8 * 12
    assert set(m["sources"]) == {"val"} and set(m["sources"]["val"]) == {"day2", "day3"}
    assert m["datasets"]["val"] == [str((merged / d / "_dataset.csv").resolve()) for d in ("day2", "day3")]
    assert not m["notes"]  # 没有用到训练数据
    assert np.isfinite(m["splits"]["val"]["loss"])

    # 与直接用 Predictor 推理一致
    predictor = Predictor.load(run_dir)
    day4 = load_table(merged / "day4" / "_dataset.csv")
    test = record.predictions[record.predictions["split"] == "test"]
    assert np.allclose(test["prob"].to_numpy(), predictor.predict_proba(day4))

    # 每个场景的准确率：验证集、测试集各 4 个场景，点位数合计等于点位总数
    scenes = pd.DataFrame(m["scenes"])
    assert set(scenes["split"]) == {"val", "test"}
    assert set(scenes["scene"]) <= {"前排座椅", "后排座椅", "车外近处", "车外远处"}
    assert {"车内", "车外"} == {"车内" if s.startswith(("前排", "后排")) else "车外" for s in scenes["scene"]}
    assert (scenes.groupby("scene")["label"].nunique() == 1).all()  # 每个场景只有一个类别
    assert scenes.groupby("split")["samples"].sum().to_dict() == {"val": 192, "test": 96}

    again = load_evaluation(record.path)
    assert again.model_dir == run_dir and again.metrics["splits"]["test"]["n"] == 96
    assert again.predictions["point"].map(type).eq(str).all()
    assert any("准确率" in line for line in logs)


def test_evaluate_warns_about_training_data_and_errors(trained, tmp_path: Path):
    run_dir, merged = trained
    record = evaluate_model(run_dir, [], [str(merged)])  # 上级文件夹：含训练用的 day1
    assert any("day1 是该模型的训练数据" in n for n in record.metrics["notes"])
    assert set(record.predictions["source"]) == {"day1", "day2", "day3", "day4"}

    cases = [
        ((run_dir, [], []), "至少指定"),
        ((tmp_path, [str(merged / "day2")], []), "没有模型文件"),
        ((run_dir, [str(merged / "day2")], [str(merged / "day2")]), "同时出现"),
    ]
    for args, message in cases:
        with pytest.raises(ValueError, match=message):
            evaluate_model(*args)
    table = pd.read_csv(merged / "day3" / "_dataset.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    table.drop(columns="rssi_FL").to_csv(merged / "day3" / "_dataset.csv", index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError, match="day3：数据中没有特征列：rssi_FL"):
        evaluate_model(run_dir, [str(merged / "day3")], [])


def test_evaluate_threshold(trained):
    run_dir, merged = trained
    default = evaluate_model(run_dir, [str(merged / "day2")], [])
    assert default.metrics["threshold"] == 0.5  # 模型训练时的阈值
    strict = evaluate_model(run_dir, [str(merged / "day2")], [], threshold=0.9)
    assert strict.metrics["threshold"] == 0.9
    pred = strict.predictions
    assert (pred["pred"] == (pred["prob"] >= 0.9).astype(int)).all()
    (tn, fp), (fn, tp) = strict.metrics["splits"]["val"]["confusion"]
    (tn0, fp0), _ = default.metrics["splits"]["val"]["confusion"]
    assert fp <= fp0  # 阈值越高，车外误判为车内越少
    with pytest.raises(ValueError, match="判定阈值"):
        evaluate_model(run_dir, [str(merged / "day2")], [], threshold=0)


def test_rows_with_missing_features_are_skipped_when_trained_with_drop(tmp_path: Path):
    train = day(tmp_path, "train", seed=1, points=6)
    holes = day(tmp_path, "holes", seed=2, missing_rate=0.2)
    cfg = ExperimentConfig(data=DataConfig(datasets=[str(train)]), train=OptimConfig(epochs=2),
                           output_dir=str(tmp_path / "runs"))
    data = prepare_data(cfg.data, cfg.train.seed)
    run_dir = new_run_dir(cfg)
    save_run(run_dir, cfg, data, fit(cfg, data), figures=False)
    record = evaluate_model(run_dir, [str(holes)], [])
    assert 0 < record.metrics["splits"]["val"]["n"] < 8 * 12
    assert any("未参与评估" in n for n in record.metrics["notes"])


def test_training_run_records_scene_accuracy(trained):
    run_dir, _ = trained
    run = load_run(run_dir)
    scenes = pd.DataFrame(run.metrics["scenes"])
    assert {"split", "scene", "label", "points", "samples", "accuracy", "mean_prob"} <= set(scenes.columns)
    table = per_scene_accuracy(run.predictions)
    for split, part in table.groupby("split", sort=False):
        assert part["accuracy"].is_monotonic_increasing  # 各集合内从低到高
    assert list(dict.fromkeys(table["split"])) == [s for s in ("train", "val", "test") if s in set(table["split"])]


def test_cli_evaluate(trained, capsys):
    run_dir, merged = trained
    args = ["--evaluate", str(run_dir), "--val-dataset", str(merged / "day2"), "--val-dataset", str(merged / "day3"),
            "--test-dataset", str(merged / "day4")]
    assert cli_main(args) == 0
    out = capsys.readouterr().out
    assert "各场景准确率" in out and "  day2" in out
    assert "前排座椅" in out or "后排座椅" in out
    (eval_dir,) = [p for p in run_dir.iterdir() if p.name.startswith("eval_")]
    assert json.loads((eval_dir / EVALUATION_FILE).read_text(encoding="utf-8"))["splits"]["test"]["n"] == 96
    assert (eval_dir / "evaluation.png").is_file()

    assert cli_main(["--evaluate", str(run_dir)]) == 1
    assert cli_main(["--evaluate", str(run_dir), "--dataset", str(merged / "day2")]) == 1
    assert "--val-dataset" in capsys.readouterr().err
