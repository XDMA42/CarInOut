from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from carinout.training.data import load_table
from carinout.training.feature_plots import (
    ALL_FEATURES,
    INSIDE_HIGHER,
    INSIDE_LOWER,
    analyze,
    correlation,
    cross_val_auc,
    describe,
    detect_anchors,
    discrimination,
    feature_discrimination,
    load_features,
    main,
    save_all,
    split_column,
)
from synthetic import make_dataset


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """模拟总表，再按 rssi 派生出 dis_sm、dis_main_sm 两个特征名带下划线的列。"""
    folder = tmp_path / "1023采集"
    folder.mkdir()
    path = make_dataset(folder / "_dataset.csv", n_inside=6, n_outside=6, rows_per_point=20)
    df = load_table(path)
    for col in [c for c in df.columns if c.startswith("rssi_")]:
        anchor = col[len("rssi_"):]
        dis = 10 ** ((-45 - df[col].astype(float)) / 20) * 100
        df[f"dis_sm_{anchor}"] = dis.round(1).astype(str)
        df[f"dis_main_sm_{anchor}"] = (dis * 2).round(1).astype(str)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return folder


def test_split_column_uses_anchor_names_from_errcode_columns():
    anchors = detect_anchors(["ts", "dis_main_sm_FL", "errcode_FL", "ErrCode_IN_F", "errcode_F", "scene"])
    assert anchors == ["FL", "IN_F", "F"]
    assert split_column("dis_main_sm_FL", anchors) == ("dis_main_sm", "FL")
    assert split_column("dis_sm_IN_F", anchors) == ("dis_sm", "IN_F")  # 锚点名本身含下划线
    assert split_column("rssi_init_F", anchors) == ("rssi_init", "F")
    assert split_column("battery", anchors) is None
    assert split_column("rssi_init_F", []) == ("rssi_init", "F")  # 没有 errcode 列时按最后一个下划线拆分


def test_load_features_splits_columns_and_skips_meta(dataset: Path):
    ft = load_features(load_table(dataset / "_dataset.csv"))
    assert ft.features == ["rssi", "dis_sm", "dis_main_sm"]
    assert ft.anchors == ["FL", "F", "FR", "R", "RR", "B", "RL", "L", "IF", "IR"]
    assert ft.column("dis_main_sm", "IR") == "dis_main_sm_IR"
    assert set(ft.labels.unique()) == {0, 1}
    assert len(ft.columns) == 30 and not any(c.startswith(("ts", "errcode")) for c in ft.values.columns)


def test_load_features_selects_features_and_anchors(dataset: Path):
    ft = load_features(load_table(dataset / "_dataset.csv"), ["dis_main_sm", "dis_sm"], ["IR", "FL"])
    assert ft.features == ["dis_main_sm", "dis_sm"]
    assert ft.anchors == ["IR", "FL"]
    assert list(ft.values.columns) == ["dis_main_sm_IR", "dis_main_sm_FL", "dis_sm_IR", "dis_sm_FL"]
    with pytest.raises(ValueError, match="没有特征：dis.*可用的特征：rssi, dis_sm, dis_main_sm"):
        load_features(load_table(dataset / "_dataset.csv"), ["dis"])
    with pytest.raises(ValueError, match="没有锚点：XX"):
        load_features(load_table(dataset / "_dataset.csv"), None, ["XX"])


def test_features_without_shared_anchor(tmp_path: Path):
    path = write(
        tmp_path / "d.csv",
        "ts,dis_sm_A,errcode_A,rssi_init_B,errcode_B,label\n"
        + "".join(f"{i},{100 + i},0,{-60 - i % 7},0,{i % 2}\n" for i in range(30)),
    )
    with pytest.raises(ValueError, match="所选锚点上没有这些特征的数据列：rssi_init"):
        load_features(load_table(path), ["rssi_init"], ["A"])
    ft = load_features(load_table(path))
    names = {p.name for p in save_all(ft, analyze(ft), tmp_path / "out", dpi=40)}
    assert "corr_pairs.png" in names and not any(n.startswith("scatter_") for n in names)


def test_correlation_between_anchors_features_and_label(dataset: Path):
    ft = load_features(load_table(dataset / "_dataset.csv"), ["dis_sm", "dis_main_sm", "rssi"])
    corr = correlation(ft)
    assert corr.at["dis_sm_FL", "dis_main_sm_FL"] == pytest.approx(1.0)
    assert corr.at["dis_sm_FL", "rssi_FL"] < -0.5
    # 车内锚点：车内时距离近、信号强
    assert corr.at["rssi_IF", "label"] > 0.5 and corr.at["dis_sm_IF", "label"] < 0
    spearman = correlation(ft, "spearman")
    assert spearman.at["dis_sm_FL", "rssi_FL"] == pytest.approx(-1.0)  # 单调关系，秩相关为 -1

    stats = describe(ft, corr).set_index("column")
    assert stats.at["rssi_IF", "mean_inside"] > stats.at["rssi_IF", "mean_outside"]
    assert stats.at["rssi_IF", "corr_label"] == pytest.approx(corr.at["rssi_IF", "label"])
    assert stats.at["rssi_IF", "count"] == len(ft.values)


def test_discrimination_per_column(dataset: Path):
    ft = load_features(load_table(dataset / "_dataset.csv"), ["dis_sm", "rssi"])
    disc = discrimination(ft)
    assert list(disc["separation"]) == sorted(disc["separation"], reverse=True)
    rows = disc.set_index("column")
    assert rows.at["rssi_IF", "direction"] == INSIDE_HIGHER and rows.at["rssi_IF", "separation"] > 0.8
    assert rows.at["dis_sm_IF", "direction"] == INSIDE_LOWER
    # dis_sm 是 rssi 的单调函数：区分度相同，方向相反
    assert rows.at["dis_sm_IF", "separation"] == pytest.approx(rows.at["rssi_IF", "separation"])
    assert rows.at["dis_sm_IF", "auc"] == pytest.approx(1 - rows.at["rssi_IF", "auc"])
    # 按规则判断的平衡准确率与表中一致
    for col in ("rssi_IF", "dis_sm_IF"):
        values, inside = ft.values[col].to_numpy(), ft.labels.to_numpy() == 1
        threshold = rows.at[col, "threshold"]
        pred = values >= threshold if rows.at[col, "direction"] == INSIDE_HIGHER else values <= threshold
        balanced = (pred[inside].mean() + (~pred[~inside]).mean()) / 2
        assert balanced == pytest.approx(rows.at[col, "balanced_accuracy"])
        assert rows.at[col, "rule"].startswith("≥" if rows.at[col, "direction"] == INSIDE_HIGHER else "≤")

    by_feature = feature_discrimination(ft, disc).set_index("feature")
    assert list(by_feature.index) == ["dis_sm", "rssi", ALL_FEATURES]
    assert (by_feature["folds"] == 5).all() and by_feature.at["rssi", "columns"] == 10
    assert by_feature.at["rssi", "cv_auc"] > 0.9
    assert by_feature.at["rssi", "best_single"] == pytest.approx(disc.query("feature == 'rssi'")["separation"].max())


def test_cross_val_auc_groups_by_point():
    rng = np.random.default_rng(0)
    points = np.repeat(np.arange(6), 50)  # 6 个点位，前 3 个车内
    y = (points < 3).astype(np.int64)
    separable = np.column_stack([y * 3 + rng.normal(0, 0.5, len(y)), rng.normal(0, 1, len(y))])
    auc, folds = cross_val_auc(separable, y, points)
    assert auc > 0.95 and folds == 3  # 每类只有 3 个点位，最多 3 折
    noise = rng.normal(0, 1, (len(y), 3))
    noise[rng.random(noise.shape) < 0.1] = np.nan  # 缺失值用均值填充
    auc, folds = cross_val_auc(noise, y, None)
    assert abs(auc - 0.5) < 0.15 and folds == 5
    assert cross_val_auc(separable, y, np.where(y == 1, 0, 1))[1] == 0  # 每类只有 1 个点位，无法交叉验证


def test_missing_columns_values_and_labels(tmp_path: Path):
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame({
        "ts": np.arange(n),
        "dis_sm_A": rng.normal(100, 10, n).round(1),
        "rssi_init_A": rng.normal(-60, 3, n).round(),
        "errcode_A": 0,
        "dis_sm_B": rng.normal(100, 10, n).round(1),
        "errcode_B": 0,
        "const_A": 5,
    })
    df.loc[:4, "dis_sm_A"] = np.nan
    df.loc[: n - 6, "dis_sm_B"] = np.nan  # 两列同时有值的行数不足 MIN_PAIRS
    path = tmp_path / "point.csv"
    df.to_csv(path, index=False)
    ft = load_features(load_table(path))
    assert ft.labels is None and ft.anchors == ["A", "B"]
    assert ft.column("rssi_init", "B") is None
    assert any("没有标签列" in n for n in ft.notes) and any("rssi_init 在以下锚点没有数据列：B" in n for n in ft.notes)
    corr = correlation(ft)
    assert np.isnan(corr.at["dis_sm_B", "dis_sm_A"]) and np.isnan(corr.at["const_A", "dis_sm_A"])
    analysis = analyze(ft)
    assert analysis.disc is None and analysis.by_feature is None
    names = {p.name for p in save_all(ft, analysis, tmp_path / "out")}
    assert names >= {"dist_dis_sm.png", "corr_matrix.png", "corr_pairs.png", "stats.csv"}
    assert not any(n.startswith("discrimination") for n in names)


def test_save_all_writes_figures_and_tables(dataset: Path, tmp_path: Path):
    ft = load_features(load_table(dataset / "_dataset.csv"), ["dis_sm", "dis_main_sm", "rssi"])
    out = tmp_path / "plots"
    names = {p.name for p in save_all(ft, analyze(ft), out, dpi=50)}
    assert names == {
        "discrimination.png", "dist_dis_sm.png", "dist_dis_main_sm.png", "dist_rssi.png",
        "corr_matrix.png", "corr_pairs.png",
        "scatter_dis_sm__dis_main_sm.png", "scatter_dis_sm__rssi.png", "scatter_dis_main_sm__rssi.png",
        "discrimination.csv", "discrimination_by_feature.csv", "stats.csv", "correlation.csv",
    }
    assert all((out / name).stat().st_size > 0 for name in names)
    corr = pd.read_csv(out / "correlation.csv", index_col=0)
    assert list(corr.columns)[-1] == "label" and corr.shape == (31, 31)


def test_cli(dataset: Path, capsys):
    assert main([str(dataset), "--list"]) == 0
    out = capsys.readouterr().out
    assert "dis_main_sm  10 个锚点" in out and "FL, F, FR" in out

    assert main([str(dataset), "--features", "dis_sm,rssi", "--anchors", "IF,IR", "--no-scatter", "--dpi", "40"]) == 0
    out = capsys.readouterr().out
    assert "区分度 AUC" in out and "全部所选特征合起来" in out and "单列区分度最高的 4 列" in out
    plots = dataset / "feature_plots"
    assert (plots / "discrimination.png").is_file() and not list(plots.glob("scatter_*"))

    assert main([str(dataset), "--features", "nope"]) == 2
    assert "没有特征：nope" in capsys.readouterr().err
