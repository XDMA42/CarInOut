from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from carinout.training.config import DataConfig
from carinout.training.data import feature_candidates, load_table, prepare_data, split_indices
from synthetic import make_dataset


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    return make_dataset(tmp_path / "_dataset.csv", n_inside=10, n_outside=10, rows_per_point=20)


def groups_of(split) -> set[tuple[str, str]]:
    return set(zip(split.meta["scene"], split.meta["point"]))


def test_feature_candidates_skip_meta_and_non_numeric(tmp_path: Path):
    path = write(
        tmp_path / "d.csv",
        "ts,ts_FL,rssi_FL,errcode_FL,mac_FL,chan_FL,rssi_FR,scene,point,label\n"
        "1,1,-50,0,aa:bb,37,,s,1,1\n"
        "2,2,-51.5,0,aa:bb,37,-60,s,1,0\n",
    )
    infos = {i.name: i for i in feature_candidates(load_table(path), "label")}
    assert list(infos) == ["rssi_FL", "chan_FL", "rssi_FR"]
    assert infos["chan_FL"].constant and not infos["rssi_FL"].constant
    assert infos["rssi_FR"].missing_ratio == 0.5


def test_split_by_point_keeps_points_together(dataset: Path):
    data = prepare_data(DataConfig(dataset=str(dataset), val_ratio=0.2, test_ratio=0.2), seed=1)
    train, val, test = (groups_of(s) for s in (data.train, data.val, data.test))
    assert train and val and test
    assert not (train & val or train & test or val & test)
    for split in (data.train, data.val, data.test):
        assert set(np.unique(split.y)) == {0, 1}  # 按标签分层
    assert len(data.train) + len(data.val) + len(data.test) == 20 * 20
    assert len(data.val) == len(data.test) == 80  # 每类 10 个点位，各取 2 个


def test_split_by_row_is_stratified():
    labels = np.array([0] * 80 + [1] * 20)
    train, val, test = split_indices(labels, None, 0.2, 0.1, seed=0)
    assert len(np.intersect1d(train, val)) == len(np.intersect1d(train, test)) == 0
    assert len(train) + len(val) + len(test) == 100
    assert (labels[val] == 1).sum() == 4 and (labels[test] == 1).sum() == 2


def test_split_by_point_leaves_one_group_for_training():
    labels = np.array([1] * 10 + [0] * 10)
    groups = np.array(["a"] * 5 + ["b"] * 5 + ["c"] * 10)
    train, val, test = split_indices(labels, groups, 0.45, 0.45, seed=0)
    assert set(labels[train]) == {0, 1}
    assert "c" in set(groups[train])  # 车外只有一个点位，只能留给训练集


def test_standardize_with_train_statistics(dataset: Path):
    data = prepare_data(DataConfig(dataset=str(dataset)), seed=0)
    assert np.allclose(data.train.x.mean(axis=0), 0, atol=1e-5)
    assert np.allclose(data.train.x.std(axis=0), 1, atol=1e-4)
    assert data.train.x.dtype == np.float32 and data.train.y.dtype == np.int64


def test_missing_values(tmp_path: Path):
    path = make_dataset(tmp_path / "d.csv", n_inside=6, n_outside=6, rows_per_point=20, missing_rate=0.1)
    dropped = prepare_data(DataConfig(dataset=str(path), standardize=False), seed=0)
    filled = prepare_data(DataConfig(dataset=str(path), standardize=False, missing="mean"), seed=0)
    constant = prepare_data(
        DataConfig(dataset=str(path), standardize=False, missing="constant", fill_value=-120), seed=0
    )
    total = 12 * 20
    assert sum(len(s) for s in (dropped.train, dropped.val, dropped.test)) < total
    assert any("缺失值" in n for n in dropped.notes)
    for data in (filled, constant):
        assert sum(len(s) for s in (data.train, data.val, data.test)) == total
        assert not np.isnan(np.concatenate([data.train.x, data.val.x, data.test.x])).any()
    assert (constant.train.x == -120).any()
    # 均值只用训练集计算
    raw = load_table(path)
    assert filled.fill_values is not None and not np.allclose(
        filled.fill_values, pd.to_numeric(raw[filled.features[0]], errors="coerce").mean()
    )


def test_independent_test_dataset(tmp_path: Path):
    main = make_dataset(tmp_path / "a.csv", n_inside=6, n_outside=6, rows_per_point=10, seed=1)
    other = make_dataset(tmp_path / "b.csv", n_inside=3, n_outside=3, rows_per_point=10, seed=2)
    data = prepare_data(DataConfig(dataset=str(main), test_dataset=str(other), val_ratio=0.3, test_ratio=0.5), seed=0)
    assert len(data.test) == 60
    assert len(data.train) + len(data.val) == 120


def test_selected_features_and_errors(dataset: Path, tmp_path: Path):
    data = prepare_data(DataConfig(dataset=str(dataset), features=["rssi_IF", "rssi_FL"]), seed=0)
    assert data.features == ["rssi_IF", "rssi_FL"] and data.train.x.shape[1] == 2
    with pytest.raises(ValueError, match="没有特征列"):
        prepare_data(DataConfig(dataset=str(dataset), features=["rssi_XX"]), seed=0)
    with pytest.raises(ValueError, match="不是数值"):
        prepare_data(DataConfig(dataset=str(dataset), features=["scene"]), seed=0)
    bad = write(tmp_path / "bad.csv", "rssi,scene,point,label\n-50,s,1,2\n")
    with pytest.raises(ValueError, match="只能是 0"):
        prepare_data(DataConfig(dataset=str(bad)), seed=0)
    one_class = write(tmp_path / "one.csv", "rssi,scene,point,label\n-50,s,1,1\n-60,s,2,1\n")
    with pytest.raises(ValueError, match="只有一个类别"):
        prepare_data(DataConfig(dataset=str(one_class), val_ratio=0), seed=0)
    with pytest.raises(ValueError, match="找不到"):
        prepare_data(DataConfig(dataset=str(tmp_path / "none.csv")), seed=0)
