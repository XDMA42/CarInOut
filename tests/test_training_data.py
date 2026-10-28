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
    data = prepare_data(DataConfig(datasets=[str(dataset)], val_ratio=0.2, test_ratio=0.2), seed=1)
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
    data = prepare_data(DataConfig(datasets=[str(dataset)]), seed=0)
    assert np.allclose(data.train.x.mean(axis=0), 0, atol=1e-5)
    assert np.allclose(data.train.x.std(axis=0), 1, atol=1e-4)
    assert data.train.x.dtype == np.float32 and data.train.y.dtype == np.int64


def test_missing_values(tmp_path: Path):
    path = make_dataset(tmp_path / "d.csv", n_inside=6, n_outside=6, rows_per_point=20, missing_rate=0.1)
    dropped = prepare_data(DataConfig(datasets=[str(path)], standardize=False), seed=0)
    filled = prepare_data(DataConfig(datasets=[str(path)], standardize=False, missing="mean"), seed=0)
    constant = prepare_data(
        DataConfig(datasets=[str(path)], standardize=False, missing="constant", fill_value=-120), seed=0
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
    cfg = DataConfig(datasets=[str(main)], test_datasets=[str(other)], val_ratio=0.3, test_ratio=0.5)
    data = prepare_data(cfg, seed=0)
    assert len(data.test) == 60 and set(data.test.meta["source"]) == {"b"}
    assert len(data.train) + len(data.val) == 120 and set(data.train.meta["source"]) == {"a"}


def day(root: Path, name: str, seed: int, points: int = 3) -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    make_dataset(folder / "_dataset.csv", n_inside=points, n_outside=points, rows_per_point=10, seed=seed)
    return folder


def test_independent_validation_folders(tmp_path: Path):
    main = day(tmp_path / "train", "1023采集", seed=1, points=6) / "_dataset.csv"
    val = [day(tmp_path / "merged", "1024采集", seed=2), day(tmp_path / "merged", "1025采集", seed=3)]
    test = day(tmp_path / "other", "1026采集", seed=4)
    cfg = DataConfig(datasets=[str(main)], val_datasets=[str(v) for v in val], test_datasets=[str(test)])
    notes = []
    data = prepare_data(cfg, seed=0, log=notes.append)
    assert len(data.train) == 120  # 验证集、测试集都是独立数据，训练数据全部用于训练
    assert len(data.val) == 120 and len(data.test) == 60
    assert data.val.meta["source"].value_counts().to_dict() == {"1024采集": 60, "1025采集": 60}
    assert set(data.train.meta["source"]) == {"1023采集"}
    assert data.val_files == [str(v / "_dataset.csv") for v in val]
    assert data.test_files == [str(test / "_dataset.csv")]
    assert any("独立数据" in n for n in notes) and any(n.strip().startswith("1025采集：") for n in notes)
    # 各集合使用训练集的标准化参数
    assert not np.allclose(data.val.x.mean(axis=0), 0, atol=1e-3)


def test_parent_folder_expands_to_subfolders(tmp_path: Path):
    main = day(tmp_path, "train", seed=1, points=6) / "_dataset.csv"
    merged = tmp_path / "merged"
    for i, name in enumerate(["day10", "day2", "day1"]):
        day(merged, name, seed=i + 2)
    (merged / "notes").mkdir()  # 没有 _dataset.csv 的子文件夹被跳过
    cfg = DataConfig(datasets=[str(main)], val_datasets=[str(merged), str(merged / "day2")])
    data = prepare_data(cfg, seed=0)
    assert [Path(f).parent.name for f in data.val_files] == ["day1", "day2", "day10"]  # 自然排序、去重
    assert len(data.val) == 180


def test_independent_dataset_errors(tmp_path: Path):
    main_dir = day(tmp_path, "train", seed=1)
    main = main_dir / "_dataset.csv"
    other = day(tmp_path, "other", seed=2)
    (tmp_path / "empty").mkdir()
    cases = [
        ({"val_datasets": [str(tmp_path / "empty")]}, "都没有 _dataset.csv"),
        ({"val_datasets": [str(tmp_path / "missing")]}, "找不到"),
        ({"val_datasets": [str(main_dir)]}, "包含训练数据"),
        ({"test_datasets": [str(main)]}, "包含训练数据"),
        ({"val_datasets": [str(other)], "test_datasets": [str(other / "_dataset.csv")]}, "同时出现"),
    ]
    for kwargs, message in cases:
        with pytest.raises(ValueError, match=message):
            prepare_data(DataConfig(datasets=[str(main)], **kwargs), seed=0)


def test_multiple_training_folders(tmp_path: Path):
    merged = tmp_path / "merged"
    folders = [day(merged, "day1", seed=1, points=6), day(merged, "day2", seed=2, points=6)]
    data = prepare_data(DataConfig(datasets=[str(merged)], val_ratio=0.25, test_ratio=0.25), seed=0)  # 上级文件夹
    assert data.train_files == [str(f / "_dataset.csv") for f in folders]
    total = len(data.train) + len(data.val) + len(data.test)
    assert total == 2 * 12 * 10
    assert set(data.train.meta["source"]) == {"day1", "day2"}
    # 两个文件夹中场景、点位相同的数据分在同一个集合
    positions = {name: set(zip(data.split(name).meta["scene"], data.split(name).meta["point"]))
                 for name in ("train", "val", "test")}
    assert not (positions["train"] & positions["val"] or positions["train"] & positions["test"]
                or positions["val"] & positions["test"])
    assert any("视为同一位置" in n for n in data.notes)
    assert any(n.strip().startswith("day2：") for n in data.notes)  # 按文件夹列出行数


def test_training_folders_with_different_columns(tmp_path: Path):
    a = day(tmp_path, "a", seed=1, points=4)
    b = day(tmp_path, "b", seed=2, points=4)
    table = pd.read_csv(a / "_dataset.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    table["rssi_EXTRA"] = "-70"  # 只有 a 有的列
    table["rssi_FL"] = ""  # a 中整列为空，b 中有值：仍是候选特征，按缺失值处理
    table.to_csv(a / "_dataset.csv", index=False, encoding="utf-8-sig")
    infos = {i.name: i for i in feature_candidates([load_table(a / "_dataset.csv"), load_table(b / "_dataset.csv")],
                                                   "label")}
    assert "rssi_EXTRA" not in infos and infos["rssi_FL"].missing_ratio == 0.5
    data = prepare_data(DataConfig(datasets=[str(a), str(b)], missing="mean"), seed=0)
    assert "rssi_EXTRA" not in data.features and any("rssi_EXTRA" in n for n in data.notes)
    with pytest.raises(ValueError, match="a 中没有特征列|b 中没有特征列"):
        prepare_data(DataConfig(datasets=[str(a), str(b)], features=["rssi_EXTRA"]), seed=0)


def test_selected_features_and_errors(dataset: Path, tmp_path: Path):
    data = prepare_data(DataConfig(datasets=[str(dataset)], features=["rssi_IF", "rssi_FL"]), seed=0)
    assert data.features == ["rssi_IF", "rssi_FL"] and data.train.x.shape[1] == 2
    with pytest.raises(ValueError, match="没有特征列"):
        prepare_data(DataConfig(datasets=[str(dataset)], features=["rssi_XX"]), seed=0)
    with pytest.raises(ValueError, match="不是数值"):
        prepare_data(DataConfig(datasets=[str(dataset)], features=["scene"]), seed=0)
    bad = write(tmp_path / "bad.csv", "rssi,scene,point,label\n-50,s,1,2\n")
    with pytest.raises(ValueError, match="只能是 0"):
        prepare_data(DataConfig(datasets=[str(bad)]), seed=0)
    one_class = write(tmp_path / "one.csv", "rssi,scene,point,label\n-50,s,1,1\n-60,s,2,1\n")
    with pytest.raises(ValueError, match="只有一个类别"):
        prepare_data(DataConfig(datasets=[str(one_class)], val_ratio=0), seed=0)
    with pytest.raises(ValueError, match="找不到"):
        prepare_data(DataConfig(datasets=[str(tmp_path / "none.csv")]), seed=0)
