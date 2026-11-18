from pathlib import Path

import pytest

from carinout.parser import DEFAULT_PATTERN, compile_pattern, group_by_point, parse_filename, scan_folder

REGEX = compile_pattern(DEFAULT_PATTERN)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("中排座椅_53_SLAVE2.csv", ("中排座椅", "53", "SLAVE2")),
        ("中排座椅_53_slave10.CSV", ("中排座椅", "53", "SLAVE10")),
        ("中排座椅_53_SLAVE2_20240501.csv", ("中排座椅", "53", "SLAVE2")),
        ("车外_左侧_101_SLAVE7.csv", ("车外_左侧", "101", "SLAVE7")),
        ("中排座椅-53-SLAVE2.csv", ("中排座椅", "53", "SLAVE2")),
        ("中间座椅+脚坑-68-[COM261][SLAVE4] 2026-10-23 15-44-12.743.csv", ("中间座椅+脚坑", "68", "SLAVE4")),
        ("中间座椅+脚坑-69-[COM263][SLAVE3] 2026-10-23 15-53-32.871.csv", ("中间座椅+脚坑", "69", "SLAVE3")),
        ("中间座椅+脚坑-68-[COM266][master_mcu] 2026-10-23 15-44-12.772.csv", ("中间座椅+脚坑", "68", "MASTER_MCU")),
        ("中间座椅+脚坑-68-[COM267][SLVAE5] 2026-10-23 15-44-12.770.csv", ("中间座椅+脚坑", "68", "SLVAE5")),
        ("车外-左侧-101-[COM261][SLAVE1] 2026-10-23 15-44-12.743.csv", ("车外-左侧", "101", "SLAVE1")),
        ("中间座椅-68-[SLAVE4].csv", ("中间座椅", "68", "SLAVE4")),
    ],
)
def test_parse_filename(name, expected):
    assert parse_filename(name, REGEX) == expected


@pytest.mark.parametrize("name", ["中排座椅_53.csv", "notes.csv", "中排座椅_53_SLAVE2.txt"])
def test_parse_filename_rejects(name):
    assert parse_filename(name, REGEX) is None


def test_compile_pattern_requires_groups():
    with pytest.raises(ValueError):
        compile_pattern(r"(?P<scene>.+)_(?P<point>\d+)\.csv")
    with pytest.raises(ValueError):
        compile_pattern(r"(")


def test_scan_and_group(tmp_path: Path):
    for name in ["A_2_SLAVE1.csv", "A_2_SLAVE10.csv", "A_2_SLAVE2.csv", "A_10_SLAVE1.csv", "B_1_SLAVE1.csv", "x.csv"]:
        (tmp_path / name).write_text("errcode\n0\n")
    parsed, unmatched = scan_folder(tmp_path)
    assert [p.name for p in unmatched] == ["x.csv"]
    groups = group_by_point(parsed)
    src = tmp_path.name  # 来源默认取文件夹名
    assert list(groups) == [(src, "A", "2"), (src, "A", "10"), (src, "B", "1")]
    assert list(groups[(src, "A", "2")]) == ["SLAVE1", "SLAVE2", "SLAVE10"]


def test_scan_folders_keeps_same_point_from_different_folders_apart(tmp_path: Path):
    from carinout.parser import scan_folders

    for day in ("1023采集", "1024采集"):
        (tmp_path / day).mkdir()
        for name in ("座椅-1-[SLAVE1].csv", "座椅-1-[SLAVE2].csv"):
            (tmp_path / day / name).write_text("ts,errcode\n1,0\n")
    (tmp_path / "1024采集" / "notes.csv").write_text("a\n")
    parsed, unmatched = scan_folders([tmp_path / "1023采集", tmp_path / "1024采集"])
    assert [p.name for p in unmatched] == ["notes.csv"]
    groups = group_by_point(parsed)
    assert list(groups) == [("1023采集", "座椅", "1"), ("1024采集", "座椅", "1")]
    assert all(list(v) == ["SLAVE1", "SLAVE2"] for v in groups.values())


def test_unique_source_names(tmp_path: Path):
    from carinout.parser import unique_source_names

    dirs = [tmp_path / "a" / "raw", tmp_path / "b" / "raw", tmp_path / "1023采集", tmp_path / "c" / "a_raw"]
    assert unique_source_names(dirs) == ["a_raw", "b_raw", "1023采集", "a_raw_2"]


def test_legacy_default_pattern_in_config_is_upgraded(tmp_path: Path):
    import json

    from carinout.anchors import AppConfig
    from carinout.parser import LEGACY_DEFAULT_PATTERNS

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"filename_pattern": LEGACY_DEFAULT_PATTERNS[0]}), encoding="utf-8")
    config = AppConfig.load(cfg)
    assert config.filename_pattern == DEFAULT_PATTERN
    config.save(cfg)
    assert json.loads(cfg.read_text(encoding="utf-8"))["filename_pattern"] == ""

    custom = r"^(?P<scene>.+)_(?P<point>\d+)_(?P<anchor>\w+)\.csv$"
    cfg.write_text(json.dumps({"filename_pattern": custom}), encoding="utf-8")
    assert AppConfig.load(cfg).filename_pattern == custom


def test_scene_labels_roundtrip(tmp_path: Path):
    import json

    from carinout.anchors import AppConfig

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"scene_labels": {"座椅": 1, "车外": 0, "坏值": 2, "布尔": True}}), encoding="utf-8")
    config = AppConfig.load(cfg)
    assert config.scene_labels == {"座椅": 1, "车外": 0}
    config.scene_labels["脚坑"] = 1
    config.save(cfg)
    assert AppConfig.load(cfg).scene_labels == {"座椅": 1, "车外": 0, "脚坑": 1}


def test_point_labels_override_scene_labels(tmp_path: Path):
    import json

    from carinout.anchors import AppConfig

    cfg = tmp_path / "config.json"
    config = AppConfig.load(cfg)
    config.scene_labels["中间座椅+脚坑"] = 1
    config.set_point_label("中间座椅+脚坑", "69", 0)
    config.set_point_label("车外左侧", "3", 1)  # 场景未标注，也可以只给某个点位打标签
    assert config.label_for("中间座椅+脚坑", "68") == 1  # 跟随场景
    assert config.label_for("中间座椅+脚坑", "69") == 0  # 单独设置
    assert config.label_for("车外左侧", "3") == 1 and config.label_for("车外左侧", "4") is None
    assert config.is_point_overridden("中间座椅+脚坑", "69") and not config.is_point_overridden("中间座椅+脚坑", "68")

    # 场景标签改了，单独设置的点位不受影响
    config.scene_labels["中间座椅+脚坑"] = 0
    config.set_point_label("中间座椅+脚坑", "69", 1)
    assert config.label_for("中间座椅+脚坑", "68") == 0 and config.label_for("中间座椅+脚坑", "69") == 1

    config.save(cfg)
    loaded = AppConfig.load(cfg)
    assert loaded.point_labels == {"中间座椅+脚坑": {"69": 1}, "车外左侧": {"3": 1}}

    # 恢复跟随场景后不留空字典
    loaded.set_point_label("车外左侧", "3", None)
    assert "车外左侧" not in loaded.point_labels

    # 旧配置里的小场景字段和无效的点位标签被忽略
    cfg.write_text(json.dumps({
        "point_labels": {"a": {"1": 0, "2": 5, "3": True}, "b": "bad", "c": {"9": 2}},
        "sub_categories": {"0": "前排"}, "scene_subs": {"a": 0},
    }), encoding="utf-8")
    assert AppConfig.load(cfg).point_labels == {"a": {"1": 0}}
