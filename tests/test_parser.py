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
    assert list(groups) == [("A", "2"), ("A", "10"), ("B", "1")]
    assert list(groups[("A", "2")]) == ["SLAVE1", "SLAVE2", "SLAVE10"]
