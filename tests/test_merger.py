from pathlib import Path

import pandas as pd
import pytest

from carinout.merger import (
    ALIGN_MAX,
    MergeOptions,
    SUMMARY_FILENAME,
    build_dataset,
    load_anchor,
    read_csv_text,
)
from carinout.parser import group_by_point, scan_folder


def write(path: Path, text: str, encoding: str = "utf-8") -> Path:
    path.write_text(text, encoding=encoding)
    return path


def read_out(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    write(src / "中排座椅_53_SLAVE1.csv", "time,rssi,errcode\n1,-50,0\n2,-51,3\n3,-52,0\n4,-53.50,0\n")
    write(src / "中排座椅_53_SLAVE2.csv", "time,rssi,errcode\n1,-60,0\n2,-61,0\n3,-62,1\n")
    write(src / "中排座椅_54_SLAVE1.csv", "time,rssi,errcode\n1,-70,0\n")
    return src


def run(src: Path, mapping: dict[str, str], **opts):
    parsed, _ = scan_folder(src)
    out = src.parent / "out"
    results = build_dataset(group_by_point(parsed), mapping, ["FL", "FR", "IF"], out, MergeOptions(**opts))
    return out, results


def test_merge_filters_errcode_and_suffixes_columns(folder: Path):
    out, results = run(folder, {"SLAVE1": "FL", "SLAVE2": "FR"})
    df = read_out(out / "中排座椅_53.csv")
    assert list(df.columns) == ["time_FL", "rssi_FL", "errcode_FL", "time_FR", "rssi_FR", "errcode_FR"]
    # SLAVE1 有效 3 行、SLAVE2 有效 2 行，默认按最短截断；数值文本保持原样
    assert df["time_FL"].tolist() == ["1", "3"]
    assert df["rssi_FL"].tolist() == ["-50", "-52"]
    assert df["time_FR"].tolist() == ["1", "2"]
    assert (df[["errcode_FL", "errcode_FR"]] == "0").all().all()
    assert results[0].rows == 2


def test_merge_keep_all_rows(folder: Path):
    out, _ = run(folder, {"SLAVE1": "FL", "SLAVE2": "FR"}, align=ALIGN_MAX)
    df = read_out(out / "中排座椅_53.csv")
    assert df["rssi_FL"].tolist() == ["-50", "-52", "-53.50"]
    assert df["rssi_FR"].tolist() == ["-60", "-61", ""]


def test_column_order_follows_anchor_order(folder: Path):
    out, _ = run(folder, {"SLAVE1": "FR", "SLAVE2": "FL"})
    df = read_out(out / "中排座椅_53.csv")
    assert df.columns[0] == "time_FL" and df["rssi_FL"].tolist()[0] == "-60"


def test_unmapped_slave_ignored(folder: Path):
    out, _ = run(folder, {"SLAVE1": "FL"})
    assert list(read_out(out / "中排座椅_53.csv").columns) == ["time_FL", "rssi_FL", "errcode_FL"]


def test_require_complete_skips_incomplete_points(folder: Path):
    out, results = run(folder, {"SLAVE1": "FL", "SLAVE2": "FR"}, require_complete=True)
    assert (out / "中排座椅_53.csv").exists()
    assert not (out / "中排座椅_54.csv").exists()
    skipped = next(r for r in results if r.point == "54")
    assert "FR" in skipped.status
    summary = read_out(out / SUMMARY_FILENAME)
    assert summary["输出文件"].tolist() == ["中排座椅_53.csv", ""]
    assert summary["FL_有效行"].tolist() == ["3", "1"]


def test_incomplete_point_exported_by_default(folder: Path):
    out, _ = run(folder, {"SLAVE1": "FL", "SLAVE2": "FR"})
    assert list(read_out(out / "中排座椅_54.csv").columns) == ["time_FL", "rssi_FL", "errcode_FL"]


def test_conflicting_mapping_rejected(folder: Path):
    with pytest.raises(ValueError):
        run(folder, {"SLAVE1": "FL", "SLAVE2": "FL"})


def test_gbk_input_and_errcode_variants(tmp_path: Path):
    path = write(tmp_path / "a.csv", "时间, ErrCode ,备注\n1, 0 ,好\n2,0.0,好\n3,,坏\n4,x,坏\n5,2,坏\n", encoding="gbk")
    df = read_csv_text(path)
    assert list(df.columns) == ["时间", "ErrCode", "备注"]
    data = load_anchor([path])
    assert data.total_rows == 5
    assert data.valid["时间"].tolist() == ["1", "2"]


def test_missing_errcode_column_reported(tmp_path: Path):
    data = load_anchor([write(tmp_path / "a.csv", "time,rssi\n1,2\n")])
    assert data.valid is None and "errcode" in data.error


def test_multiple_files_same_anchor_concatenated(tmp_path: Path):
    a = write(tmp_path / "a.csv", "v,errcode\n1,0\n2,1\n")
    b = write(tmp_path / "b.csv", "v,errcode\n3,0\n")
    assert load_anchor([a, b]).valid["v"].tolist() == ["1", "3"]
