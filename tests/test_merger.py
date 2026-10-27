from pathlib import Path

import pandas as pd
import pytest

from carinout.merger import (
    COMBINED_FILENAME,
    JOIN_OUTER,
    SUMMARY_FILENAME,
    MergeOptions,
    align_by_ts,
    build_dataset,
    load_anchor,
    read_csv_text,
    unsynced_anchors,
)
from carinout.parser import group_by_point, scan_folder


def write(path: Path, text: str, encoding: str = "utf-8") -> Path:
    path.write_text(text, encoding=encoding)
    return path


def read_out(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def frame(text: str) -> pd.DataFrame:
    rows = [line.split(",") for line in text.strip().splitlines()]
    return pd.DataFrame(rows[1:], columns=rows[0]).astype(str)


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    # 有效 ts：SLAVE1 = 100, 300, 400；SLAVE2 = 100, 300, 500（乱序）
    write(src / "座椅-53-[COM1][SLAVE1] 2026-10-23 15-44-12.743.csv",
          "ts,rssi,errcode\n100,-50,0\n200,-51,3\n300,-52,0\n400,-53.50,0\n")
    write(src / "座椅-53-[COM2][SLAVE2] 2026-10-23 15-44-12.770.csv",
          "ts,rssi,errcode\n500,-65,0\n100,-60,0\n400,-63,1\n300,-62,0\n")
    write(src / "座椅-54-[COM1][SLAVE1] 2026-10-23 15-50-00.000.csv", "ts,rssi,errcode\n100,-70,0\n")
    write(src / "车外-1-[COM2][SLAVE2] 2026-10-23 16-00-00.000.csv", "ts,rssi,errcode\n7,-80,0\n8,-81,0\n")
    write(src / "车外-1-[COM3][SLAVE3] 2026-10-23 16-00-00.000.csv", "ts,rssi,errcode\n7,-90,0\n")
    return src


def run(src: Path, mapping: dict[str, str], labels: dict[str, int] | None = None, **opts):
    parsed, _ = scan_folder(src)
    out = src.parent / "out"
    results = build_dataset(
        group_by_point(parsed), mapping, ["FL", "FR", "IF"], out, MergeOptions(**opts), scene_labels=labels
    )
    return out, results


def test_align_by_ts_inner(folder: Path):
    out, results = run(folder, {"SLAVE1": "FL", "SLAVE2": "FR"})
    df = read_out(out / "座椅_53.csv")
    assert list(df.columns) == ["ts", "rssi_FL", "errcode_FL", "rssi_FR", "errcode_FR"]
    # 只保留两个锚点都有 errcode=0 数据的 ts，按 ts 升序，数值文本保持原样
    assert df["ts"].tolist() == ["100", "300"]
    assert df["rssi_FL"].tolist() == ["-50", "-52"]
    assert df["rssi_FR"].tolist() == ["-60", "-62"]
    assert results[0].rows == 2


def test_align_by_ts_outer(folder: Path):
    out, _ = run(folder, {"SLAVE1": "FL", "SLAVE2": "FR"}, join=JOIN_OUTER)
    df = read_out(out / "座椅_53.csv")
    assert df["ts"].tolist() == ["100", "300", "400", "500"]
    assert df["rssi_FL"].tolist() == ["-50", "-52", "-53.50", ""]
    assert df["rssi_FR"].tolist() == ["-60", "-62", "", "-65"]


def test_numeric_ts_compared_by_value():
    a = frame("ts,v\n100,1\n200,2")
    b = frame("ts,v\n100.0,3\n200.0,4")
    out = align_by_ts([("A", a), ("B", b)])
    assert out["ts"].tolist() == ["100", "200"] and out["v_B"].tolist() == ["3", "4"]


def test_text_ts_exact_match():
    a = frame("ts,v\n15:44:12.743,1\n15:44:12.843,2")
    b = frame("ts,v\n15:44:12.843,3")
    out = align_by_ts([("A", a), ("B", b)])
    assert out.to_dict("list") == {"ts": ["15:44:12.843"], "v_A": ["2"], "v_B": ["3"]}


def test_duplicate_ts_paired_in_order():
    a = frame("ts,v\n1,a1\n1,a2\n2,a3")
    b = frame("ts,v\n1,b1\n2,b3\n1,b2")
    out = align_by_ts([("A", a), ("B", b)])
    assert out["v_A"].tolist() == ["a1", "a2", "a3"]
    assert out["v_B"].tolist() == ["b1", "b2", "b3"]


def test_tolerance_nearest_match():
    a = frame("ts,v\n1000,a1\n1100,a2\n1200,a3")
    b = frame("ts,v\n1003,b1\n1098,b2\n1300,b3")
    inner = align_by_ts([("A", a), ("B", b)], tolerance=5)
    assert list(inner.columns) == ["ts", "ts_A", "v_A", "ts_B", "v_B"]
    assert inner["ts"].tolist() == ["1000", "1100"]
    assert inner["ts_B"].tolist() == ["1003", "1098"]
    outer = align_by_ts([("A", a), ("B", b)], join=JOIN_OUTER, tolerance=5)
    assert outer["ts"].tolist() == ["1000", "1100", "1200"] and pd.isna(outer["v_B"].iloc[2])


def test_tolerance_requires_numeric_ts(folder: Path, tmp_path: Path):
    with pytest.raises(ValueError):
        align_by_ts([("A", frame("ts,v\nx,1")), ("B", frame("ts,v\nx,2"))], tolerance=1)


def test_low_match_rate_flagged(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    write(src / "s-1-[SLAVE1].csv", "ts,errcode\n" + "".join(f"{i},0\n" for i in range(10)))
    write(src / "s-1-[SLAVE2].csv", "ts,errcode\n" + "".join(f"{i}.5,0\n" for i in range(10)))
    _, results = run(src, {"SLAVE1": "FL", "SLAVE2": "FR"})
    assert results[0].output is None and "ts 对不上" in results[0].status
    write(src / "s-1-[SLAVE2].csv", "ts,errcode\n0,0\n" + "".join(f"{i}.5,0\n" for i in range(9)))
    _, results = run(src, {"SLAVE1": "FL", "SLAVE2": "FR"})
    assert results[0].rows == 1 and "对不上" in results[0].status


def test_independent_packet_loss_not_flagged_as_unsynced():
    # 三个锚点共用同一时钟，但各自丢掉不同的 30% 数据：交集变小，不应报 ts 不同步
    frames = []
    for name, drop in (("A", {1, 4, 7, 10, 13, 16}), ("B", {2, 5, 8, 11, 14, 17}), ("C", {3, 6, 9, 12, 15, 18})):
        frames.append((name, frame("ts,v\n" + "".join(f"{i},x\n" for i in range(20) if i not in drop))))
    assert unsynced_anchors(frames) == []
    shifted = frame("ts,v\n" + "".join(f"{i}.5,x\n" for i in range(20)))
    assert unsynced_anchors(frames + [("D", shifted)]) == ["D"]


def test_combined_dataset_with_labels(folder: Path):
    out, _ = run(folder, {"SLAVE1": "FR", "SLAVE2": "IF", "SLAVE3": "FL"}, labels={"座椅": 1, "车外": 0},
                 join=JOIN_OUTER)
    df = read_out(out / COMBINED_FILENAME)
    # 第一个点位（座椅_53）没有 FL，FL 的列仍按锚点顺序排在最前
    assert list(df.columns) == [
        "ts", "rssi_FL", "errcode_FL", "rssi_FR", "errcode_FR", "rssi_IF", "errcode_IF", "scene", "point", "label",
    ]
    assert df.groupby("scene")["label"].unique().to_dict() == {"座椅": ["1"], "车外": ["0"]}
    assert len(df) == 4 + 1 + 2  # 座椅_53 + 座椅_54 + 车外_1
    row = df[(df["scene"] == "车外") & (df["ts"] == "7")].iloc[0]
    assert (row["point"], row["rssi_FL"], row["rssi_FR"], row["rssi_IF"]) == ("1", "-90", "", "-80")
    summary = read_out(out / SUMMARY_FILENAME)
    assert summary.set_index("场景")["标签"].to_dict() == {"座椅": "1", "车外": "0"}


def test_combined_dataset_skips_unlabeled_scenes(folder: Path):
    out, _ = run(folder, {"SLAVE1": "FL", "SLAVE2": "FR"}, labels={"座椅": 1})
    df = read_out(out / COMBINED_FILENAME)
    assert set(df["scene"]) == {"座椅"}
    assert (out / "车外_1.csv").exists()


def test_no_combined_without_labels(folder: Path):
    out, _ = run(folder, {"SLAVE1": "FL", "SLAVE2": "FR"})
    assert not (out / COMBINED_FILENAME).exists()


def test_column_order_follows_anchor_order(folder: Path):
    out, _ = run(folder, {"SLAVE1": "FR", "SLAVE2": "FL"})
    df = read_out(out / "座椅_53.csv")
    assert list(df.columns)[:2] == ["ts", "rssi_FL"] and df["rssi_FL"].tolist()[0] == "-60"


def test_unmapped_slave_ignored(folder: Path):
    out, _ = run(folder, {"SLAVE1": "FL"})
    assert list(read_out(out / "座椅_53.csv").columns) == ["ts", "rssi_FL", "errcode_FL"]


def test_require_complete_skips_incomplete_points(folder: Path):
    out, results = run(folder, {"SLAVE1": "FL", "SLAVE2": "FR"}, require_complete=True)
    assert (out / "座椅_53.csv").exists()
    assert not (out / "座椅_54.csv").exists()
    skipped = next(r for r in results if r.point == "54")
    assert "FR" in skipped.status


@pytest.fixture
def one_bad_anchor(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    write(src / "座椅-1-[SLAVE1].csv", "ts,rssi,errcode\n100,-50,0\n200,-51,0\n")
    write(src / "座椅-1-[SLAVE2].csv", "ts,rssi,errcode\n100,-60,0\n200,-61,0\n")
    write(src / "座椅-1-[SLAVE3].csv", "ts,rssi,errcode\n100,-70,7\n200,-71,7\n")  # 全部 errcode≠0
    write(src / "座椅-2-[SLAVE1].csv", "ts,rssi,errcode\n100,-50,0\n")
    write(src / "座椅-2-[SLAVE2].csv", "ts,rssi,errcode\n100,-60,0\n")
    write(src / "座椅-2-[SLAVE3].csv", "ts,rssi\n100,-70\n")  # 缺 errcode 列
    return src


def test_require_complete_reports_reason(one_bad_anchor: Path):
    _, results = run(one_bad_anchor, {"SLAVE1": "FL", "SLAVE2": "FR", "SLAVE3": "IF"}, require_complete=True)
    assert [r.output for r in results] == [None, None]
    assert "IF 无 errcode=0 的行（共 2 行）" in results[0].status
    assert "IF 读取失败（缺少 errcode 列）" in results[1].status
    assert all("只导出已映射锚点齐全" in r.status for r in results)


def test_anchor_without_valid_rows_does_not_empty_point(one_bad_anchor: Path):
    # 不要求锚点齐全时，没有有效数据的锚点和缺失的锚点一样被略过，而不是让 ts 交集变空
    out, results = run(one_bad_anchor, {"SLAVE1": "FL", "SLAVE2": "FR", "SLAVE3": "IF"})
    df = read_out(out / "座椅_1.csv")
    assert list(df.columns) == ["ts", "rssi_FL", "errcode_FL", "rssi_FR", "errcode_FR"]
    assert df["ts"].tolist() == ["100", "200"]
    assert "IF 无 errcode=0 的行" in results[0].status
    assert results[1].rows == 1 and "IF 读取失败" in results[1].status


def test_conflicting_mapping_rejected(folder: Path):
    with pytest.raises(ValueError):
        run(folder, {"SLAVE1": "FL", "SLAVE2": "FL"})


def test_gbk_input_and_errcode_variants(tmp_path: Path):
    path = write(tmp_path / "a.csv", "TS, ErrCode ,备注\n1, 0 ,好\n2,0.0,好\n3,,坏\n4,x,坏\n5,2,坏\n", encoding="gbk")
    df = read_csv_text(path)
    assert list(df.columns) == ["TS", "ErrCode", "备注"]
    data = load_anchor([path])
    assert data.total_rows == 5
    assert data.valid["TS"].tolist() == ["1", "2"]


def test_hex_errcode(tmp_path: Path):
    rows = ["0x00000000", "0", " 0X0 ", "0x00000001", "0x0000000A", "0x", "0x0000000g", "0x00000000"]
    path = write(tmp_path / "a.csv", "ts,errcode\n" + "".join(f"{i},{c}\n" for i, c in enumerate(rows)))
    data = load_anchor([path])
    assert data.valid["ts"].tolist() == ["0", "1", "2", "7"]
    assert data.valid["errcode"].tolist() == ["0x00000000", "0", " 0X0 ", "0x00000000"]  # 原文本保留


def test_unrecognized_errcode_format_reported(tmp_path: Path):
    data = load_anchor([write(tmp_path / "a.csv", "ts,errcode\n1,OK\n2,FAIL\n")])
    assert data.valid is None and "无法识别" in data.error and "OK" in data.error


@pytest.mark.parametrize("header, missing", [("ts,rssi", "errcode"), ("time,errcode", "ts")])
def test_missing_required_column_reported(tmp_path: Path, header: str, missing: str):
    data = load_anchor([write(tmp_path / "a.csv", f"{header}\n1,0\n")])
    assert data.valid is None and missing in data.error


def test_multiple_files_same_anchor_concatenated(tmp_path: Path):
    a = write(tmp_path / "a.csv", "ts,v,errcode\n1,1,0\n2,2,1\n")
    b = write(tmp_path / "b.csv", "ts,v,errcode\n3,3,0\n")
    assert load_anchor([a, b]).valid["v"].tolist() == ["1", "3"]
