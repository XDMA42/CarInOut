"""按点位合并各锚点的 CSV。

同一 (场景, 点位) 下不同锚点的 CSV 横向拼接：每个锚点只保留 errcode == 0 的行，
列名加上 “_锚点名” 后缀，按行号对齐。所有值按原始文本读写，不改变数值格式。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ERRCODE_COLUMN = "errcode"
INPUT_ENCODINGS = ("utf-8-sig", "gb18030")
SUMMARY_FILENAME = "_summary.csv"

ALIGN_MIN = "min"  # 按最短锚点截断，每行都包含所有锚点的数据
ALIGN_MAX = "max"  # 保留全部行，较短的锚点补空


class CsvReadError(Exception):
    pass


@dataclass
class AnchorData:
    valid: pd.DataFrame | None
    total_rows: int = 0
    error: str = ""

    @property
    def valid_rows(self) -> int:
        return 0 if self.valid is None else len(self.valid)


@dataclass
class MergeOptions:
    align: str = ALIGN_MIN
    # 为 True 时，缺少任一已映射锚点（或该锚点没有有效数据）的点位不导出
    require_complete: bool = False
    output_encoding: str = "utf-8-sig"


@dataclass
class PointResult:
    scene: str
    point: str
    output: Path | None
    rows: int
    status: str
    anchors: dict[str, AnchorData] = field(default_factory=dict)


def read_csv_text(path: Path) -> pd.DataFrame:
    """以字符串方式读取 CSV，依次尝试 UTF-8 与 GB18030 编码。"""
    last_exc: Exception | None = None
    for enc in INPUT_ENCODINGS:
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
        except UnicodeDecodeError as exc:
            last_exc = exc
            continue
        except pd.errors.EmptyDataError as exc:
            raise CsvReadError("文件为空") from exc
        except (OSError, pd.errors.ParserError) as exc:
            raise CsvReadError(str(exc)) from exc
        df.columns = [str(c).strip() for c in df.columns]
        return df
    raise CsvReadError(f"无法识别文件编码：{last_exc}")


def filter_valid(df: pd.DataFrame) -> pd.DataFrame:
    """只保留 errcode 为 0 的行（列名不区分大小写）。"""
    col = next((c for c in df.columns if c.lower() == ERRCODE_COLUMN), None)
    if col is None:
        raise CsvReadError(f"缺少 {ERRCODE_COLUMN} 列")
    codes = pd.to_numeric(df[col].str.strip(), errors="coerce")
    return df[codes == 0].reset_index(drop=True)


def load_anchor(paths: list[Path]) -> AnchorData:
    """读取同一点位、同一锚点的一个或多个 CSV（多个时按顺序纵向拼接）。"""
    frames = []
    total = 0
    try:
        for path in paths:
            df = read_csv_text(path)
            total += len(df)
            frames.append(filter_valid(df))
    except CsvReadError as exc:
        return AnchorData(None, total, f"{path.name}: {exc}")
    return AnchorData(pd.concat(frames, ignore_index=True), total)


def merge_frames(frames: list[tuple[str, pd.DataFrame]], align: str = ALIGN_MIN) -> pd.DataFrame:
    """横向拼接 [(锚点名, 有效数据)]，列名加 “_锚点名” 后缀。"""
    parts = [df.add_suffix(f"_{name}") for name, df in frames]
    if not parts:
        return pd.DataFrame()
    if align == ALIGN_MIN:
        n = min(len(p) for p in parts)
        parts = [p.iloc[:n] for p in parts]
    return pd.concat(parts, axis=1)


def resolve_anchors(
    slave_files: dict[str, list[Path]], slave_to_anchor: dict[str, str]
) -> dict[str, list[Path]]:
    """把 {SLAVE: 文件} 转成 {锚点名: 文件}，未映射的 SLAVE 被忽略。"""
    by_anchor: dict[str, list[Path]] = {}
    for slave, paths in slave_files.items():
        anchor = slave_to_anchor.get(slave)
        if anchor:
            by_anchor.setdefault(anchor, []).extend(paths)
    return by_anchor


def check_mapping(slave_to_anchor: dict[str, str]) -> None:
    seen: dict[str, str] = {}
    for slave, anchor in slave_to_anchor.items():
        if anchor in seen:
            raise ValueError(f"{seen[anchor]} 和 {slave} 都映射到了锚点 {anchor}")
        seen[anchor] = slave


def build_dataset(
    groups: dict[tuple[str, str], dict[str, list[Path]]],
    slave_to_anchor: dict[str, str],
    anchor_order: list[str],
    out_dir: Path,
    options: MergeOptions | None = None,
    progress: Callable[[int, int], None] | None = None,
    log: Callable[[str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> list[PointResult]:
    """对每个 (场景, 点位) 合并并写出 “场景_点位.csv”，最后写出 _summary.csv 汇总。

    slave_to_anchor: SLAVE 标识 → 锚点名；anchor_order: 输出列中锚点的先后顺序。
    """
    options = options or MergeOptions()
    log = log or (lambda _msg: None)
    check_mapping(slave_to_anchor)
    expected = [a for a in anchor_order if a in slave_to_anchor.values()]
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[PointResult] = []
    total = len(groups)
    for i, ((scene, point), slave_files) in enumerate(groups.items(), start=1):
        if cancelled and cancelled():
            log("已取消")
            break
        result = _merge_point(scene, point, resolve_anchors(slave_files, slave_to_anchor),
                              expected, out_dir, options)
        results.append(result)
        for name, data in result.anchors.items():
            if data.error:
                log(f"[{scene}_{point}] 锚点 {name} 读取失败：{data.error}")
        log(f"[{scene}_{point}] {result.status}")
        if progress:
            progress(i, total)

    _write_summary(results, expected, out_dir / SUMMARY_FILENAME, options.output_encoding)
    return results


def _merge_point(
    scene: str,
    point: str,
    anchor_files: dict[str, list[Path]],
    expected: list[str],
    out_dir: Path,
    options: MergeOptions,
) -> PointResult:
    anchors = {name: load_anchor(anchor_files[name]) for name in expected if name in anchor_files}
    result = PointResult(scene, point, None, 0, "", anchors)

    missing = [name for name in expected if name not in anchors]
    unusable = [name for name, data in anchors.items() if data.valid_rows == 0]
    if options.require_complete and (missing or unusable):
        parts = []
        if missing:
            parts.append("缺少锚点 " + ",".join(missing))
        if unusable:
            parts.append("无有效数据 " + ",".join(unusable))
        result.status = "跳过：" + "；".join(parts)
        return result

    frames = [(name, data.valid) for name, data in anchors.items() if data.valid is not None]
    merged = merge_frames(frames, options.align)
    if merged.empty:
        result.status = "跳过：合并后没有有效数据" + (f"（{','.join(unusable)} 无 errcode=0 的行）" if unusable else "")
        return result

    output = out_dir / f"{scene}_{point}.csv"
    merged.to_csv(output, index=False, encoding=options.output_encoding)
    result.output, result.rows = output, len(merged)
    result.status = f"已导出 {len(merged)} 行"
    if missing:
        result.status += "（缺少锚点 " + ",".join(missing) + "）"
    return result


def _write_summary(results: list[PointResult], anchor_order: list[str], path: Path, encoding: str) -> None:
    rows = []
    for r in results:
        row = {
            "场景": r.scene,
            "点位": r.point,
            "输出文件": r.output.name if r.output else "",
            "输出行数": r.rows,
            "状态": r.status,
        }
        for name in anchor_order:
            data = r.anchors.get(name)
            row[f"{name}_有效行"] = data.valid_rows if data else ""
            row[f"{name}_原始行"] = data.total_rows if data else ""
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False, encoding=encoding)
