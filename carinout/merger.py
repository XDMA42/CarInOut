"""按点位合并各锚点的 CSV，并生成带场景标签的总表。

同一 (场景, 点位) 下不同锚点的 CSV 按 ts 列横向对齐：每个锚点只保留 errcode == 0 的行，
输出首列为 ts，其余列名加上 “_锚点名” 后缀。所有值按原始文本读写，不改变数值格式。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

TS_COLUMN = "ts"
ERRCODE_COLUMN = "errcode"
INPUT_ENCODINGS = ("utf-8-sig", "gb18030")
SUMMARY_FILENAME = "_summary.csv"
COMBINED_FILENAME = "_dataset.csv"

JOIN_INNER = "inner"  # 只保留所有锚点都有数据的 ts
JOIN_OUTER = "outer"  # 保留任一锚点出现过的 ts，缺数据的锚点补空

# 总表中追加的列；label：0 = 车外，1 = 车内
SCENE_COLUMN = "scene"
POINT_COLUMN = "point"
LABEL_COLUMN = "label"
LABEL_OUTSIDE = 0
LABEL_INSIDE = 1

_KEY = "__ts_key"
_OCC = "__ts_occ"


class CsvReadError(Exception):
    pass


@dataclass
class AnchorData:
    valid: pd.DataFrame | None
    total_rows: int = 0
    error: str = ""  # 读取失败的原因
    error_file: str = ""  # 读取失败的文件名

    @property
    def valid_rows(self) -> int:
        return 0 if self.valid is None else len(self.valid)

    def problem(self) -> str:
        """该锚点不可用的原因；有有效数据时返回空串。"""
        if self.error:
            return f"读取失败（{self.error}）"
        if self.valid_rows == 0:
            return f"无 errcode=0 的行（共 {self.total_rows} 行）"
        return ""


@dataclass
class MergeOptions:
    join: str = JOIN_INNER
    # 0 表示 ts 必须精确相等；> 0 时按最近的 ts 匹配，单位与 ts 列相同
    ts_tolerance: float = 0.0
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


def dataset_folder(input_dir: Path, base: Path | None = None) -> Path:
    """导出目录：base / 原始数据文件夹名；未指定 base 时为与原始数据文件夹同级的 merged 文件夹。"""
    input_dir = input_dir.resolve()
    base = base or input_dir.parent / "merged"
    return base / (input_dir.name or "dataset")


def find_column(df: pd.DataFrame, name: str) -> str | None:
    """不区分大小写查找列名。"""
    return next((c for c in df.columns if c.lower() == name), None)


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
    """只保留 errcode 为 0 的行（列名不区分大小写）。

    errcode 可以是十进制（0、0.0）或十六进制（0x00000000）；无法识别的值视为非 0。
    """
    col = find_column(df, ERRCODE_COLUMN)
    if col is None:
        raise CsvReadError(f"缺少 {ERRCODE_COLUMN} 列")
    codes = df[col].str.strip()
    decimal = pd.to_numeric(codes, errors="coerce")
    is_hex = codes.str.fullmatch(r"[+-]?0[xX][0-9a-fA-F]+")
    if len(df) and not (decimal.notna() | is_hex).any():
        raise CsvReadError(f"{ERRCODE_COLUMN} 的值无法识别（如 “{codes.iat[0]}”），应为 0 或 0x00000000 这样的格式")
    is_zero = (decimal == 0) | codes.str.fullmatch(r"[+-]?0[xX]0+")
    return df[is_zero].reset_index(drop=True)


def load_anchor(paths: list[Path]) -> AnchorData:
    """读取同一点位、同一锚点的一个或多个 CSV（多个时按顺序纵向拼接）。"""
    frames = []
    total = 0
    try:
        for path in paths:
            df = read_csv_text(path)
            if find_column(df, TS_COLUMN) is None:
                raise CsvReadError(f"缺少 {TS_COLUMN} 列")
            total += len(df)
            frames.append(filter_valid(df))
    except CsvReadError as exc:
        return AnchorData(None, total, str(exc), path.name)
    return AnchorData(pd.concat(frames, ignore_index=True), total)


def anchor_columns(name: str, df: pd.DataFrame, tolerance: float = 0.0) -> list[str]:
    """该锚点在合并结果中的列名（不含公共的 ts 列）。"""
    ts_col = find_column(df, TS_COLUMN)
    cols = [f"{c}_{name}" for c in df.columns if c != ts_col]
    return ([f"{TS_COLUMN}_{name}"] if tolerance > 0 else []) + cols


def align_by_ts(
    frames: list[tuple[str, pd.DataFrame]], join: str = JOIN_INNER, tolerance: float = 0.0
) -> pd.DataFrame:
    """按 ts 横向对齐 [(锚点名, 有效数据)]，输出首列为 ts、按 ts 升序。

    tolerance == 0：ts 精确相等才对齐（数值型 ts 按数值比较）；同一锚点同一 ts 出现多次时，
                    按出现顺序与其他锚点的同 ts 行一一配对。
    tolerance > 0：以第一个锚点的 ts 为基准，其他锚点取 |Δts| ≤ tolerance 的最近一行，
                   ts 必须是数值；额外保留各锚点实际的 “ts_锚点名” 列。
    """
    if not frames:
        return pd.DataFrame()
    names = [name for name, _ in frames]
    ts_cols = [f"{TS_COLUMN}_{name}" for name in names]
    parts = []
    for name, df in frames:
        col = find_column(df, TS_COLUMN)
        ts = df[col].str.strip()
        keep = ts != ""
        part = df.loc[keep].drop(columns=[col]).add_suffix(f"_{name}")
        part.insert(0, f"{TS_COLUMN}_{name}", ts[keep])
        parts.append(part.reset_index(drop=True))

    keys, numeric = _ts_keys([p.iloc[:, 0] for p in parts])
    if tolerance > 0:
        if not numeric:
            raise ValueError("ts 不是数值，无法按容差匹配")
        for p, k in zip(parts, keys):
            p[_KEY] = k.astype("float64")
        merged = parts[0].sort_values(_KEY, kind="stable")
        for p in parts[1:]:
            merged = pd.merge_asof(
                merged, p.sort_values(_KEY, kind="stable"), on=_KEY, direction="nearest", tolerance=tolerance
            )
        if join == JOIN_INNER:
            merged = merged.dropna(subset=ts_cols)
        out = merged.drop(columns=[_KEY])
        out.insert(0, TS_COLUMN, out[ts_cols[0]])
        return out.reset_index(drop=True)

    for p, k in zip(parts, keys):
        p[_KEY] = k.to_numpy()
        p[_OCC] = p.groupby(_KEY, sort=False).cumcount()
    merged = parts[0]
    for p in parts[1:]:
        merged = merged.merge(p, on=[_KEY, _OCC], how=join)
    merged = merged.sort_values([_KEY, _OCC], kind="stable")
    ts = merged[ts_cols[0]]
    for col in ts_cols[1:]:
        ts = ts.fillna(merged[col])
    out = merged.drop(columns=ts_cols + [_KEY, _OCC])
    out.insert(0, TS_COLUMN, ts)
    return out.reset_index(drop=True)


def _ts_keys(ts_list: list[pd.Series]) -> tuple[list[pd.Series], bool]:
    """ts 全是数值时按数值比较（"100" 与 "100.0" 相等），否则按文本比较。返回 (比较键, 是否数值)。"""
    keys = [pd.to_numeric(t, errors="coerce") for t in ts_list]
    if not all(k.notna().all() for k in keys):
        return list(ts_list), False
    if not all(pd.api.types.is_integer_dtype(k) for k in keys):
        keys = [k.astype("float64") for k in keys]
    return keys, True


def unsynced_anchors(frames: list[tuple[str, pd.DataFrame]], threshold: float = 0.2) -> list[str]:
    """找出 ts 几乎和其他锚点都对不上的锚点（精确匹配时用来提示 ts 不同步）。

    只比较“该锚点的 ts 是否出现在任一其他锚点中”，各锚点独立丢包不会误报。
    """
    if len(frames) < 2:
        return []
    ts_list = []
    for _, df in frames:
        ts = df[find_column(df, TS_COLUMN)].str.strip()
        ts_list.append(ts[ts != ""])
    sets = [set(k) for k in _ts_keys(ts_list)[0]]
    result = []
    for i, (name, _) in enumerate(frames):
        others = set().union(*(s for j, s in enumerate(sets) if j != i))
        if sets[i] and len(sets[i] & others) < threshold * len(sets[i]):
            result.append(name)
    return result


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
    scene_labels: dict[str, int] | None = None,
    progress: Callable[[int, int], None] | None = None,
    log: Callable[[str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> list[PointResult]:
    """对每个 (场景, 点位) 合并并写出 “场景_点位.csv”，最后写出 _summary.csv 汇总。

    slave_to_anchor: SLAVE 标识 → 锚点名；anchor_order: 输出列中锚点的先后顺序。
    scene_labels: 场景 → 标签（0 车外 / 1 车内）；不为 None 时另写出总表 _dataset.csv，
                  只包含已标注场景，末尾追加 scene、point、label 三列。
    """
    options = options or MergeOptions()
    log = log or (lambda _msg: None)
    check_mapping(slave_to_anchor)
    expected = [a for a in anchor_order if a in slave_to_anchor.values()]
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[PointResult] = []
    labeled_frames: list[pd.DataFrame] = []
    columns_by_anchor: dict[str, list[str]] = {name: [] for name in expected}
    total = len(groups)
    for i, ((scene, point), slave_files) in enumerate(groups.items(), start=1):
        if cancelled and cancelled():
            log("已取消")
            break
        result, merged = _merge_point(
            scene, point, resolve_anchors(slave_files, slave_to_anchor), expected, out_dir, options
        )
        results.append(result)
        for name, data in result.anchors.items():
            if data.error:
                log(f"[{scene}_{point}] 锚点 {name} 读取失败：{data.error_file}: {data.error}")
        log(f"[{scene}_{point}] {result.status}")
        if merged is not None and scene_labels is not None and scene in scene_labels:
            labeled_frames.append(
                merged.assign(**{SCENE_COLUMN: scene, POINT_COLUMN: point, LABEL_COLUMN: scene_labels[scene]})
            )
            for name, data in result.anchors.items():
                if data.valid is not None:
                    seen = columns_by_anchor[name]
                    seen.extend(c for c in anchor_columns(name, data.valid, options.ts_tolerance) if c not in seen)
        if progress:
            progress(i, total)

    _write_summary(results, expected, scene_labels, out_dir / SUMMARY_FILENAME, options.output_encoding)
    if scene_labels is not None:
        columns = (
            [TS_COLUMN]
            + [c for name in expected for c in columns_by_anchor[name]]
            + [SCENE_COLUMN, POINT_COLUMN, LABEL_COLUMN]
        )
        combined = (
            pd.concat(labeled_frames, ignore_index=True).reindex(columns=columns)
            if labeled_frames
            else pd.DataFrame(columns=columns)
        )
        combined.to_csv(out_dir / COMBINED_FILENAME, index=False, encoding=options.output_encoding)
        n_scenes = len({f[SCENE_COLUMN].iat[0] for f in labeled_frames})
        log(f"总表 {COMBINED_FILENAME}：{n_scenes} 个场景，{len(labeled_frames)} 个点位，共 {len(combined)} 行")
    return results


def _merge_point(
    scene: str,
    point: str,
    anchor_files: dict[str, list[Path]],
    expected: list[str],
    out_dir: Path,
    options: MergeOptions,
) -> tuple[PointResult, pd.DataFrame | None]:
    anchors = {name: load_anchor(anchor_files[name]) for name in expected if name in anchor_files}
    result = PointResult(scene, point, None, 0, "", anchors)

    missing = [name for name in expected if name not in anchors]
    problems = (["缺少锚点 " + ",".join(missing)] if missing else []) + [
        f"{name} {data.problem()}" for name, data in anchors.items() if data.problem()
    ]
    if options.require_complete and problems:
        result.status = "跳过（已勾选“只导出已映射锚点齐全的点位”）：" + "；".join(problems)
        return result, None

    # 没有有效数据的锚点与缺失的锚点同样处理：不参与对齐，否则会把整个点位的交集清空
    frames = [(name, data.valid) for name, data in anchors.items() if data.valid_rows > 0]
    if not frames:
        result.status = "跳过：没有可用的锚点数据（" + "；".join(problems) + "）"
        return result, None
    try:
        merged = align_by_ts(frames, options.join, options.ts_tolerance)
    except ValueError as exc:
        result.status = f"失败：{exc}"
        return result, None
    if merged.empty:
        hint = "可调大 ts 容差" if options.ts_tolerance > 0 else "可设置 ts 容差"
        result.status = f"跳过：按 ts 对齐后没有数据（各锚点的 ts 对不上，{hint}）"
        return result, None

    output = out_dir / f"{scene}_{point}.csv"
    merged.to_csv(output, index=False, encoding=options.output_encoding)
    result.output, result.rows = output, len(merged)
    result.status = f"已导出 {len(merged)} 行"
    if problems:
        result.status += "（" + "；".join(problems) + "）"
    if options.ts_tolerance == 0:
        unsynced = unsynced_anchors(frames)
        if unsynced:
            result.status += f"（注意：{','.join(unsynced)} 的 ts 与其他锚点几乎对不上，可能不同步，可设置 ts 容差）"
    return result, merged


def _write_summary(
    results: list[PointResult],
    anchor_order: list[str],
    scene_labels: dict[str, int] | None,
    path: Path,
    encoding: str,
) -> None:
    rows = []
    for r in results:
        row = {
            "场景": r.scene,
            "点位": r.point,
            "标签": (scene_labels or {}).get(r.scene, ""),
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
