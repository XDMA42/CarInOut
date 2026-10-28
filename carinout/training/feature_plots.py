"""数据集特征分析：各锚点特征的分布（车内 / 车外对比）与相关性，输出 PNG 和 CSV。

总表的特征列名为 “<特征>_<锚点名>”（见 merger.build_dataset），如 dis_sm_FL、dis_main_sm_FL、rssi_init_IR。
锚点名取自 errcode_<锚点名> 列（每个锚点的原始 CSV 都有 errcode 列），据此把列名拆成 (特征, 锚点)，
特征名本身可以含下划线；没有 errcode 列时按最后一个下划线拆分。

命令行：python plot_features.py 数据集文件夹 [--features dis_sm,dis_main_sm,rssi_init]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from ..merger import COMBINED_FILENAME, ERRCODE_COLUMN, LABEL_COLUMN, LABEL_INSIDE, LABEL_OUTSIDE
from .data import CLASS_NAMES, feature_candidates, load_table, parse_labels, to_numeric
from .plots import GRID, INK, INK_2, LINE_WIDTH, MUTED, SERIES, SURFACE, _placeholder, _style, new_figure

METHODS = {"pearson": "Pearson", "spearman": "Spearman"}
MIN_PAIRS = 10  # 两列同时有值的行数少于此数时不计算相关系数
HIST_BINS = 40
GRID_COLS = 5  # 按锚点分图时每行最多几张
SCATTER_MAX_POINTS = 2000  # 散点图每个锚点、每个类别最多画的点数（随机抽样），避免点太密
SCATTER_MAX_PAIRS = 15  # 特征两两组合超过这么多对时不画散点图
ANNOTATE_MAX = 30  # 相关矩阵超过这么多列时不在格子里写数值
STATS_FILENAME = "stats.csv"
CORR_FILENAME = "correlation.csv"

CLASS_COLORS = {LABEL_INSIDE: SERIES[0], LABEL_OUTSIDE: SERIES[1]}
CLASS_KEYS = {LABEL_INSIDE: "inside", LABEL_OUTSIDE: "outside"}  # stats.csv 的列名
ALL_ROWS = "全部"
# 发散色阶：蓝（负相关）← 灰（0）→ 红（正相关），两侧亮度对称
DIVERGING = (
    "#0d366b", "#1c5cab", "#2a78d6", "#6da7ec", "#b7d3f6",
    "#f0efec",
    "#f5c6c3", "#e9827e", "#d33a3b", "#a8262a", "#6e1517",
)


@dataclass
class FeatureTable:
    """按 (特征, 锚点) 拆分后的数值数据。"""

    name: str  # 数据集简称，用于图表标题
    values: pd.DataFrame  # 列为原列名（如 dis_sm_FL），float，空值为 NaN
    labels: pd.Series | None  # 0 / 1，与 values 逐行对应；数据中没有标签列时为 None
    features: list[str]  # 特征名（不带锚点后缀），按列顺序
    anchors: list[str]  # 锚点名，按列顺序（即车模上的顺序）
    columns: dict[tuple[str, str], str]  # (特征, 锚点) → 列名，按特征分组；某锚点没有该特征时不在字典中
    label_column: str = LABEL_COLUMN
    notes: list[str] = field(default_factory=list)

    def column(self, feature: str, anchor: str) -> str | None:
        return self.columns.get((feature, anchor))

    def has_classes(self) -> bool:
        """车内、车外都有数据（才能按类别对比、计算与标签的相关系数）。"""
        return self.labels is not None and self.labels.nunique() == 2

    def groups(self) -> list[tuple[str, str, np.ndarray]]:
        """分组对比用的 [(名称, 颜色, 行掩码)]：有标签时为车内、车外，否则为全部数据。"""
        if self.labels is None:
            return [(ALL_ROWS, SERIES[0], np.ones(len(self.values), dtype=bool))]
        result = []
        for label in (LABEL_INSIDE, LABEL_OUTSIDE):
            mask = (self.labels == label).to_numpy()
            if mask.any():
                result.append((CLASS_NAMES[label], CLASS_COLORS[label], mask))
        return result


def resolve_dataset(path: str | Path) -> Path:
    """含 _dataset.csv 的文件夹 → 该文件；csv 文件 → 本身。"""
    path = Path(path)
    if path.is_dir():
        if not (path / COMBINED_FILENAME).is_file():
            raise ValueError(f"{path} 中没有 {COMBINED_FILENAME}")
        return path / COMBINED_FILENAME
    if not path.is_file():
        raise ValueError(f"找不到 {path}")
    return path


def default_output(csv_path: Path) -> Path:
    """默认输出位置：数据集文件夹下的 feature_plots（其他 csv 为 feature_plots_文件名）。"""
    suffix = "" if csv_path.name == COMBINED_FILENAME else f"_{csv_path.stem}"
    return csv_path.parent / f"feature_plots{suffix}"


def detect_anchors(columns) -> list[str]:
    """从 errcode_<锚点名> 列得到锚点名（列名前缀不区分大小写），按列顺序。"""
    prefix = f"{ERRCODE_COLUMN}_"
    return [c[len(prefix):] for c in columns if c.lower().startswith(prefix) and len(c) > len(prefix)]


def split_column(column: str, anchors: list[str]) -> tuple[str, str] | None:
    """把列名拆成 (特征, 锚点)。有多个锚点名都能匹配时取最长的；anchors 为空时按最后一个下划线拆分。"""
    if not anchors:
        feature, _, anchor = column.rpartition("_")
        return (feature, anchor) if feature and anchor else None
    for anchor in sorted(anchors, key=len, reverse=True):
        suffix = f"_{anchor}"
        if column.endswith(suffix) and len(column) > len(suffix):
            return column[: -len(suffix)], anchor
    return None


def load_features(
    table: pd.DataFrame,
    features: list[str] | None = None,
    anchors: list[str] | None = None,
    label_column: str = LABEL_COLUMN,
    name: str = "",
) -> FeatureTable:
    """从总表中取出数值特征列并按 (特征, 锚点) 拆分。features / anchors 为空时取全部。"""
    known = detect_anchors(table.columns)
    columns: dict[tuple[str, str], str] = {}
    unmatched = []
    for info in feature_candidates(table, label_column):
        parts = split_column(info.name, known)
        if parts is None:
            unmatched.append(info.name)
        else:
            columns[parts] = info.name
    all_features = list(dict.fromkeys(f for f, _ in columns))
    used = {a for _, a in columns}
    all_anchors = [a for a in (known or dict.fromkeys(a for _, a in columns)) if a in used]
    if not all_features:
        raise ValueError("数据中没有 “特征_锚点名” 格式的数值列")
    features = _choose(features, all_features, "特征")
    anchors = _choose(anchors, all_anchors, "锚点")
    columns = {(f, a): columns[(f, a)] for f in features for a in anchors if (f, a) in columns}
    if not columns:
        raise ValueError(f"所选锚点上没有这些特征的数据列：{'、'.join(features)}")

    notes = []
    if unmatched:
        notes.append(f"以下数值列不是 “特征_锚点名” 格式，未分析：{_shorten(unmatched)}")
    values = pd.DataFrame({col: to_numeric(table[col]) for col in columns.values()}, index=table.index)
    labels = None
    if label_column in table.columns:
        labels = parse_labels(table, label_column)
        unlabeled = labels.isna()
        if unlabeled.any():
            notes.append(f"{int(unlabeled.sum())} 行没有标签，已忽略")
        values, labels = values[~unlabeled], labels[~unlabeled].astype(np.int64)
        labels = labels.reset_index(drop=True)
        if labels.nunique() < 2:
            notes.append(f"数据中只有{CLASS_NAMES[int(labels.iat[0])]}的样本" if len(labels) else "")
    else:
        notes.append(f"数据中没有标签列 “{label_column}”，不区分车内 / 车外")
    if values.empty:
        raise ValueError("数据中没有可用的数据行")
    for f in features:
        lacking = [a for a in anchors if (f, a) not in columns]
        if lacking:
            notes.append(f"特征 {f} 在以下锚点没有数据列：{'、'.join(lacking)}")
    return FeatureTable(
        name, values.reset_index(drop=True), labels, features, anchors, columns, label_column,
        [n for n in notes if n],
    )


def correlation(ft: FeatureTable, method: str = "pearson") -> pd.DataFrame:
    """所有特征列两两之间的相关系数；有标签时最后加上标签列。

    缺失值按列对处理：每对列只用两列都有值的行，少于 MIN_PAIRS 行时为 NaN；常数列的相关系数也是 NaN。
    与标签（车内 = 1）的相关系数为正，表示该特征在车内时偏大。
    """
    frame = ft.values[list(ft.columns.values())]
    if ft.labels is not None:
        frame = frame.assign(**{ft.label_column: ft.labels.to_numpy(dtype=float)})
    with np.errstate(divide="ignore", invalid="ignore"):
        return frame.corr(method=method, min_periods=MIN_PAIRS)


def label_correlation(ft: FeatureTable, corr: pd.DataFrame) -> pd.DataFrame | None:
    """与标签的相关系数，行 = 特征，列 = 锚点；没有车内 / 车外两类数据时为 None。"""
    if not ft.has_classes():
        return None
    return _grid(ft, lambda col: corr.at[col, ft.label_column])


def describe(ft: FeatureTable, corr: pd.DataFrame) -> pd.DataFrame:
    """每个 (特征, 锚点) 一行的统计量；有标签时另列车内 / 车外各自的统计量和与标签的相关系数。"""
    rows = []
    for (feature, anchor), col in ft.columns.items():
        v = ft.values[col]
        row = {"feature": feature, "anchor": anchor, "column": col, "count": int(v.notna().sum()),
               "missing_ratio": float(v.isna().mean()), **_summary(v)}
        if ft.labels is not None:
            for label, key in CLASS_KEYS.items():
                part = v[(ft.labels == label).to_numpy()]
                row[f"count_{key}"] = int(part.notna().sum())
                row[f"mean_{key}"] = part.mean()
                row[f"std_{key}"] = part.std()
                row[f"median_{key}"] = part.median()
            row["corr_label"] = corr.at[col, ft.label_column] if ft.has_classes() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- 图表
def draw_distribution(fig: Figure, ft: FeatureTable, feature: str) -> None:
    """一个特征在各锚点上的分布：上方为各锚点的箱线图，下方每个锚点一张直方图（各类别分别归一化）。"""
    fig.clear()
    groups = ft.groups()
    anchors = ft.anchors
    nrows, ncols = _grid_shape(len(anchors))
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.1, nrows], hspace=0.04)
    for sub in (top, bottom):
        sub.set_facecolor(SURFACE)
    fig.suptitle(_title(ft, f"{feature} 在各锚点的分布"), x=0.01, ha="left", fontsize=12, color=INK)

    pooled = np.concatenate([_finite(ft.values[c]) for (f, _), c in ft.columns.items() if f == feature] or [[]])
    low, high = _display_range(pooled)

    ax = top.subplots()
    _style(ax, "箱体 25%–75%，横线为中位数，须线 5%–95%")
    width = 0.8 / len(groups)
    for k, (_, color, mask) in enumerate(groups):
        data, positions = [], []
        for i, anchor in enumerate(anchors):
            col = ft.column(feature, anchor)
            values = _finite(ft.values.loc[mask, col]) if col else np.array([])
            if len(values):
                data.append(values)
                positions.append(i + (k - (len(groups) - 1) / 2) * width)
        if data:
            ax.boxplot(
                data, positions=positions, widths=width * 0.75, whis=(5, 95), showfliers=False, patch_artist=True,
                manage_ticks=False,
                boxprops={"facecolor": color, "edgecolor": color, "linewidth": 1},
                medianprops={"color": SURFACE, "linewidth": 1.6},
                whiskerprops={"color": color, "linewidth": 1.2},
                capprops={"color": color, "linewidth": 1.2},
            )
    ax.set_xticks(range(len(anchors)), anchors)
    ax.tick_params(axis="x", labelsize=9, labelcolor=INK_2)
    ax.set_xlim(-0.6, len(anchors) - 0.4)
    if low < high:
        pad = (high - low) * 0.04
        ax.set_ylim(low - pad, high + pad)
    if len(groups) > 1 or ft.labels is not None:
        handles = [Line2D([], [], color=color, lw=6) for _, color, _ in groups]
        names = [f"{name}（{int(mask.sum()):,} 行）" for name, _, mask in groups]
        ax.legend(handles, names, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=len(groups), frameon=False,
                  fontsize=8, labelcolor=INK_2, handlelength=1.0, borderaxespad=0.2)

    edges = _bin_edges(pooled, low, high)
    axes = bottom.subplots(nrows, ncols, squeeze=False, sharex=True, sharey=True)
    for ax, anchor in zip(axes.flat, anchors):
        _style(ax, anchor)
        col = ft.column(feature, anchor)
        if col is None or not ft.values[col].notna().any():
            _placeholder(ax, "没有该列" if col is None else "全部为空")
            continue
        column = ft.values[col]
        for _, color, mask in groups:
            values = _finite(column[mask])
            if not len(values):
                continue
            shown = values[(values >= edges[0]) & (values <= edges[-1])]
            ax.hist(shown, bins=edges, weights=np.full(len(shown), 1 / len(values)), histtype="step", color=color,
                    lw=LINE_WIDTH)
        info = []
        if column.isna().any():
            info.append(f"缺失 {column.isna().mean():.1%}")
        outside = int(((column < edges[0]) | (column > edges[-1])).sum())
        if outside:
            info.append(f"{outside} 个离群值未画出")
        if info:
            ax.text(1.0, 1.0, "\n".join(info), transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                    color=MUTED)
    for ax in list(axes.flat)[len(anchors):]:
        ax.set_visible(False)
    for row in axes:
        row[0].set_ylabel("占该类行数", color=MUTED, fontsize=8)
        row[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    for ax in axes[-1]:
        ax.set_xlabel(feature, color=MUTED, fontsize=8)
        ax.tick_params(axis="x", labelbottom=True)


def draw_correlation_matrix(fig: Figure, ft: FeatureTable, corr: pd.DataFrame, method: str) -> None:
    """所有 (特征, 锚点) 列两两之间的相关系数，按特征分块（块内为同一特征在不同锚点之间的相关性）。"""
    fig.clear()
    cols = list(ft.columns.values())
    ax = fig.subplots()
    mesh = _heatmap(ax, corr.loc[cols, cols].to_numpy(), cols, cols, annotate=len(cols) <= ANNOTATE_MAX)
    # 特征之间留出较宽的空隙，看出分块
    sizes = [sum(1 for f, _ in ft.columns if f == feature) for feature in ft.features]
    for edge in np.cumsum(sizes)[:-1]:
        ax.axhline(edge, color=SURFACE, lw=8)
        ax.axvline(edge, color=SURFACE, lw=8)
    ax.set_title(_title(ft, f"特征相关矩阵（{METHODS[method]}）"), loc="left", fontsize=11, color=INK, pad=8)
    _colorbar(fig, mesh, ax, method)


def draw_pair_correlation(fig: Figure, ft: FeatureTable, corr: pd.DataFrame, method: str) -> None:
    """每两个特征在同一锚点上的相关系数：行 = 特征对，列 = 锚点。"""
    fig.clear()
    pairs = list(combinations(ft.features, 2))
    matrix = np.array([
        [_corr_at(corr, ft.column(fx, a), ft.column(fy, a)) for a in ft.anchors] for fx, fy in pairs
    ])
    ax = fig.subplots()
    mesh = _heatmap(ax, matrix, ft.anchors, [f"{fx} × {fy}" for fx, fy in pairs], annotate=True)
    ax.set_title(_title(ft, f"同一锚点上两个特征的相关系数（{METHODS[method]}）"), loc="left", fontsize=11, color=INK,
                 pad=8)
    _colorbar(fig, mesh, ax, method)


def draw_label_correlation(fig: Figure, ft: FeatureTable, by_anchor: pd.DataFrame, method: str) -> None:
    """各特征与标签（车内 = 1）的相关系数：行 = 特征，列 = 锚点。绝对值越大，单独用它区分车内外越有效。"""
    fig.clear()
    ax = fig.subplots()
    mesh = _heatmap(ax, by_anchor.to_numpy(dtype=float), ft.anchors, ft.features, annotate=True)
    ax.set_title(_title(ft, f"与标签的相关系数（{METHODS[method]}，正值 = 车内时偏大）"), loc="left", fontsize=11,
                 color=INK, pad=8)
    _colorbar(fig, mesh, ax, method)


def draw_scatter(fig: Figure, ft: FeatureTable, fx: str, fy: str, corr: pd.DataFrame, method: str) -> None:
    """两个特征在各锚点上的散点图（每个锚点一张），标题中为该锚点上两者的相关系数。"""
    fig.clear()
    anchors = [a for a in ft.anchors if ft.column(fx, a) and ft.column(fy, a)]
    groups = ft.groups()
    fig.suptitle(_title(ft, f"{fx} × {fy}（每个锚点、每类最多随机取 {SCATTER_MAX_POINTS:,} 个点）"), x=0.01,
                 ha="left", fontsize=12, color=INK)
    nrows, ncols = _grid_shape(len(anchors))
    axes = fig.subplots(nrows, ncols, squeeze=False, sharex=True, sharey=True)
    pooled = {f: np.concatenate([_finite(ft.values[ft.column(f, a)]) for a in anchors]) for f in (fx, fy)}
    limits = {f: _display_range(v) for f, v in pooled.items()}
    rng = np.random.default_rng(0)
    # 行数多的类别先画，少的画在上面不被盖住
    order = sorted(groups, key=lambda g: -g[2].sum())
    for ax, anchor in zip(axes.flat, anchors):
        x, y = ft.values[ft.column(fx, anchor)].to_numpy(), ft.values[ft.column(fy, anchor)].to_numpy()
        r = _corr_at(corr, ft.column(fx, anchor), ft.column(fy, anchor))
        _style(ax, f"{anchor}    r = {r:.2f}" if np.isfinite(r) else anchor)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        for _, color, mask in order:
            rows = np.flatnonzero(mask & np.isfinite(x) & np.isfinite(y))
            if len(rows) > SCATTER_MAX_POINTS:
                rows = rng.choice(rows, SCATTER_MAX_POINTS, replace=False)
            ax.scatter(x[rows], y[rows], s=8, color=color, alpha=0.35, linewidths=0, rasterized=True)
    for ax in list(axes.flat)[len(anchors):]:
        ax.set_visible(False)
    for f, setter in ((fx, "set_xlim"), (fy, "set_ylim")):
        low, high = limits[f]
        if low < high:
            pad = (high - low) * 0.03
            getattr(axes[0][0], setter)(low - pad, high + pad)
    for row in axes:
        row[0].set_ylabel(fy, color=MUTED, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel(fx, color=MUTED, fontsize=8)
        ax.tick_params(axis="x", labelbottom=True)
    if ft.labels is not None:
        handles = [Line2D([], [], color=c, marker="o", lw=0, markersize=6) for _, c, _ in groups]
        fig.legend(handles, [name for name, _, _ in groups], loc="outside upper right", ncols=len(groups),
                   frameon=False, fontsize=8, labelcolor=INK_2)


def save_all(
    ft: FeatureTable,
    out_dir: Path,
    method: str = "pearson",
    scatter: bool = True,
    dpi: int = 120,
    log: Callable[[str], None] | None = None,
) -> list[Path]:
    """画出全部图表并写出统计表，返回写出的文件。"""
    log = log or (lambda _msg: None)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corr = correlation(ft, method)
    written: list[Path] = []

    def save(filename: str, size: tuple[float, float], draw: Callable[[Figure], None]) -> None:
        fig = new_figure(*size, dpi=dpi)
        draw(fig)
        path = out_dir / filename
        fig.savefig(path, facecolor=SURFACE)
        written.append(path)
        log(f"  {filename}")

    nrows, ncols = _grid_shape(len(ft.anchors))
    for feature in ft.features:
        save(f"dist_{_safe(feature)}.png", (max(8.0, 2.8 * ncols), 3.6 + 2.3 * nrows),
             lambda fig, f=feature: draw_distribution(fig, ft, f))
    n = len(ft.columns)
    side = min(24.0, max(6.0, 0.42 * n + 3.0))
    save("corr_matrix.png", (side + 1.2, side), lambda fig: draw_correlation_matrix(fig, ft, corr, method))
    heat_width = max(6.0, 0.75 * len(ft.anchors) + 3.5)
    if len(ft.features) > 1:
        n_pairs = len(ft.features) * (len(ft.features) - 1) // 2
        save("corr_pairs.png", (heat_width + 1.5, 1.6 + 0.45 * n_pairs),
             lambda fig: draw_pair_correlation(fig, ft, corr, method))
    by_label = label_correlation(ft, corr)
    if by_label is not None:
        save("corr_label.png", (heat_width, 1.6 + 0.45 * len(ft.features)),
             lambda fig: draw_label_correlation(fig, ft, by_label, method))
    if scatter and len(ft.features) > 1:
        # 只画在同一锚点上都有数据列的特征对
        pairs = [(fx, fy) for fx, fy in combinations(ft.features, 2)
                 if any(ft.column(fx, a) and ft.column(fy, a) for a in ft.anchors)]
        if len(pairs) > SCATTER_MAX_PAIRS:
            log(f"  特征两两组合共 {len(pairs)} 对，未画散点图；可用 --features 选几个特征")
        for fx, fy in pairs if len(pairs) <= SCATTER_MAX_PAIRS else []:
            save(f"scatter_{_safe(fx)}__{_safe(fy)}.png", (max(8.0, 2.8 * ncols), 0.8 + 2.6 * nrows),
                 lambda fig, a=fx, b=fy: draw_scatter(fig, ft, a, b, corr, method))
    for filename, frame, index in ((STATS_FILENAME, describe(ft, corr), False), (CORR_FILENAME, corr, True)):
        frame.to_csv(out_dir / filename, index=index, encoding="utf-8-sig", float_format="%.6g")
        written.append(out_dir / filename)
        log(f"  {filename}")
    return written


# ---------------------------------------------------------------------------- 命令行
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python plot_features.py",
        description="画出数据集中各锚点特征的分布（车内 / 车外对比）与相关性，保存为 PNG，并输出统计表。",
    )
    p.add_argument("dataset", help=f"数据集：含 {COMBINED_FILENAME} 的文件夹（数据集构建工具的导出文件夹）或 csv 文件")
    p.add_argument("--features", help="要分析的特征，逗号分隔，不带锚点后缀，如 dis_sm,dis_main_sm,rssi_init；默认全部")
    p.add_argument("--anchors", help="只分析这些锚点，逗号分隔，如 FL,F,FR；默认全部")
    p.add_argument("--method", choices=list(METHODS), default="pearson",
                   help="相关系数：pearson（线性）/ spearman（按秩，适合 RSSI 与距离这类单调非线性关系，"
                        "也不易受 65535 这类无效值影响），默认 pearson")
    p.add_argument("-o", "--output", help="输出文件夹，默认为数据集文件夹下的 feature_plots")
    p.add_argument("--label-column", default=LABEL_COLUMN, help=f"标签列名，默认 {LABEL_COLUMN}")
    p.add_argument("--no-scatter", action="store_true", help="不画特征两两之间的散点图")
    p.add_argument("--dpi", type=int, default=120, help="图片分辨率，默认 120")
    p.add_argument("--list", action="store_true", help="列出数据集中的锚点和特征后退出")
    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    try:
        csv_path = resolve_dataset(args.dataset)
        table = load_table(csv_path)
        name = csv_path.parent.name if csv_path.name == COMBINED_FILENAME else csv_path.stem
        ft = load_features(
            table,
            None if args.list else _split_list(args.features),
            None if args.list else _split_list(args.anchors),
            args.label_column,
            name,
        )
    except ValueError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    print(f"数据：{csv_path}，{_row_counts(ft)}")
    print(f"锚点（{len(ft.anchors)}）：{', '.join(ft.anchors)}")
    if args.list:
        print(f"特征（{len(ft.features)}）：")
        width = max(len(f) for f in ft.features)
        for feature in ft.features:
            lacking = [a for a in ft.anchors if ft.column(feature, a) is None]
            text = f"{len(ft.anchors) - len(lacking)} 个锚点" + (f"（缺 {', '.join(lacking)}）" if lacking else "")
            print(f"  {feature:<{width}}  {text}")
        return 0
    print(f"特征（{len(ft.features)}）：{', '.join(ft.features)}")
    for note in ft.notes:
        print(note)

    out_dir = Path(args.output) if args.output else default_output(csv_path)
    print(f"\n输出到 {out_dir}：")
    save_all(ft, out_dir, args.method, scatter=not args.no_scatter, dpi=args.dpi, log=print)

    by_label = label_correlation(ft, correlation(ft, args.method))
    if by_label is not None:
        print(f"\n与标签的相关系数（{METHODS[args.method]}，正值 = 车内时偏大，绝对值越大越能区分车内外）：")
        print(by_label.to_string(float_format=lambda v: f"{v:.2f}", na_rep="—"))
    return 0


# ---------------------------------------------------------------------------- 内部
def _choose(requested: list[str] | None, available: list[str], kind: str) -> list[str]:
    if not requested:
        return available
    unknown = [r for r in requested if r not in available]
    if unknown:
        raise ValueError(f"数据中没有{kind}：{'、'.join(unknown)}。可用的{kind}：{', '.join(available)}")
    return list(dict.fromkeys(requested))


def _split_list(text: str | None) -> list[str] | None:
    return [s.strip() for s in text.split(",") if s.strip()] if text else None


def _shorten(names: list[str], limit: int = 10) -> str:
    return "、".join(names[:limit]) + (" 等" if len(names) > limit else "")


def _row_counts(ft: FeatureTable) -> str:
    text = f"{len(ft.values):,} 行"
    if ft.labels is not None:
        text += "（" + "，".join(f"{name} {int(mask.sum()):,}" for name, _, mask in ft.groups()) + "）"
    return text


def _title(ft: FeatureTable, text: str) -> str:
    return f"{ft.name} · {text}" if ft.name else text


def _safe(name: str) -> str:
    """用作文件名：替换 Windows 文件名中不允许的字符。"""
    return re.sub(r'[\\/:*?"<>|\s]+', "-", name)


def _finite(series: pd.Series) -> np.ndarray:
    values = series.to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _summary(v: pd.Series) -> dict[str, float]:
    q = v.quantile([0.05, 0.5, 0.95]) if v.notna().any() else pd.Series([np.nan] * 3, index=[0.05, 0.5, 0.95])
    return {"mean": v.mean(), "std": v.std(), "min": v.min(), "p5": q[0.05], "median": q[0.5], "p95": q[0.95],
            "max": v.max()}


def _grid(ft: FeatureTable, value: Callable[[str], float]) -> pd.DataFrame:
    """行 = 特征，列 = 锚点的表；锚点没有该特征时为 NaN。"""
    return pd.DataFrame(
        [[value(ft.column(f, a)) if ft.column(f, a) else np.nan for a in ft.anchors] for f in ft.features],
        index=pd.Index(ft.features, name="feature"),
        columns=pd.Index(ft.anchors, name="anchor"),
        dtype=float,
    )


def _corr_at(corr: pd.DataFrame, a: str | None, b: str | None) -> float:
    return float(corr.at[a, b]) if a and b else np.nan


def _grid_shape(n: int) -> tuple[int, int]:
    ncols = max(1, min(GRID_COLS, n))
    return (n + ncols - 1) // ncols or 1, ncols


def _display_range(values: np.ndarray) -> tuple[float, float]:
    """作图范围：一般为最小值到最大值；有极端离群值（如 65535 这样的无效值）时截到 1% / 99% 分位数外延半个跨度。"""
    if not len(values):
        return 0.0, 0.0
    low, high = float(values.min()), float(values.max())
    p1, p99 = np.percentile(values, [1, 99])
    margin = (p99 - p1) * 0.5
    if p99 > p1:
        low, high = max(low, p1 - margin), min(high, p99 + margin)
    return low, high


def _bin_edges(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """直方图分箱。整数值（如 RSSI）按整数对齐，避免出现一格有数据一格没有的锯齿。"""
    if low >= high:
        return np.array([low - 0.5, low + 0.5])
    if len(values) and np.all(np.mod(values, 1) == 0):
        low, high = np.floor(low), np.ceil(high)
        step = max(1.0, np.ceil((high - low + 1) / HIST_BINS))
        return np.arange(low - 0.5, high + 0.5 + step, step)
    return np.linspace(low, high, HIST_BINS + 1)


def _heatmap(ax, matrix: np.ndarray, xlabels: list[str], ylabels: list[str], annotate: bool):
    cmap = LinearSegmentedColormap.from_list("diverging", DIVERGING).with_extremes(bad=SURFACE)
    rows, cols = matrix.shape
    gap = 2 if max(rows, cols) <= 40 else 0.5
    _style(ax, "")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    mesh = ax.pcolormesh(np.ma.masked_invalid(matrix), cmap=cmap, vmin=-1, vmax=1, edgecolors=SURFACE, linewidth=gap)
    size = 8 if max(rows, cols) <= 40 else 6
    rotate = max(map(len, xlabels), default=0) > 4
    ax.set_xticks(np.arange(cols) + 0.5, xlabels, rotation=90 if rotate else 0, fontsize=size)
    ax.set_yticks(np.arange(rows) + 0.5, ylabels, fontsize=size)
    ax.tick_params(labelcolor=INK_2, top=False, labeltop=False)
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    if annotate:
        for i in range(rows):
            for j in range(cols):
                v = matrix[i, j]
                ax.text(j + 0.5, i + 0.5, f"{v:.2f}" if np.isfinite(v) else "—", ha="center", va="center",
                        fontsize=7 if cols > 15 else 8,
                        color=MUTED if not np.isfinite(v) else "#ffffff" if abs(v) >= 0.6 else INK)
    return mesh


def _colorbar(fig: Figure, mesh, ax, method: str) -> None:
    bar = fig.colorbar(mesh, ax=ax, shrink=0.6, aspect=24, pad=0.02)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=MUTED, labelsize=8, length=0)
    bar.set_label(f"{METHODS[method]} 相关系数", color=MUTED, fontsize=8)
