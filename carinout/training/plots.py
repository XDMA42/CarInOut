"""训练过程与评估结果的图表（matplotlib）。界面中的实时图表和训练结果文件夹里的 PNG 用同一套绘图函数。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import font_manager, rcParams
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from .data import CLASS_NAMES, SPLIT_NAMES
from .metrics import METRIC_NAMES, binary_metrics, roc_curve

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# 分类色按固定顺序使用（已校验色觉障碍下相邻色可区分）
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SPLIT_COLORS = {"train": SERIES[0], "val": SERIES[1], "test": SERIES[2]}
# 单色顺序色阶（浅 → 深），用于混淆矩阵
SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
LINE_WIDTH = 1.6

CJK_FONTS = (
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Hiragino Sans GB",
    "Heiti SC",
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "Source Han Sans SC",
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
)


def configure_fonts() -> None:
    """选用系统中已安装的中文字体，否则图表中的中文会显示为方框。"""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = [f for f in CJK_FONTS if f in installed] + ["DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False


configure_fonts()


@dataclass
class EpochSeries:
    label: str
    color: str
    x: np.ndarray
    y: np.ndarray


@dataclass
class EpochPanel:
    """一个以 epoch 为横轴的子图及其中的曲线，界面用来实现悬停读数。"""

    ax: object
    series: list[EpochSeries] = field(default_factory=list)


def new_figure(width: float, height: float, dpi: int = 100) -> Figure:
    return Figure(figsize=(width, height), dpi=dpi, facecolor=SURFACE, layout="constrained")


def draw_losses(fig: Figure, history: pd.DataFrame | list[dict], best_epoch: int = 0) -> list[EpochPanel]:
    """三个子图：总损失、交叉熵、L2 正则项。训练集与验证集同一子图对比。"""
    history = pd.DataFrame(history)
    fig.clear()
    axes = fig.subplots(1, 3)

    def pair(prefix: str) -> list[tuple[str, str, str]]:
        return [(f"{split}_{prefix}", SPLIT_NAMES[split], SPLIT_COLORS[split]) for split in ("train", "val")]

    return [
        _epoch_panel(axes[0], "总损失（交叉熵 + L2）", history, pair("loss"), best_epoch),
        _epoch_panel(axes[1], "交叉熵", history, pair("ce"), best_epoch),
        _epoch_panel(axes[2], "L2 正则项  λ·Σ‖W‖²", history, [("l2", "L2 项", SERIES[0])], best_epoch),
    ]


def draw_metrics(fig: Figure, history: pd.DataFrame | list[dict], best_epoch: int = 0) -> list[EpochPanel]:
    """验证集上各评价指标随 epoch 的变化（阈值 0.5，正类 = 车内）。"""
    history = pd.DataFrame(history)
    fig.clear()
    ax = fig.subplots()
    series = [(f"val_{key}", name, SERIES[i]) for i, (key, name) in enumerate(METRIC_NAMES.items())]
    title = "验证集评价指标（正类 = 车内，阈值 0.5）"
    panel = _epoch_panel(ax, title, history, series, best_epoch, legend_loc="lower right", tag_at_top=False)
    if panel.series:
        low = min(np.nanmin(s.y) for s in panel.series)
        ax.set_ylim(max(0.0, low - max(0.02, (1 - low) * 0.15)), 1.005)
    return [panel]


def draw_evaluation(fig: Figure, scores: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    """验证集、测试集的混淆矩阵与 ROC 曲线。scores：split → (真实标签, 车内概率)。"""
    fig.clear()
    axes = fig.subplots(1, 3, gridspec_kw={"width_ratios": [1, 1, 1.15]})
    for ax, name in zip(axes[:2], ("val", "test")):
        if name in scores:
            _confusion(ax, name, *scores[name])
        else:
            _style(ax, f"{SPLIT_NAMES[name]}混淆矩阵")
            _placeholder(ax, f"没有{SPLIT_NAMES[name]}")
            ax.set_xticks([])
            ax.set_yticks([])
    _roc(axes[2], scores)


def draw_sweep(
    fig: Figure,
    results: pd.DataFrame,
    axes: list[tuple[str, str]],
    metric: str,
    metric_label: str,
    lower_is_better: bool = False,
) -> None:
    """参数遍历结果：每个有多个取值的参数一张子图，横轴为取值，点为各组合的指标，折线为同一取值下的平均。"""
    fig.clear()
    shown = [(key, label) for key, label in axes if key in results and results[key].nunique() > 1]
    finished = results[results[metric].notna()] if metric in results else pd.DataFrame()
    if not shown or finished.empty:
        ax = fig.subplots()
        _style(ax, f"各参数对{metric_label}的影响")
        _placeholder(ax, "等待结果…" if shown else "填写多个取值的参数会在这里对比")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    cols = len(shown) if len(shown) <= 4 else 3
    grid = fig.subplots((len(shown) + cols - 1) // cols, cols, sharey=True, squeeze=False)
    rng = np.random.default_rng(0)
    jitter = pd.Series((rng.random(len(finished)) - 0.5) * 0.18, index=finished.index)  # 同一取值的点左右错开
    ranked = finished.sort_values(metric, ascending=lower_is_better, kind="stable")
    best = ranked.index[0]
    for ax, (key, label) in zip(grid.flat, shown):
        values = sorted(results[key].dropna().unique())
        position = {v: i for i, v in enumerate(values)}
        x = finished[key].map(position) + jitter
        _style(ax, label)
        ax.scatter(x, finished[metric], s=36, color=SERIES[0], edgecolors=SURFACE, linewidths=1.5, zorder=3,
                   label="各组合")
        means = finished.groupby(key)[metric].mean().reindex(values)
        ax.plot(range(len(values)), means.to_numpy(), color=INK_2, lw=1.2, marker="o", markersize=4, zorder=2,
                label="平均")
        ax.scatter([x[best]], [finished.at[best, metric]], s=80, color=SERIES[1], edgecolors=SURFACE, linewidths=2,
                   zorder=4, label="最佳组合")
        ax.set_xticks(range(len(values)), [f"{v:g}" if isinstance(v, float) else str(v) for v in values])
        ax.set_xlim(-0.5, len(values) - 0.5)
    for ax in list(grid.flat)[len(shown):]:
        ax.set_visible(False)
    for row in grid:
        row[0].set_ylabel(metric_label + ("（越低越好）" if lower_is_better else ""), color=MUTED, fontsize=8)
    handles, labels = grid[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper right", ncols=3, frameon=False, fontsize=8, labelcolor=INK_2,
               handlelength=1.2)


def save_figures(run_dir: Path, history: pd.DataFrame, best_epoch: int, scores: dict) -> None:
    for filename, size, draw in (
        ("loss.png", (13, 3.8), lambda f: draw_losses(f, history, best_epoch)),
        ("metrics.png", (8, 4.2), lambda f: draw_metrics(f, history, best_epoch)),
        ("evaluation.png", (13, 4.2), lambda f: draw_evaluation(f, scores)),
    ):
        fig = new_figure(*size, dpi=120)
        draw(fig)
        fig.savefig(Path(run_dir) / filename, facecolor=SURFACE)


# ---------------------------------------------------------------------------- 内部
def _style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=0, pad=4)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _placeholder(ax, text: str) -> None:
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=9)


def _epoch_panel(
    ax, title, history: pd.DataFrame, columns, best_epoch: int, legend_loc="upper right", tag_at_top=True
) -> EpochPanel:
    _style(ax, title)
    ax.set_xlabel("epoch", color=MUTED, fontsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    panel = EpochPanel(ax)
    for column, label, color in columns:
        if column not in history or not history[column].notna().any():
            continue
        x = history["epoch"].to_numpy(dtype=float)
        y = history[column].to_numpy(dtype=float)
        ax.plot(x, y, color=color, lw=LINE_WIDTH, label=label, solid_capstyle="round", solid_joinstyle="round",
                marker="o" if len(x) == 1 else None, markersize=4)
        panel.series.append(EpochSeries(label, color, x, y))
    if not panel.series:
        _placeholder(ax, "等待训练…" if history.empty else "没有验证集")
        return panel
    if best_epoch:
        ax.axvline(best_epoch, color=AXIS, lw=0.8, zorder=1)
        # 损失曲线下降，标注放在顶部；指标曲线上升，标注放在底部，避免压住曲线
        ax.annotate(f"最佳 epoch {best_epoch}", xy=(best_epoch, 1 if tag_at_top else 0),
                    xycoords=("data", "axes fraction"), xytext=(3, -2 if tag_at_top else 2),
                    textcoords="offset points", fontsize=8, color=MUTED, va="top" if tag_at_top else "bottom")
    if len(panel.series) > 1:
        # 图例与“最佳 epoch”标注在同一侧时让开一行
        anchor_y = (0.93 if tag_at_top else 1.0) if "upper" in legend_loc else (0.0 if tag_at_top else 0.07)
        ax.legend(loc=legend_loc, bbox_to_anchor=(1.0, anchor_y), frameon=False, fontsize=8, labelcolor=INK_2,
                  handlelength=1.4)
    return panel


def _confusion(ax, name: str, labels: np.ndarray, probs: np.ndarray) -> None:
    m = binary_metrics(labels, probs)
    cm = np.asarray(m["confusion"], dtype=float)
    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)  # 按真实类别（行）归一化
    cmap = LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
    _style(ax, f"{SPLIT_NAMES[name]}混淆矩阵 · 准确率 {m['accuracy']:.1%}")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # 单元格之间留出背景色的空隙，不画边框
    ax.pcolormesh(np.arange(3), np.arange(3), share, cmap=cmap, vmin=0, vmax=1, edgecolors=SURFACE, linewidth=3)
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.5, f"{int(cm[i, j]):,}\n{share[i, j]:.1%}", ha="center", va="center",
                    fontsize=10, color="#ffffff" if share[i, j] >= 0.5 else INK)
    ticks = [CLASS_NAMES[c] for c in sorted(CLASS_NAMES)]
    ax.set_xticks([0.5, 1.5], ticks)
    ax.set_yticks([0.5, 1.5], ticks)
    ax.set_xlabel("预测", color=MUTED, fontsize=8)
    ax.set_ylabel("真实", color=MUTED, fontsize=8)
    ax.tick_params(labelsize=9, labelcolor=INK_2)
    ax.set_xlim(0, 2)
    ax.set_ylim(2, 0)
    ax.set_aspect("equal")


def _roc(ax, scores: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    _style(ax, "ROC 曲线")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.plot([0, 1], [0, 1], color=AXIS, lw=0.8)
    shown = [n for n in ("val", "test") if n in scores] or [n for n in ("train",) if n in scores]
    drawn = False
    for name in shown:
        labels, probs = scores[name]
        fpr, tpr = roc_curve(labels, probs)
        if np.isnan(fpr).any() or np.isnan(tpr).any():
            continue
        auc = binary_metrics(labels, probs)["auc"]
        ax.plot(fpr, tpr, color=SPLIT_COLORS[name], lw=LINE_WIDTH, label=f"{SPLIT_NAMES[name]}  AUC {auc:.3f}",
                solid_joinstyle="round")
        drawn = True
    if drawn:
        ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK_2, handlelength=1.4)
    else:
        _placeholder(ax, "需要同时包含车内、车外样本" if shown else "等待训练…")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("假正率（车外判为车内）", color=MUTED, fontsize=8)
    ax.set_ylabel("真正率（车内判为车内）", color=MUTED, fontsize=8)
    ax.set_aspect("equal")
