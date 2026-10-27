"""训练界面中的 matplotlib 图表：以 epoch 为横轴的曲线（鼠标悬停显示数值）和评估结果图。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ..training import plots


class _ChartBase(QWidget):
    def __init__(self, width: float, height: float, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = plots.new_figure(width, height)
        self.canvas = FigureCanvasQTAgg(self.figure)
        toolbar = NavigationToolbar2QT(self.canvas, self)  # 缩放、平移、保存图片
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(toolbar)
        lay.addWidget(self.canvas, 1)


class EpochChart(_ChartBase):
    """draw 为 plots.draw_losses / draw_metrics。鼠标移到子图上时，各子图显示同一 epoch 的竖线，
    当前子图显示各曲线在该 epoch 的数值。"""

    def __init__(self, draw: Callable, width: float = 10, height: float = 4, parent: QWidget | None = None) -> None:
        super().__init__(width, height, parent)
        self._draw = draw
        self._panels: list[plots.EpochPanel] = []
        self._cursor: list = []  # 每个子图的 (竖线, 标记点列表, 文本框)
        self.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.canvas.mpl_connect("axes_leave_event", lambda _e: self._hide_cursor())
        self.update_history([], 0)

    def update_history(self, history, best_epoch: int) -> None:
        self._panels = self._draw(self.figure, history, best_epoch)
        self._cursor = []
        for panel in self._panels:
            ax = panel.ax
            line = ax.axvline(0, color=plots.MUTED, lw=0.8, visible=False, zorder=4)
            dots = [
                ax.plot([], [], "o", color=s.color, markersize=7, markeredgecolor=plots.SURFACE,
                        markeredgewidth=1.5, visible=False, zorder=5)[0]
                for s in panel.series
            ]
            label = ax.annotate(
                "", xy=(0, 1), xycoords=("data", "axes fraction"), xytext=(8, -44), textcoords="offset points",
                va="top", fontsize=8, color=plots.INK, visible=False, zorder=6,
                bbox={"boxstyle": "round,pad=0.4", "fc": plots.SURFACE, "ec": plots.AXIS, "lw": 0.8},
            )
            self._cursor.append((line, dots, label))
        self.canvas.draw_idle()

    def _on_move(self, event) -> None:
        index = next((i for i, p in enumerate(self._panels) if p.ax is event.inaxes), None)
        if index is None or not self._panels[index].series or event.xdata is None:
            self._hide_cursor()
            return
        epochs = self._panels[index].series[0].x
        i = int(np.argmin(np.abs(epochs - event.xdata)))
        epoch = epochs[i]
        for k, (panel, (line, dots, label)) in enumerate(zip(self._panels, self._cursor)):
            if not panel.series:
                continue
            line.set_xdata([epoch, epoch])
            line.set_visible(True)
            for dot, s in zip(dots, panel.series):
                visible = i < len(s.y) and np.isfinite(s.y[i])
                dot.set_data([epoch], [s.y[i]] if visible else [np.nan])
                dot.set_visible(visible)
            if k == index:
                rows = [f"epoch {int(epoch)}"] + [f"{s.label}  {s.y[i]:.4g}" for s in panel.series if i < len(s.y)]
                label.set_text("\n".join(rows))
                label.xy = (epoch, 1)
                right_half = epoch > (epochs[0] + epochs[-1]) / 2
                label.set_position((-8, -44) if right_half else (8, -44))  # 避开右上角图例
                label.set_ha("right" if right_half else "left")
                label.set_visible(True)
            else:
                label.set_visible(False)
        self.canvas.draw_idle()

    def _hide_cursor(self) -> None:
        changed = False
        for line, dots, label in self._cursor:
            for artist in (line, label, *dots):
                changed |= artist.get_visible()
                artist.set_visible(False)
        if changed:
            self.canvas.draw_idle()


class EvaluationChart(_ChartBase):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(12, 4, parent)
        self.update_scores({})

    def update_scores(self, scores: dict) -> None:
        plots.draw_evaluation(self.figure, scores)
        self.canvas.draw_idle()
