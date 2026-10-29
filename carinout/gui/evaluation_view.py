"""评估结果的展示：指标表始终显示，下方分页显示各场景准确率、各点位准确率、混淆矩阵与 ROC。

训练结果页和“模型评估”页共用。数据都来自 predictions.csv 格式的预测表和 metrics 字典。
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..training.data import CLASS_NAMES, SPLIT_NAMES
from ..training.metrics import METRIC_NAMES
from ..training.runs import per_point_accuracy, per_scene_accuracy, split_scores
from .train_charts import EvaluationChart

WARN_BG = QColor(245, 158, 11, 60)
WARN_ACCURACY = 0.8  # 场景、点位准确率低于该值时标黄


def make_table(headers: list[str]) -> QTableWidget:
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    table.verticalHeader().setDefaultSectionSize(table.fontMetrics().height() + 10)  # 行高紧凑一些
    return table


class EvaluationView(QWidget):
    def __init__(self, placeholder: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._placeholder = placeholder
        self._path: Path | None = None

        self.header = QLabel(placeholder)
        self.header.setWordWrap(True)
        self.header.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.open_btn = QPushButton("打开结果文件夹")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._open_folder)
        head = QHBoxLayout()
        head.addWidget(self.header, 1)
        head.addWidget(self.open_btn)

        self.summary_table = make_table(["数据集", "样本数", "交叉熵", *METRIC_NAMES.values()])
        self._fit_summary_height(3)
        self.chart = EvaluationChart()
        self.scene_table = make_table(["数据集", "场景", "真实类别", "点位数", "样本数", "准确率", "平均车内概率"])
        self.point_table = make_table(
            ["数据集", "来源", "场景", "点位", "真实类别", "样本数", "准确率", "平均车内概率"]
        )

        self.details = QTabWidget()
        scene_hint = "验证集、测试集中每个场景的准确率，各集合内从低到高；低于 80% 标黄"
        self.details.addTab(_titled(self.scene_table, scene_hint), "各场景准确率")
        self.details.addTab(
            _titled(self.point_table, "每个点位的准确率，从低到高 —— 找出容易判错的位置"), "各点位准确率"
        )
        self.details.addTab(self.chart, "混淆矩阵与 ROC")

        lay = QVBoxLayout(self)
        lay.addLayout(head)
        lay.addWidget(self.summary_table)
        lay.addWidget(self.details, 1)

    def clear(self, text: str | None = None) -> None:
        self._path = None
        self.open_btn.setEnabled(False)
        self.header.setText(text or self._placeholder)
        for table in (self.summary_table, self.scene_table, self.point_table):
            table.setRowCount(0)
        self._fit_summary_height(3)
        self.chart.update_scores({})

    def show_results(self, path: Path, header: str, metrics: dict, predictions: pd.DataFrame) -> None:
        """metrics 中用到 splits（各集合指标）和 sources（按文件夹的指标）；表格和图表由 predictions 计算。"""
        self.clear(header)
        self._path = Path(path)
        self.open_btn.setEnabled(True)
        sources = metrics.get("sources", {})
        for name, values in metrics.get("splits", {}).items():
            self._add_summary_row(SPLIT_NAMES.get(name, name), values)
            for source, source_values in sources.get(name, {}).items():  # 来自多个文件夹时逐个列出
                self._add_summary_row(f"    └ {source}", source_values)
        self._fit_summary_height(self.summary_table.rowCount())
        self.chart.update_scores(split_scores(predictions), metrics.get("threshold", 0.5))

        for rec in _evaluation_rows(per_scene_accuracy(predictions)):
            cells = [
                SPLIT_NAMES.get(rec.split, rec.split),
                rec.scene,
                CLASS_NAMES.get(rec.label, rec.label),
                getattr(rec, "points", ""),
                f"{rec.samples:,}",
                f"{rec.accuracy:.1%}",
                f"{rec.mean_prob:.3f}",
            ]
            _add_row(self.scene_table, cells, rec.accuracy < WARN_ACCURACY)
        for rec in _evaluation_rows(per_point_accuracy(predictions)):
            cells = [
                SPLIT_NAMES.get(rec.split, rec.split),
                getattr(rec, "source", ""),
                rec.scene,
                rec.point,
                CLASS_NAMES.get(rec.label, rec.label),
                f"{rec.samples:,}",
                f"{rec.accuracy:.1%}",
                f"{rec.mean_prob:.3f}",
            ]
            _add_row(self.point_table, cells, rec.accuracy < WARN_ACCURACY)

    def _add_summary_row(self, label: str, values: dict) -> None:
        cells = [label, f"{values.get('n', 0):,}", _fmt(values.get("loss"), "{:.4f}")]
        cells += [_fmt(values.get(k), "{:.4f}" if k == "auc" else "{:.2%}") for k in METRIC_NAMES]
        _add_row(self.summary_table, cells, False)

    def _fit_summary_height(self, rows: int) -> None:
        """指标表按行数调整高度，最多显示 8 行，再多时滚动。"""
        visible = min(max(rows, 3), 8)
        height = self.summary_table.verticalHeader().defaultSectionSize() * visible
        self.summary_table.setFixedHeight(self.summary_table.horizontalHeader().sizeHint().height() + height + 4)
        policy = Qt.ScrollBarPolicy.ScrollBarAsNeeded if rows > 8 else Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        self.summary_table.setVerticalScrollBarPolicy(policy)

    def _open_folder(self) -> None:
        if self._path is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._path)))


def _evaluation_rows(table: pd.DataFrame):
    """只列验证集、测试集（训练集通常接近 100%）；都没有时列出全部。"""
    if table.empty:
        return []
    shown = table[table["split"] != "train"]
    return (shown if not shown.empty else table).itertuples(index=False)


def _titled(table: QTableWidget, title: str) -> QWidget:
    box = QWidget()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    label = QLabel(title)
    label.setStyleSheet("color: gray")
    lay.addWidget(label)
    lay.addWidget(table)
    return box


def _add_row(table: QTableWidget, cells: list, warn: bool) -> None:
    row = table.rowCount()
    table.insertRow(row)
    for col, text in enumerate(cells):
        item = QTableWidgetItem(str(text))
        if warn:
            item.setBackground(WARN_BG)
        table.setItem(row, col, item)


def _fmt(value, pattern: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return pattern.format(value)
