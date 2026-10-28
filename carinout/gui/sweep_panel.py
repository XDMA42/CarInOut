"""参数遍历页：填写要遍历的参数取值，多进程并行训练所有组合，用表格和图表比较。

其余设置（数据集、特征、未遍历的参数）都取自左侧的训练配置。
"""

from __future__ import annotations

import math
import numbers
import os
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from PySide6.QtCore import QObject, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..training.config import ExperimentConfig
from ..training.sweep import (
    FINAL_STATUSES,
    LOWER_IS_BETTER,
    RANK_METRICS,
    STATUS_PENDING,
    STATUS_RUNNING,
    SweepAxis,
    SweepRecord,
    Trial,
    available_axes,
    best_trial,
    expand_grid,
    load_sweep,
    run_sweep,
)
from .train_charts import SweepChart

ROLE_TRIAL = Qt.ItemDataRole.UserRole
BEST_BG = QColor(42, 120, 214, 40)
CONFIRM_TRIALS = 200  # 组合数超过这个值时先确认

# 表格中的结果列：(结果列, 表头, 格式)
RESULT_COLUMNS = (
    ("status", "状态", None),
    ("best_epoch", "最佳 epoch", "{:.0f}"),
    ("val_ce", "验证交叉熵", "{:.4f}"),
    ("val_accuracy", "验证准确率", "{:.2%}"),
    ("val_precision", "验证精确率", "{:.2%}"),
    ("val_recall", "验证召回率", "{:.2%}"),
    ("val_f1", "验证 F1", "{:.2%}"),
    ("val_auc", "验证 AUC", "{:.4f}"),
    ("test_accuracy", "测试准确率", "{:.2%}"),
    ("test_auc", "测试 AUC", "{:.4f}"),
    ("seconds", "耗时 (s)", "{:.1f}"),
)


class SweepWorker(QObject):
    event = Signal(object)
    message = Signal(str)
    done = Signal(object)  # 遍历结果文件夹
    failed = Signal(str)

    def __init__(self, base: ExperimentConfig, trials: list[Trial], workers: int, rank_by: str) -> None:
        super().__init__()
        self._args = (base, trials, workers, rank_by)
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            path = run_sweep(*self._args, self.event.emit, lambda: self._stop, self.message.emit)
        except Exception as exc:  # noqa: BLE001 —— 任何异常都要回报给界面，避免线程静默退出
            self.failed.emit(str(exc))
        else:
            self.done.emit(path)


class _Item(QTableWidgetItem):
    """按数值排序的表格单元；没有数值的排在最后。"""

    def __init__(self, text: str, key=None) -> None:
        super().__init__(text)
        self.key = key

    def __lt__(self, other: QTableWidgetItem) -> bool:
        a, b = self.key, getattr(other, "key", None)
        if isinstance(a, numbers.Real) and isinstance(b, numbers.Real):
            if math.isnan(b):
                return not math.isnan(a)
            return not math.isnan(a) and a < b
        return self.text() < other.text()


class SweepPanel(QWidget):
    runningChanged = Signal(bool)
    showRunRequested = Signal(object)  # 某个组合的训练结果文件夹
    applyConfigRequested = Signal(object)  # 某个组合的 ExperimentConfig
    message = Signal(str)

    def __init__(self, base_config: Callable[[], ExperimentConfig | None], parent: QWidget | None = None) -> None:
        """base_config 返回左侧的训练配置；配置有误时由它提示用户并返回 None。"""
        super().__init__(parent)
        self._base_config = base_config
        self._axes: list[SweepAxis] = []
        self._edits: dict[str, QLineEdit] = {}
        self._texts: dict[str, str] = {}  # 各参数填写的取值，切换模型后同名参数保留
        self._keys: list[str] = []  # 当前表格中遍历的参数
        self._labels: dict[str, str] = {}
        self._rows: dict[int, dict] = {}  # 组合序号 → 结果行
        self._epochs: dict[int, int] = {}  # 组合序号 → epoch 数
        self._sweep_dir: Path | None = None
        self._thread: QThread | None = None
        self._worker: SweepWorker | None = None
        self._busy = False  # 左侧单次训练进行中
        self._chart_timer = QTimer(self)
        self._chart_timer.setSingleShot(True)
        self._chart_timer.setInterval(500)
        self._chart_timer.timeout.connect(self._redraw_chart)

        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self._build_settings())
        split.addWidget(self._build_results())
        self.chart = SweepChart()
        split.addWidget(self.chart)
        split.setSizes([170, 330, 300])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split)

    # ------------------------------------------------------------------ 界面搭建
    def _build_settings(self) -> QWidget:
        box = QGroupBox("遍历设置")
        self.axes_grid = QGridLayout()  # 两列：标签、输入框、标签、输入框
        self.axes_grid.setColumnStretch(1, 1)
        self.axes_grid.setColumnStretch(3, 1)
        hint = QLabel(
            "每项填一个或多个取值，用逗号分隔；留空则使用左侧的设置。各参数的取值做全组合，每组训练一个模型。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray")
        self.count_label = QLabel()
        self.count_label.setWordWrap(True)
        left = QVBoxLayout()
        left.addWidget(hint)
        left.addLayout(self.axes_grid)
        left.addWidget(self.count_label)
        left.addStretch(1)

        cpus = os.cpu_count() or 1
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, cpus)
        self.workers_spin.setValue(cpus)
        self.workers_spin.setSuffix(f" / {cpus} 个 CPU 核")
        self.workers_spin.setToolTip(
            "同时训练的组合数，每个组合在独立的进程中训练（不受 Python GIL 限制，能用满多个 CPU 核）。\n"
            "每个进程约占几百 MB 内存，内存紧张时调小。"
        )
        self.rank_combo = QComboBox()
        for key, label in RANK_METRICS.items():
            self.rank_combo.addItem(label, key)
        self.rank_combo.setToolTip("用哪个验证集指标挑选最佳组合，也是表格默认的排序依据")
        self.rank_combo.currentIndexChanged.connect(self._on_rank_changed)
        self.start_btn = QPushButton("开始遍历")
        self.start_btn.setMinimumHeight(34)
        self.start_btn.clicked.connect(self._start_or_stop)
        self.progress = QProgressBar()
        self.progress.setFormat("%v / %m 组")
        self.progress.setValue(0)
        form = QFormLayout()
        form.addRow("并行进程数", self.workers_spin)
        form.addRow("最佳组合依据", self.rank_combo)
        right = QVBoxLayout()
        right.addLayout(form)
        right.addStretch(1)
        right.addWidget(self.start_btn)
        right.addWidget(self.progress)

        lay = QHBoxLayout(box)
        lay.addLayout(left, 3)
        lay.addLayout(right, 2)
        return box

    def _build_results(self) -> QWidget:
        self.table = QTableWidget(0, 0)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.cellDoubleClicked.connect(lambda *_: self._show_selected())
        self.result_label = QLabel("开始遍历后，每个组合一行；点击表头排序，双击一行查看它的训练曲线和评估结果")
        self.result_label.setWordWrap(True)

        show = QPushButton("查看训练曲线")
        show.clicked.connect(self._show_selected)
        apply = QPushButton("把选中组合载入左侧")
        apply.setToolTip("用这组参数覆盖左侧的配置，可以在它的基础上继续调整或单独训练")
        apply.clicked.connect(self._apply_selected)
        self.open_dir_btn = QPushButton("打开结果文件夹")
        self.open_dir_btn.setEnabled(False)
        self.open_dir_btn.clicked.connect(self._open_folder)
        open_sweep = QPushButton("打开遍历结果…")
        open_sweep.clicked.connect(self._open_sweep_dialog)
        brow = QHBoxLayout()
        brow.addWidget(self.result_label, 1)
        brow.addWidget(show)
        brow.addWidget(apply)
        brow.addWidget(self.open_dir_btn)
        brow.addWidget(open_sweep)

        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(brow)
        lay.addWidget(self.table, 1)
        return page

    # ------------------------------------------------------------------ 遍历参数
    def refresh_axes(self, base: ExperimentConfig) -> None:
        """按左侧配置（模型、隐藏层数）更新可遍历的参数；已填写的取值按参数保留。"""
        try:
            axes = available_axes(base)
        except ValueError:
            return
        if [a.key for a in axes] != [a.key for a in self._axes]:
            self._save_texts()
            while self.axes_grid.count():
                self.axes_grid.takeAt(0).widget().deleteLater()
            self._edits = {}
            for i, axis in enumerate(axes):
                edit = QLineEdit(self._texts.get(axis.key, ""))
                edit.textChanged.connect(self._update_count)
                row, col = divmod(i, 2)
                self.axes_grid.addWidget(QLabel(axis.label), row, col * 2)
                self.axes_grid.addWidget(edit, row, col * 2 + 1)
                self._edits[axis.key] = edit
        self._axes = axes
        for axis in axes:
            base_text = axis.base if not isinstance(axis.base, float) else f"{axis.base:g}"
            self._edits[axis.key].setPlaceholderText(f"左侧当前为 {base_text}，如填 “{axis.example}”")
        self._update_count()

    def _save_texts(self) -> None:
        self._texts.update({key: edit.text() for key, edit in self._edits.items()})

    def _parse_axes(self) -> dict[str, list]:
        return {a.key: a.parse(self._edits[a.key].text()) for a in self._axes if self._edits[a.key].text().strip()}

    def _update_count(self) -> None:
        try:
            axes = self._parse_axes()
        except ValueError as exc:
            self.count_label.setStyleSheet("color: #d03b3b")
            self.count_label.setText(str(exc))
            return
        self.count_label.setStyleSheet("")
        sizes = [len(v) for v in axes.values()]
        total = math.prod(sizes)
        if not sizes:
            self.count_label.setText("尚未填写取值：只有左侧配置这 1 组")
        else:
            formula = " × ".join(map(str, sizes))
            self.count_label.setText(f"共 {formula} = {total} 组" if len(sizes) > 1 else f"共 {total} 组")

    # ------------------------------------------------------------------ 运行
    def set_busy(self, busy: bool) -> None:
        """左侧单次训练进行中时不能同时开始遍历。"""
        self._busy = busy
        if self._worker is None:
            self.start_btn.setEnabled(not busy)

    def _start_or_stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self.start_btn.setEnabled(False)
            self.start_btn.setText("正在停止…")
            return
        self._start()

    def _start(self) -> None:
        base = self._base_config()
        if base is None:
            return
        self.refresh_axes(base)
        try:
            trials = expand_grid(base, self._parse_axes())
        except ValueError as exc:
            QMessageBox.warning(self, "遍历参数有误", str(exc))
            return
        questions = []
        if len(trials) == 1:
            questions.append("没有填写多个取值，只会训练 1 组。")
        if len(trials) > CONFIRM_TRIALS:
            questions.append(f"共 {len(trials)} 组，可能需要很长时间。")
        if base.data.val_ratio == 0 and not base.data.val_datasets:
            questions.append("验证集比例为 0 且没有独立验证集，无法按验证集指标比较各组合。")
        if questions:
            ok = QMessageBox.question(self, "开始遍历", "\n".join(questions) + "\n是否继续？")
            if ok != QMessageBox.StandardButton.Yes:
                return

        self._save_texts()
        self._sweep_dir = None
        self.open_dir_btn.setEnabled(False)
        known = {a.key: a.label for a in self._axes}
        self._setup_table(list(trials[0].overrides), known)
        self._epochs = {t.index: t.config.train.epochs for t in trials}
        self._rows = {
            t.index: {"trial": t.index, "name": t.name, **t.overrides, "status": STATUS_PENDING} for t in trials
        }
        for row in self._rows.values():
            self._append_row(row)
        self.table.resizeColumnsToContents()
        self.progress.setRange(0, len(trials))
        self.progress.setValue(0)
        self.start_btn.setText("停止遍历")
        self.result_label.setText(f"共 {len(trials)} 组，正在准备数据并启动 {self.workers_spin.value()} 个训练进程…")
        self._redraw_chart()

        self._thread = QThread(self)
        self._worker = SweepWorker(base, trials, self.workers_spin.value(), self.rank_combo.currentData())
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.event.connect(self._on_event)
        self._worker.message.connect(self.message)
        self._worker.done.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()
        self.runningChanged.emit(True)

    def _on_event(self, event: tuple) -> None:
        kind = event[0]
        if kind == "ready":
            self._sweep_dir = event[1]
            self.open_dir_btn.setEnabled(True)
            self.result_label.setText(f"结果保存在 {self._sweep_dir}")
            return
        index = event[1]
        row = self._rows.get(index)
        if row is None or row["status"] in FINAL_STATUSES:  # 进程中晚到的 epoch 消息
            return
        if kind == "start":
            self._set_status(index, f"{STATUS_RUNNING} 0/{self._epochs[index]}")
        elif kind == "epoch":
            self._set_status(index, f"{STATUS_RUNNING} {event[2]['epoch']}/{self._epochs[index]}")
        elif kind == "done":
            row.update(event[2])
            self._fill_row(self._row_of(index), row)
            self.table.resizeColumnsToContents()
            self.progress.setValue(sum(r["status"] in FINAL_STATUSES for r in self._rows.values()))
            self._highlight_best()
            if not self._chart_timer.isActive():
                self._chart_timer.start()

    def _cleanup_worker(self) -> None:
        if self._thread:
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None
        self.start_btn.setText("开始遍历")
        self.start_btn.setEnabled(not self._busy)
        self.runningChanged.emit(False)

    def _on_finished(self, path: Path) -> None:
        self._cleanup_worker()
        try:
            self.show_sweep(load_sweep(path), restore_inputs=False)
        except (OSError, ValueError) as exc:
            self.message.emit(f"读取遍历结果失败：{exc}")

    def _on_failed(self, error: str) -> None:
        self._cleanup_worker()
        self.message.emit(f"参数遍历失败：{error}")
        QMessageBox.critical(self, "参数遍历失败", error)

    def shutdown(self) -> None:
        """关闭窗口时调用：停止遍历并等待各进程保存当前结果。"""
        if self._worker is not None:
            self._worker.stop()
            self._cleanup_worker()

    # ------------------------------------------------------------------ 结果展示
    def show_sweep(self, record: SweepRecord, restore_inputs: bool = True) -> None:
        """显示一次遍历的结果。restore_inputs 时把它的遍历取值填回输入框（左侧配置应已载入 record.base）。"""
        self._sweep_dir = record.path
        self.open_dir_btn.setEnabled(True)
        labels = {a.key: a.label for a in available_axes(record.base)}
        if restore_inputs:
            self.refresh_axes(record.base)
            for key, values in record.axes.items():
                if key in self._edits:
                    self._edits[key].setText(", ".join(f"{v:g}" if isinstance(v, float) else str(v) for v in values))
            index = self.rank_combo.findData(record.rank_by)
            if index >= 0:
                self.rank_combo.setCurrentIndex(index)
        self._setup_table(list(record.axes), labels)
        results = record.results
        self._rows = {int(r["trial"]): r for r in results.to_dict("records")} if not results.empty else {}
        self._epochs = {}
        for row in self._rows.values():
            self._append_row(row)
        self.table.resizeColumnsToContents()
        finished = sum(r["status"] in FINAL_STATUSES for r in self._rows.values())
        self.progress.setRange(0, max(len(self._rows), 1))
        self.progress.setValue(finished)
        self.table.setSortingEnabled(True)
        self._sort_by_rank()
        self._highlight_best()
        self._redraw_chart()
        best = best_trial(self._results_frame(), self.rank_combo.currentData())
        text = f"{record.path}\n完成 {finished} / {len(self._rows)} 组"
        if best is not None:
            text += f"；按{self.rank_combo.currentText()}最佳的是 #{int(best['trial'])}：{best['name']}"
        self.result_label.setText(text)

    def _setup_table(self, keys: list[str], labels: dict[str, str]) -> None:
        self._keys = keys
        self._labels = {k: labels.get(k, k) for k in keys}
        self.table.setSortingEnabled(False)
        self.table.clear()
        self.table.setRowCount(0)
        headers = ["#", *(self._labels[k] for k in keys), *(h for _, h, _ in RESULT_COLUMNS)]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)

    def _append_row(self, row: dict) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        self._fill_row(r, row)

    def _fill_row(self, r: int, row: dict) -> None:
        """单元格的排序键：数值列按数值，其余按文本。"""
        cells = [(str(row["trial"]), int(row["trial"]))]
        for key in self._keys:
            value = row.get(key)
            text = f"{value:g}" if isinstance(value, float) else str(value)
            cells.append((text, float(value) if isinstance(value, numbers.Real) else None))
        for column, _, fmt in RESULT_COLUMNS:
            value = row.get(column)
            if fmt is None:
                text = str(value)
                if column == "status" and isinstance(row.get("error"), str) and row["error"]:
                    text += f"：{row['error']}"
                cells.append((text, None))
            elif value is None or (isinstance(value, float) and math.isnan(value)):
                cells.append(("", math.nan))
            else:
                cells.append((fmt.format(value), float(value)))
        for c, (text, key) in enumerate(cells):
            item = _Item(text, key)
            item.setData(ROLE_TRIAL, int(row["trial"]))
            if key is not None:
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            else:
                item.setToolTip(text)  # 状态列可能带有较长的失败原因
            self.table.setItem(r, c, item)

    def _set_status(self, index: int, text: str) -> None:
        item = self.table.item(self._row_of(index), 1 + len(self._keys))
        if item is not None:
            item.setText(text)
        self._rows[index]["status"] = STATUS_RUNNING

    def _row_of(self, index: int) -> int:
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item is not None and item.data(ROLE_TRIAL) == index:
                return r
        return -1

    def _results_frame(self) -> pd.DataFrame:
        return pd.DataFrame(list(self._rows.values()))

    def _highlight_best(self) -> None:
        best = best_trial(self._results_frame(), self.rank_combo.currentData()) if self._rows else None
        best_index = int(best["trial"]) if best is not None else None
        for r in range(self.table.rowCount()):
            is_best = self.table.item(r, 0).data(ROLE_TRIAL) == best_index
            for c in range(self.table.columnCount()):
                item = self.table.item(r, c)
                if item is not None:
                    item.setBackground(BEST_BG if is_best else QColor(0, 0, 0, 0))

    def _sort_by_rank(self) -> None:
        metric = self.rank_combo.currentData()
        column = next(i for i, (c, _, _) in enumerate(RESULT_COLUMNS) if c == metric) + 1 + len(self._keys)
        order = Qt.SortOrder.AscendingOrder if metric in LOWER_IS_BETTER else Qt.SortOrder.DescendingOrder
        self.table.sortItems(column, order)

    def _on_rank_changed(self) -> None:
        if self._worker is None and self.table.isSortingEnabled():
            self._sort_by_rank()
        self._highlight_best()
        self._redraw_chart()

    def _redraw_chart(self) -> None:
        metric = self.rank_combo.currentData()
        results = self._results_frame() if self._rows else None
        axes = [(k, self._labels[k]) for k in self._keys]
        self.chart.update_results(results, axes, metric, RANK_METRICS[metric], metric in LOWER_IS_BETTER)

    def _selected_row(self) -> dict | None:
        items = self.table.selectedItems()
        if not items:
            QMessageBox.information(self, "提示", "请先在表格中选中一个组合")
            return None
        return self._rows.get(items[0].data(ROLE_TRIAL))

    def _trial_dir(self, row: dict) -> Path | None:
        if self._sweep_dir is None:
            return None
        path = Path(self._sweep_dir) / str(row["name"])
        if not (path / "history.csv").is_file():
            QMessageBox.information(self, "提示", f"组合 #{row['trial']} 还没有训练结果（{row['status']}）")
            return None
        return path

    def _show_selected(self) -> None:
        row = self._selected_row()
        path = self._trial_dir(row) if row else None
        if path is not None:
            self.showRunRequested.emit(path)

    def _apply_selected(self) -> None:
        row = self._selected_row()
        path = self._trial_dir(row) if row else None
        if path is not None:
            self.applyConfigRequested.emit(ExperimentConfig.load(path / "config.json"))

    def _open_folder(self) -> None:
        if self._sweep_dir is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._sweep_dir)))

    def _open_sweep_dialog(self) -> None:
        start = str(Path(self._sweep_dir).parent) if self._sweep_dir else ""
        folder = QFileDialog.getExistingDirectory(self, "选择参数遍历结果文件夹（名称含 sweep）", start)
        if not folder:
            return
        try:
            record = load_sweep(folder)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "打开失败", str(exc))
            return
        self.applyConfigRequested.emit(record.base)
        self.show_sweep(record)
        self.message.emit(f"已打开参数遍历结果：{folder}（基础配置已载入左侧）")

    # ------------------------------------------------------------------ 状态保存
    def state(self) -> dict:
        self._save_texts()
        return {
            "axes": {k: v for k, v in self._texts.items() if v.strip()},
            "workers": self.workers_spin.value(),
            "rank_by": self.rank_combo.currentData(),
        }

    def restore(self, state: dict) -> None:
        self._texts.update({k: str(v) for k, v in (state.get("axes") or {}).items()})
        for key, edit in self._edits.items():
            edit.setText(self._texts.get(key, ""))
        if isinstance(state.get("workers"), int):
            self.workers_spin.setValue(state["workers"])
        index = self.rank_combo.findData(state.get("rank_by"))
        if index >= 0:
            self.rank_combo.setCurrentIndex(index)

