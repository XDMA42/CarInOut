"""模型评估页：用已有的训练结果（模型）评估新的验证集 / 测试集，不再训练。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..training.config import ExperimentConfig
from ..training.data import SPLIT_NAMES, dataset_name
from ..training.evaluate import EvaluationRecord, evaluate_model, save_evaluation_figure
from ..training.runs import CONFIG_FILE, MODEL_FILE, Predictor
from .dataset_list import DatasetList
from .evaluation_view import EvaluationView


class EvaluateWorker(QObject):
    message = Signal(str)
    done = Signal(object)  # EvaluationRecord
    failed = Signal(str)

    def __init__(self, model_dir: str, val: list[str], test: list[str], threshold: float) -> None:
        super().__init__()
        self._args = (model_dir, val, test)
        self._threshold = threshold

    def run(self) -> None:
        try:
            record = evaluate_model(*self._args, log=self.message.emit, threshold=self._threshold)
        except Exception as exc:  # noqa: BLE001 —— 任何异常都要回报给界面，避免线程静默退出
            self.failed.emit(str(exc))
        else:
            self.done.emit(record)


class EvaluatePanel(QWidget):
    message = Signal(str)

    def __init__(
        self, current_run: Callable[[], Path | None], browse_dir: Callable[[], str], parent: QWidget | None = None
    ) -> None:
        """current_run 返回界面中当前显示的训练结果文件夹；browse_dir 返回选择数据文件夹时的起始位置。"""
        super().__init__(parent)
        self._current_run = current_run
        self._browse_dir = browse_dir
        self._thread: QThread | None = None
        self._worker: EvaluateWorker | None = None

        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText(f"训练结果文件夹（含 {MODEL_FILE}），参数遍历中某一组的文件夹也可以")
        self.model_edit.editingFinished.connect(self._update_model_info)
        browse = QPushButton("浏览…")
        browse.clicked.connect(self._choose_model)
        use_current = QPushButton("用当前训练结果")
        use_current.setToolTip("使用“评估结果”页正在显示的训练结果（刚训练完的，或打开的、在参数遍历中查看的）")
        use_current.clicked.connect(self._use_current)
        self.model_info = QLabel()
        self.model_info.setWordWrap(True)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.01, 0.99)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setValue(0.5)
        self.threshold_spin.setToolTip("车内概率 ≥ 阈值判为车内；选择模型后自动设为训练时的阈值，可改为其他值比较")

        self.val_list = DatasetList(
            "未指定验证集", "共 {n} 个文件夹", browse_dir, lambda: {p: "已在测试集中" for p in self.test_list.paths()}
        )
        self.test_list = DatasetList(
            "未指定测试集", "共 {n} 个文件夹", browse_dir, lambda: {p: "已在验证集中" for p in self.val_list.paths()}
        )
        self.start_btn = QPushButton("开始评估")
        self.start_btn.setMinimumWidth(110)
        self.start_btn.clicked.connect(self._start)

        box = QGroupBox("用已有模型评估（不训练）")
        grid = QGridLayout(box)
        grid.addWidget(QLabel("模型"), 0, 0)
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_edit, 1)
        model_row.addWidget(browse)
        model_row.addWidget(use_current)
        model_row.addSpacing(12)
        model_row.addWidget(QLabel("判定阈值"))
        model_row.addWidget(self.threshold_spin)
        model_row.addSpacing(12)
        model_row.addWidget(self.start_btn)
        grid.addLayout(model_row, 0, 1, 1, 2)
        grid.addWidget(self.model_info, 1, 1, 1, 2)
        grid.addWidget(QLabel("验证集（可多个文件夹）"), 2, 1)
        grid.addWidget(QLabel("测试集（可多个文件夹）"), 2, 2)
        grid.addWidget(self.val_list, 3, 1)
        grid.addWidget(self.test_list, 3, 2)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)

        self.view = EvaluationView(
            "选择模型和数据后点“开始评估”：模型会按训练时的方式选取特征、填充缺失值、标准化。\n"
            "结果保存在模型所在文件夹的 eval_时间 子文件夹中，之后可用“打开训练结果…”重新查看。"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(box)
        lay.addWidget(self.view, 1)
        self._update_model_info()

    # ------------------------------------------------------------------ 模型
    def _choose_model(self) -> None:
        start = self.model_edit.text().strip() or str(self._current_run() or "")
        folder = QFileDialog.getExistingDirectory(self, "选择训练结果文件夹（含 model.pt）", start)
        if folder:
            self.model_edit.setText(folder)
            self._update_model_info()

    def _use_current(self) -> None:
        run = self._current_run()
        if run is None:
            QMessageBox.information(self, "提示", "当前没有显示中的训练结果：请先训练，或用“打开训练结果…”打开一个")
            return
        self.model_edit.setText(str(run))
        self._update_model_info()

    def _update_model_info(self) -> None:
        """显示模型结构、输入特征数和训练数据，路径无效时提示。"""
        path = self.model_edit.text().strip()
        if not path:
            self.model_info.setStyleSheet("color: gray")
            self.model_info.setText("尚未选择模型")
            return
        try:
            predictor = Predictor.load(path)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            self.model_info.setStyleSheet("color: #d03b3b")
            self.model_info.setText(f"无法加载模型：{exc}")
            return
        self.threshold_spin.setValue(predictor.threshold)
        text = f"{predictor.describe()}；{len(predictor.features)} 个输入特征；训练时判定阈值 {predictor.threshold:g}"
        try:
            datasets = ExperimentConfig.load(Path(path) / CONFIG_FILE).data.datasets
            text += "；训练数据：" + "、".join(dataset_name(p) for p in datasets)
        except (OSError, ValueError):
            pass
        self.model_info.setStyleSheet("color: gray")
        self.model_info.setText(text)

    # ------------------------------------------------------------------ 评估
    def _start(self) -> None:
        model = self.model_edit.text().strip()
        if not model:
            QMessageBox.warning(self, "提示", "请先选择要评估的模型（训练结果文件夹）")
            return
        if not self.val_list.paths() and not self.test_list.paths():
            QMessageBox.warning(self, "提示", "请至少添加一个验证集或测试集文件夹")
            return
        self.start_btn.setEnabled(False)
        self.start_btn.setText("正在评估…")
        self.view.clear("正在评估…")
        self._thread = QThread(self)
        self._worker = EvaluateWorker(
            model, self.val_list.paths(), self.test_list.paths(), round(self.threshold_spin.value(), 4)
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.message.connect(self.message)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    def _cleanup(self) -> None:
        if self._thread:
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None
        self.start_btn.setEnabled(True)
        self.start_btn.setText("开始评估")

    def _on_done(self, record: EvaluationRecord) -> None:
        self._cleanup()
        save_evaluation_figure(record)  # matplotlib 在界面线程中画图
        self.show_record(record, restore_inputs=False)

    def _on_failed(self, error: str) -> None:
        self._cleanup()
        self.view.clear()
        self.message.emit(f"评估失败：{error}")
        QMessageBox.critical(self, "评估失败", error)

    def show_record(self, record: EvaluationRecord, restore_inputs: bool = True) -> None:
        """显示一次评估的结果。restore_inputs 时把它用的模型和数据填回输入框。"""
        m = record.metrics
        datasets = m.get("datasets", {})
        if restore_inputs:
            self.model_edit.setText(str(record.model_dir))
            self.val_list.set_paths(datasets.get("val", []))
            self.test_list.set_paths(datasets.get("test", []))
            self._update_model_info()
            self.threshold_spin.setValue(m.get("threshold", 0.5))
        used = "，".join(
            f"{SPLIT_NAMES[split]}：" + "、".join(dataset_name(p) for p in datasets[split])
            for split in ("val", "test")
            if split in datasets
        )
        header = (
            f"{record.path}\n模型 {record.model_dir.name}（{m.get('model_description', '')}）；{used}"
            f"；判定阈值 {m.get('threshold', 0.5):g}"
        )
        if m.get("notes"):
            header += "\n" + "；".join(m["notes"])
        self.view.show_results(record.path, header, m, record.predictions)

    def shutdown(self) -> None:
        if self._worker is not None:
            self._cleanup()

    # ------------------------------------------------------------------ 状态保存
    def state(self) -> dict:
        return {"model": self.model_edit.text().strip(), "val": self.val_list.paths(), "test": self.test_list.paths()}

    def restore(self, state: dict) -> None:
        self.model_edit.setText(str(state.get("model", "")))
        self.val_list.set_paths([str(p) for p in state.get("val", [])])
        self.test_list.set_paths([str(p) for p in state.get("test", [])])
        self._update_model_info()
