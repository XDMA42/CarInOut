"""模型训练平台主窗口：左侧为训练配置，右侧为训练过程与评估结果的可视化，底部为日志。"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
from PySide6.QtCore import QObject, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QDoubleValidator
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..anchors import AppConfig
from ..merger import COMBINED_FILENAME, POINT_COLUMN, SCENE_COLUMN, dataset_folder
from ..training.config import (
    MISSING_CONSTANT,
    MISSING_DROP,
    MISSING_MEAN,
    SPLIT_POINT,
    SPLIT_ROW,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimConfig,
)
from ..training.data import (
    CLASS_NAMES,
    NUM_CLASSES,
    SPLIT_NAMES,
    FeatureInfo,
    default_features,
    feature_candidates,
    load_table,
    parse_labels,
    prepare_data,
)
from ..training.metrics import METRIC_NAMES
from ..training.models import available_models, get_spec
from ..training.models.base import KIND_BOOL, KIND_CHOICE, KIND_INT, ModelSpec, ParamSpec
from ..training.plots import draw_losses, draw_metrics
from ..training.runs import (
    RunRecord,
    load_run,
    new_run_dir,
    per_point_accuracy,
    save_run,
    save_run_figures,
    split_scores,
)
from ..training.sweep import is_sweep_dir, load_sweep
from ..training.trainer import HISTORY_COLUMNS, fit
from .sweep_panel import SweepPanel
from .train_charts import EpochChart, EvaluationChart

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = PROJECT_ROOT / "train_gui.json"  # 上次使用的训练配置，下次启动自动恢复
ROLE_NAME = Qt.ItemDataRole.UserRole
WARN_BG = QColor(245, 158, 11, 60)
BEST_BG = QColor(42, 120, 214, 40)

# 训练记录表格的列，顺序同 trainer.HISTORY_COLUMNS
HISTORY_HEADERS = {
    "epoch": "epoch",
    "train_loss": "训练总损失",
    "train_ce": "训练交叉熵",
    "l2": "L2 项",
    "val_loss": "验证总损失",
    "val_ce": "验证交叉熵",
    "train_accuracy": "训练准确率",
    "val_accuracy": "验证准确率",
    "val_precision": "验证精确率",
    "val_recall": "验证召回率",
    "val_f1": "验证 F1",
    "val_auc": "验证 AUC",
    "seconds": "耗时 (s)",
}
MISSING_TEXT = {
    MISSING_DROP: "丢弃含缺失值的行",
    MISSING_MEAN: "用训练集均值填充",
    MISSING_CONSTANT: "用固定值填充",
}
SPLIT_TEXT = {SPLIT_POINT: "按点位整组划分（推荐）", SPLIT_ROW: "按行随机划分"}


class TrainWorker(QObject):
    epoch = Signal(object)
    message = Signal(str)
    done = Signal(object)  # 训练结果文件夹
    failed = Signal(str)

    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        self._cfg = cfg
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            data = prepare_data(self._cfg.data, self._cfg.train.seed, log=self.message.emit)
            result = fit(self._cfg, data, self.epoch.emit, lambda: self._stop, self.message.emit)
            run_dir = new_run_dir(self._cfg)
            # 图表在界面线程中绘制保存（matplotlib 不保证线程安全）
            save_run(run_dir, self._cfg, data, result, figures=False)
        except Exception as exc:  # noqa: BLE001 —— 任何异常都要回报给界面，避免线程静默退出
            self.failed.emit(str(exc))
        else:
            if result.stopped:
                self.message.emit(f"已手动停止，完成了 {len(result.history)} 个 epoch")
            self.done.emit(run_dir)


class TrainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("钥匙车内外识别 · 模型训练")
        self.resize(1560, 960)

        self._ready = False  # 界面搭建完成后才同步参数遍历页
        self._infos: list[FeatureInfo] = []
        self._loaded_dataset = ""  # 特征列表对应的数据集路径
        self._label_column = DataConfig().label_column
        self._param_widgets: dict[str, tuple[ParamSpec, QWidget]] = {}
        self._history: list[dict] = []
        self._run: RunRecord | None = None
        self._thread: QThread | None = None
        self._worker: TrainWorker | None = None
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(300)  # 训练很快时限制重绘频率
        self._redraw_timer.timeout.connect(self._redraw_curves)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.log_view.setPlaceholderText("日志")

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self._build_tabs())
        right.addWidget(self.log_view)
        right.setStretchFactor(0, 1)
        right.setSizes([760, 150])

        main = QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(self._build_config_panel())
        main.addWidget(right)
        main.setStretchFactor(1, 1)
        main.setSizes([430, 1130])
        container = QWidget()
        lay = QVBoxLayout(container)
        lay.setContentsMargins(8, 8, 8, 0)
        lay.addWidget(main)
        self.setCentralWidget(container)
        self.statusBar()

        self._ready = True
        cfg, sweep_state = self._load_state()
        self._apply_config(cfg)
        self.sweep_panel.restore(sweep_state)

    # ------------------------------------------------------------------ 界面搭建
    def _build_config_panel(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 6, 0)
        lay.addWidget(self._build_data_group())
        lay.addWidget(self._build_feature_group(), 1)
        lay.addWidget(self._build_processing_group())
        lay.addWidget(self._build_model_group())
        lay.addWidget(self._build_train_group())
        lay.addWidget(self._build_output_group())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(400)

        self.start_btn = QPushButton("开始训练")
        self.start_btn.setMinimumHeight(34)
        self.start_btn.clicked.connect(self._start_or_stop)
        self.progress = QProgressBar()
        self.progress.setFormat("%v / %m epoch")
        self.progress.setValue(0)
        load_cfg = QPushButton("加载配置…")
        load_cfg.clicked.connect(self._load_config_file)
        save_cfg = QPushButton("保存配置…")
        save_cfg.setToolTip("保存为 JSON，可用于命令行训练：python train.py --config 文件名")
        save_cfg.clicked.connect(self._save_config_file)
        open_run = QPushButton("打开训练结果…")
        open_run.setToolTip("查看以前某次训练（或参数遍历）的结果，并把它的配置载入左侧")
        open_run.clicked.connect(self._open_run_dialog)
        brow = QHBoxLayout()
        brow.addWidget(load_cfg)
        brow.addWidget(save_cfg)
        brow.addWidget(open_run)

        outer = QWidget()
        olay = QVBoxLayout(outer)
        olay.setContentsMargins(0, 0, 0, 0)
        olay.addWidget(scroll, 1)
        olay.addWidget(self.start_btn)
        olay.addWidget(self.progress)
        olay.addLayout(brow)
        return outer

    def _build_data_group(self) -> QWidget:
        box = QGroupBox("数据集")
        self.dataset_edit = QLineEdit()
        self.dataset_edit.setPlaceholderText(f"数据集构建工具导出的 {COMBINED_FILENAME}")
        self.dataset_edit.returnPressed.connect(lambda: self._load_dataset(self.dataset_edit.text().strip()))
        browse = QPushButton("浏览…")
        browse.clicked.connect(self._choose_dataset)
        row = QHBoxLayout()
        row.addWidget(self.dataset_edit, 1)
        row.addWidget(browse)
        self.dataset_info = QLabel("尚未加载数据集")
        self.dataset_info.setWordWrap(True)
        self.dataset_info.setStyleSheet("color: gray")

        self.test_edit = QLineEdit()
        self.test_edit.setPlaceholderText("可选：独立测试集，如另一天采集的总表")
        self.test_edit.setToolTip("指定后用该文件全部数据做测试集，上面的数据集只划分训练集和验证集")
        self.test_edit.textChanged.connect(self._update_ratio_state)
        tbrowse = QPushButton("浏览…")
        tbrowse.clicked.connect(self._choose_test_dataset)
        tclear = QPushButton("清除")
        tclear.clicked.connect(self.test_edit.clear)
        trow = QHBoxLayout()
        trow.addWidget(self.test_edit, 1)
        trow.addWidget(tbrowse)
        trow.addWidget(tclear)

        lay = QVBoxLayout(box)
        lay.addLayout(row)
        lay.addWidget(self.dataset_info)
        lay.addWidget(QLabel("独立测试集"))
        lay.addLayout(trow)
        return box

    def _build_feature_group(self) -> QWidget:
        box = QGroupBox("输入特征")
        self.feature_filter = QLineEdit()
        self.feature_filter.setPlaceholderText("筛选列名，如 rssi")
        self.feature_filter.setClearButtonEnabled(True)
        self.feature_filter.textChanged.connect(self._filter_features)
        check_all = QPushButton("全选")
        check_all.setToolTip("勾选当前显示（符合筛选条件）的所有特征")
        check_all.clicked.connect(lambda: self._check_visible(True))
        check_none = QPushButton("全不选")
        check_none.setToolTip("取消勾选当前显示（符合筛选条件）的所有特征")
        check_none.clicked.connect(lambda: self._check_visible(False))
        reset = QPushButton("默认")
        reset.setToolTip("勾选所有非常数的数值列")
        reset.clicked.connect(self._check_default_features)
        frow = QHBoxLayout()
        frow.addWidget(self.feature_filter, 1)
        frow.addWidget(check_all)
        frow.addWidget(check_none)
        frow.addWidget(reset)

        self.feature_tree = QTreeWidget()
        self.feature_tree.setHeaderLabels(["特征列", "缺失"])
        self.feature_tree.setRootIsDecorated(False)
        self.feature_tree.setUniformRowHeights(True)
        self.feature_tree.setMinimumHeight(160)
        self.feature_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.feature_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.feature_tree.header().setStretchLastSection(False)
        self.feature_tree.itemChanged.connect(self._on_features_changed)
        self.feature_count = QLabel()

        lay = QVBoxLayout(box)
        lay.addLayout(frow)
        lay.addWidget(self.feature_tree, 1)
        lay.addWidget(self.feature_count)
        return box

    def _build_processing_group(self) -> QWidget:
        box = QGroupBox("数据处理")
        self.missing_combo = QComboBox()
        for key, text in MISSING_TEXT.items():
            self.missing_combo.addItem(text, key)
        self.missing_combo.currentIndexChanged.connect(self._update_fill_state)
        self.fill_spin = QDoubleSpinBox()
        self.fill_spin.setRange(-1e6, 1e6)
        self.fill_spin.setDecimals(2)
        self.fill_spin.setToolTip("例如 RSSI 缺失表示信号太弱，可填 -120")
        mrow = QHBoxLayout()
        mrow.addWidget(self.missing_combo, 1)
        mrow.addWidget(self.fill_spin)
        self.standardize_check = QCheckBox("z-score 标准化（按训练集均值、标准差）")
        self.split_combo = QComboBox()
        for key, text in SPLIT_TEXT.items():
            self.split_combo.addItem(text, key)
        self.split_combo.setToolTip(
            "按点位整组划分：同一 (场景, 点位) 的数据只出现在一个集合中，评估的是对没见过的位置的泛化能力。\n"
            "按行随机划分：同一点位相邻时刻的数据会同时出现在训练集和验证集，指标会偏乐观。"
        )
        self.val_spin = self._ratio_spin()
        self.test_spin = self._ratio_spin()
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2**31 - 1)
        self.seed_spin.setToolTip("用于数据划分和模型初始化，相同种子可复现结果")

        form = QFormLayout(box)
        form.addRow("缺失值", mrow)
        form.addRow(self.standardize_check)
        form.addRow("划分方式", self.split_combo)
        form.addRow("验证集比例", self.val_spin)
        form.addRow("测试集比例", self.test_spin)
        form.addRow("随机种子", self.seed_spin)
        return box

    def _build_model_group(self) -> QWidget:
        box = QGroupBox("模型")
        self.model_combo = QComboBox()
        for spec in available_models():
            self.model_combo.addItem(spec.label, spec.name)
        self.model_combo.currentIndexChanged.connect(lambda: self._build_param_form(self._current_spec()))
        self.param_form = QFormLayout()
        self.structure_label = QLabel()
        self.structure_label.setWordWrap(True)
        lay = QFormLayout(box)
        lay.addRow("模型", self.model_combo)
        lay.addRow(self.param_form)
        lay.addRow(self.structure_label)
        return box

    def _build_train_group(self) -> QWidget:
        box = QGroupBox("训练参数")
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1_000_000)
        self.lr_edit = self._float_edit("学习率，如 0.001 或 1e-3")
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1_000_000)
        self.l2_edit = self._float_edit("总损失 = 交叉熵 + λ × Σ‖W‖²（只对权重，不含偏置）；0 表示不加 L2")
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItem("Adam", "adam")
        self.optimizer_combo.addItem("SGD（momentum 0.9）", "sgd")
        self.class_weight_combo = QComboBox()
        self.class_weight_combo.addItem("不加权", "none")
        self.class_weight_combo.addItem("按样本数平衡", "balanced")
        self.class_weight_combo.setToolTip("车内、车外样本数相差较大时，可按样本数的倒数给交叉熵加权")
        self.early_spin = QSpinBox()
        self.early_spin.setRange(0, 1_000_000)
        self.early_spin.setSpecialValueText("不提前停止")
        self.early_spin.setSuffix(" 个 epoch 无改善即停止")
        self.early_spin.setToolTip(
            "验证集交叉熵连续这么多个 epoch 没有下降时提前结束；训练结束后总是使用验证集交叉熵最低的模型"
        )
        self.device_combo = QComboBox()
        self.device_combo.addItem("自动（有 GPU 时使用 GPU）", "auto")
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.addItem("CUDA GPU", "cuda")

        for signal in (self.batch_spin.valueChanged, self.lr_edit.textChanged, self.epochs_spin.valueChanged,
                       self.l2_edit.textChanged):
            signal.connect(self._refresh_sweep_axes)

        form = QFormLayout(box)
        form.addRow("batch size", self.batch_spin)
        form.addRow("learning rate", self.lr_edit)
        form.addRow("epoch 数", self.epochs_spin)
        form.addRow("L2 系数 λ", self.l2_edit)
        form.addRow("优化器", self.optimizer_combo)
        form.addRow("类别权重", self.class_weight_combo)
        form.addRow("提前停止", self.early_spin)
        form.addRow("设备", self.device_combo)
        return box

    def _build_output_group(self) -> QWidget:
        box = QGroupBox("训练结果保存位置")
        self.output_edit = QLineEdit()
        browse = QPushButton("浏览…")
        browse.clicked.connect(self._choose_output)
        row = QHBoxLayout()
        row.addWidget(self.output_edit, 1)
        row.addWidget(browse)
        self.run_name_edit = QLineEdit()
        self.run_name_edit.setPlaceholderText("可选，如 h64-32_lr1e-3")
        form = QFormLayout(box)
        form.addRow("文件夹", row)
        form.addRow("本次名称", self.run_name_edit)
        note = QLabel("每次训练新建子文件夹“时间_名称”，保存配置、曲线数据、指标、预测结果、模型和图表")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray")
        form.addRow(note)
        return box

    def _build_tabs(self) -> QWidget:
        self.tabs = QTabWidget()
        self.loss_chart = EpochChart(draw_losses, 12, 4)
        self.metric_chart = EpochChart(draw_metrics, 10, 4)
        self.tabs.addTab(self.loss_chart, "损失曲线")
        self.tabs.addTab(self.metric_chart, "评价指标")
        self.tabs.addTab(self._build_eval_tab(), "评估结果")

        self.history_table = self._table([HISTORY_HEADERS[c] for c in HISTORY_COLUMNS])
        self.tabs.addTab(self.history_table, "训练记录")

        self.sweep_panel = SweepPanel(self._checked_config)
        self.sweep_panel.runningChanged.connect(self._on_sweep_running)
        self.sweep_panel.showRunRequested.connect(self._show_run_dir)
        self.sweep_panel.applyConfigRequested.connect(self._apply_sweep_config)
        self.sweep_panel.message.connect(self._log)
        self.tabs.addTab(self.sweep_panel, "参数遍历")
        return self.tabs

    def _build_eval_tab(self) -> QWidget:
        self.run_label = QLabel("训练完成后在这里显示最佳模型在各数据集上的表现")
        self.run_label.setWordWrap(True)
        self.run_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.open_run_btn = QPushButton("打开结果文件夹")
        self.open_run_btn.setEnabled(False)
        self.open_run_btn.clicked.connect(self._open_run_folder)
        head = QHBoxLayout()
        head.addWidget(self.run_label, 1)
        head.addWidget(self.open_run_btn)

        self.summary_table = self._table(["数据集", "样本数", "交叉熵", *METRIC_NAMES.values()])
        self.summary_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        rows = self.summary_table.verticalHeader().defaultSectionSize() * 3  # 训练 / 验证 / 测试
        self.summary_table.setFixedHeight(self.summary_table.horizontalHeader().sizeHint().height() + rows + 4)
        self.eval_chart = EvaluationChart()
        self.point_table = self._table(["数据集", "场景", "点位", "真实类别", "样本数", "准确率", "平均车内概率"])
        point_box = QWidget()
        play = QVBoxLayout(point_box)
        play.setContentsMargins(0, 0, 0, 0)
        hint = QLabel("各点位准确率（验证集、测试集，从低到高）—— 找出容易判错的位置")
        hint.setStyleSheet("color: gray")
        play.addWidget(hint)
        play.addWidget(self.point_table)

        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.eval_chart)
        split.addWidget(point_box)
        split.setSizes([420, 200])

        self.eval_page = QWidget()
        lay = QVBoxLayout(self.eval_page)
        lay.addLayout(head)
        lay.addWidget(self.summary_table)
        lay.addWidget(split, 1)
        return self.eval_page

    @staticmethod
    def _table(headers: list[str]) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        return table

    @staticmethod
    def _ratio_spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0, 0.9)
        spin.setSingleStep(0.05)
        spin.setDecimals(2)
        return spin

    @staticmethod
    def _float_edit(tooltip: str) -> QLineEdit:
        edit = QLineEdit()
        validator = QDoubleValidator(0.0, 1e9, 12)
        validator.setNotation(QDoubleValidator.Notation.ScientificNotation)
        edit.setValidator(validator)
        edit.setToolTip(tooltip)
        return edit

    # ------------------------------------------------------------------ 数据集与特征
    def _choose_dataset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择数据集", self.dataset_edit.text() or str(PROJECT_ROOT), "CSV 文件 (*.csv);;所有文件 (*)"
        )
        if path:
            self.dataset_edit.setText(path)
            self._load_dataset(path)

    def _choose_test_dataset(self) -> None:
        start = self.test_edit.text() or self.dataset_edit.text() or str(PROJECT_ROOT)
        path, _ = QFileDialog.getOpenFileName(self, "选择独立测试集", start, "CSV 文件 (*.csv);;所有文件 (*)")
        if path:
            self.test_edit.setText(path)

    def _load_dataset(self, path: str, keep: list[str] | None = None) -> None:
        """读取数据集并列出候选特征。keep 为要勾选的特征（来自配置），为 None 时沿用当前勾选或使用默认。"""
        if keep is None and self._loaded_dataset:
            keep = self._checked_features()
        self._infos, self._loaded_dataset = [], ""
        if not path:
            self._populate_features([])
            self.dataset_info.setText("尚未加载数据集")
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            df = load_table(path)
            infos = feature_candidates(df, self._label_column)
            labels = parse_labels(df, self._label_column)
        except ValueError as exc:
            self._populate_features([])
            self.dataset_info.setText(f"加载失败：{exc}")
            self._log(f"加载数据集失败：{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        self._infos, self._loaded_dataset = infos, path
        counts = labels.value_counts()
        parts = [f"{len(df):,} 行", "，".join(f"{CLASS_NAMES[c]} {int(counts.get(c, 0)):,}" for c in CLASS_NAMES)]
        if {SCENE_COLUMN, POINT_COLUMN} <= set(df.columns):
            n_points = len(df.groupby([SCENE_COLUMN, POINT_COLUMN]))
            parts.append(f"{df[SCENE_COLUMN].nunique()} 个场景，{n_points} 个点位")
        parts.append(f"{len(infos)} 个数值特征列")
        self.dataset_info.setText("；".join(parts))
        self._log(f"加载数据集：{path}（{'；'.join(parts)}）")

        names = [i.name for i in infos]
        if keep:
            absent = [f for f in keep if f not in names]
            if absent:
                self._log(f"  配置中的特征在该数据集中不存在，已忽略：{'、'.join(absent)}")
            keep = [f for f in keep if f in names]
        self._populate_features(keep or default_features(infos))

    def _populate_features(self, checked: list[str]) -> None:
        self.feature_tree.blockSignals(True)
        self.feature_tree.clear()
        for info in self._infos:
            text = info.name + ("（常数）" if info.constant else "")
            item = QTreeWidgetItem([text, f"{info.missing_ratio:.1%}" if info.missing_ratio else ""])
            item.setData(0, ROLE_NAME, info.name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Checked if info.name in checked else Qt.CheckState.Unchecked)
            if info.constant:
                item.setToolTip(0, "所有行的值都相同，对模型没有信息量")
            if info.missing_ratio:
                item.setToolTip(1, "空值所占比例，按“数据处理 → 缺失值”的方式处理")
            self.feature_tree.addTopLevelItem(item)
        self.feature_tree.blockSignals(False)
        self._filter_features(self.feature_filter.text())
        self._on_features_changed()

    def _feature_items(self) -> list[QTreeWidgetItem]:
        return [self.feature_tree.topLevelItem(i) for i in range(self.feature_tree.topLevelItemCount())]

    def _checked_features(self) -> list[str]:
        return [it.data(0, ROLE_NAME) for it in self._feature_items() if it.checkState(0) == Qt.CheckState.Checked]

    def _filter_features(self, text: str) -> None:
        text = text.strip().lower()
        for item in self._feature_items():
            item.setHidden(bool(text) and text not in item.data(0, ROLE_NAME).lower())

    def _check_visible(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self.feature_tree.blockSignals(True)
        for item in self._feature_items():
            if not item.isHidden():
                item.setCheckState(0, state)
        self.feature_tree.blockSignals(False)
        self._on_features_changed()

    def _check_default_features(self) -> None:
        self._populate_features(default_features(self._infos))

    def _on_features_changed(self, *_args) -> None:
        n = len(self._checked_features())
        self.feature_count.setText(f"已选 {n} / {len(self._infos)} 个特征（即模型输入维度）")
        self._update_structure()

    # ------------------------------------------------------------------ 模型参数
    def _current_spec(self) -> ModelSpec:
        return get_spec(self.model_combo.currentData())

    def _build_param_form(self, spec: ModelSpec, values: dict | None = None) -> None:
        """按模型声明的超参数生成输入控件。"""
        while self.param_form.rowCount():
            self.param_form.removeRow(0)
        self._param_widgets = {}
        self.model_combo.setToolTip(spec.description)
        try:
            values = spec.resolve_params(values)
        except ValueError as exc:
            self._log(f"模型参数无效，已恢复默认值：{exc}")
            values = spec.resolve_params(None)
        for param in spec.params:
            value = values[param.name]
            if param.kind == KIND_INT:
                widget = QSpinBox()
                low = int(param.minimum) if param.minimum is not None else -(10**9)
                widget.setRange(low, int(param.maximum) if param.maximum is not None else 10**9)
                widget.setValue(value)
                widget.valueChanged.connect(self._update_structure)
            elif param.kind == KIND_CHOICE:
                widget = QComboBox()
                widget.addItems(list(param.choices))
                widget.setCurrentText(value)
                widget.currentIndexChanged.connect(self._update_structure)
            elif param.kind == KIND_BOOL:
                widget = QCheckBox()
                widget.setChecked(value)
                widget.toggled.connect(self._update_structure)
            else:  # 浮点数、整数列表用文本框
                widget = QLineEdit(param.format(value))
                widget.textChanged.connect(self._update_structure)
            widget.setToolTip(param.help)
            self.param_form.addRow(param.label, widget)
            self._param_widgets[param.name] = (param, widget)
        self._update_structure()

    def _param_values(self) -> dict:
        """读取模型参数并校验，非法时抛出 ValueError。"""
        values = {}
        for name, (param, widget) in self._param_widgets.items():
            if isinstance(widget, QSpinBox):
                raw = widget.value()
            elif isinstance(widget, QComboBox):
                raw = widget.currentText()
            elif isinstance(widget, QCheckBox):
                raw = widget.isChecked()
            else:
                raw = widget.text()
            values[name] = param.parse(raw)
        return values

    def _update_structure(self, *_args) -> None:
        if not hasattr(self, "structure_label"):
            return
        n_features = len(self._checked_features())
        try:
            params = self._param_values()
            text = self._current_spec().summary(max(n_features, 1), NUM_CLASSES, params)
        except ValueError as exc:
            self.structure_label.setStyleSheet("color: #d03b3b")
            self.structure_label.setText(str(exc))
            return
        self.structure_label.setStyleSheet("color: gray")
        self.structure_label.setText(("结构：" + text) if n_features else "请先选择输入特征")
        self._refresh_sweep_axes()

    def _refresh_sweep_axes(self, *_args) -> None:
        """模型或隐藏层数变化时，参数遍历页的可选参数随之变化。"""
        if not self._ready:
            return
        try:
            cfg = self._collect_config()
        except ValueError:
            return
        self.sweep_panel.refresh_axes(cfg)

    # ------------------------------------------------------------------ 配置 ↔ 界面
    def _collect_config(self) -> ExperimentConfig:
        """从界面读取配置，非法时抛出 ValueError。"""
        def number(edit: QLineEdit, name: str) -> float:
            try:
                return float(edit.text())
            except ValueError:
                raise ValueError(f"{name} 不是有效的数字") from None

        cfg = ExperimentConfig(
            data=DataConfig(
                dataset=self.dataset_edit.text().strip(),
                test_dataset=self.test_edit.text().strip(),
                features=self._checked_features(),
                label_column=self._label_column,
                missing=self.missing_combo.currentData(),
                fill_value=self.fill_spin.value(),
                standardize=self.standardize_check.isChecked(),
                split=self.split_combo.currentData(),
                val_ratio=round(self.val_spin.value(), 4),
                test_ratio=round(self.test_spin.value(), 4),
            ),
            model=ModelConfig(self.model_combo.currentData(), self._param_values()),
            train=OptimConfig(
                batch_size=self.batch_spin.value(),
                learning_rate=number(self.lr_edit, "learning rate"),
                epochs=self.epochs_spin.value(),
                l2_lambda=number(self.l2_edit, "L2 系数"),
                optimizer=self.optimizer_combo.currentData(),
                class_weight=self.class_weight_combo.currentData(),
                early_stopping=self.early_spin.value(),
                seed=self.seed_spin.value(),
                device=self.device_combo.currentData(),
            ),
            output_dir=self.output_edit.text().strip() or str(PROJECT_ROOT / "runs"),
            run_name=self.run_name_edit.text().strip(),
        )
        cfg.validate()
        return cfg

    def _apply_config(self, cfg: ExperimentConfig) -> None:
        d, t = cfg.data, cfg.train
        self._label_column = d.label_column
        self.test_edit.setText(d.test_dataset)
        self._set_combo(self.missing_combo, d.missing)
        self.fill_spin.setValue(d.fill_value)
        self.standardize_check.setChecked(d.standardize)
        self._set_combo(self.split_combo, d.split)
        self.val_spin.setValue(d.val_ratio)
        self.test_spin.setValue(d.test_ratio)
        self.seed_spin.setValue(t.seed)

        self.batch_spin.setValue(t.batch_size)
        self.lr_edit.setText(f"{t.learning_rate:g}")
        self.epochs_spin.setValue(t.epochs)
        self.l2_edit.setText(f"{t.l2_lambda:g}")
        self._set_combo(self.optimizer_combo, t.optimizer)
        self._set_combo(self.class_weight_combo, t.class_weight)
        self.early_spin.setValue(t.early_stopping)
        self._set_combo(self.device_combo, t.device)
        self.output_edit.setText(cfg.output_dir)
        self.run_name_edit.setText(cfg.run_name)

        self.model_combo.blockSignals(True)
        if not self._set_combo(self.model_combo, cfg.model.name):
            self._log(f"未知的模型 {cfg.model.name}，已改为 {self.model_combo.currentText()}")
        self.model_combo.blockSignals(False)
        spec = self._current_spec()
        self._build_param_form(spec, cfg.model.params if spec.name == cfg.model.name else None)

        self.dataset_edit.setText(d.dataset)
        self._loaded_dataset = ""
        self._load_dataset(d.dataset, list(d.features))
        self._update_fill_state()
        self._update_ratio_state()

    @staticmethod
    def _set_combo(combo: QComboBox, data) -> bool:
        index = combo.findData(data)
        if index >= 0:
            combo.setCurrentIndex(index)
        return index >= 0

    def _update_fill_state(self) -> None:
        self.fill_spin.setEnabled(self.missing_combo.currentData() == MISSING_CONSTANT)

    def _update_ratio_state(self) -> None:
        has_test = bool(self.test_edit.text().strip())
        self.test_spin.setEnabled(not has_test)
        self.test_spin.setToolTip("已指定独立测试集，此项不生效" if has_test else "从数据集中划出的测试集比例")

    def _load_state(self) -> tuple[ExperimentConfig, dict]:
        """返回 (训练配置, 参数遍历页的设置)。"""
        cfg = ExperimentConfig(output_dir=str(PROJECT_ROOT / "runs"))
        if STATE_PATH.exists():
            try:
                state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
                sweep = state.get("sweep")
                return ExperimentConfig.from_dict(state.get("config", {})), sweep if isinstance(sweep, dict) else {}
            except (OSError, ValueError, AttributeError) as exc:
                self._log(f"上次的训练配置无法读取，使用默认配置：{exc}")
                return cfg, {}
        # 第一次打开：尝试使用数据集构建工具上次导出的总表
        tool = AppConfig.load()
        if tool.last_input_dir:
            base = Path(tool.last_output_dir) if tool.last_output_dir else None
            candidate = dataset_folder(Path(tool.last_input_dir), base) / COMBINED_FILENAME
            if candidate.is_file():
                cfg.data.dataset = str(candidate)
        return cfg, {}

    def _save_state(self) -> None:
        try:
            cfg = self._collect_config()
        except ValueError:
            return
        try:
            STATE_PATH.write_text(
                json.dumps({"config": cfg.to_dict(), "sweep": self.sweep_panel.state()}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            self.statusBar().showMessage(f"配置保存失败：{exc}", 5000)

    def _load_config_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "加载训练配置", str(PROJECT_ROOT), "JSON (*.json)")
        if not path:
            return
        try:
            cfg = ExperimentConfig.load(Path(path))
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "加载失败", str(exc))
            return
        self._apply_config(cfg)
        self._log(f"已加载配置：{path}")

    def _save_config_file(self) -> None:
        try:
            cfg = self._collect_config()
        except ValueError as exc:
            QMessageBox.warning(self, "配置有误", str(exc))
            return
        start = str(PROJECT_ROOT / "train_config.json")
        path, _ = QFileDialog.getSaveFileName(self, "保存训练配置", start, "JSON (*.json)")
        if path:
            cfg.save(Path(path))
            self._log(f"已保存配置：{path}（命令行训练：python train.py --config \"{path}\"）")

    def _choose_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择训练结果保存位置", self.output_edit.text())
        if folder:
            self.output_edit.setText(folder)

    # ------------------------------------------------------------------ 训练
    def _start_or_stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self.start_btn.setEnabled(False)
            self.start_btn.setText("正在停止…")
            return
        self._start_training()

    def _checked_config(self) -> ExperimentConfig | None:
        """读取左侧配置并做训练前检查；有问题时提示并返回 None。单次训练和参数遍历共用。"""
        try:
            cfg = self._collect_config()
        except ValueError as exc:
            QMessageBox.warning(self, "配置有误", str(exc))
            return None
        if not cfg.data.dataset:
            QMessageBox.warning(self, "提示", "请先选择数据集")
            return None
        if cfg.data.dataset != self._loaded_dataset:
            self._load_dataset(cfg.data.dataset)
            cfg.data.features = self._checked_features()
        if not cfg.data.features:
            QMessageBox.warning(self, "提示", "请至少勾选一个输入特征")
            return None
        self._save_state()
        return cfg

    def _start_training(self) -> None:
        cfg = self._checked_config()
        if cfg is None:
            return

        self._history = []
        self._run = None
        self.history_table.setRowCount(0)
        self._redraw_curves()
        self._show_evaluation(None)
        self.progress.setRange(0, cfg.train.epochs)
        self.progress.setValue(0)
        self.start_btn.setText("停止训练")
        self.tabs.setCurrentWidget(self.loss_chart)
        self._log(
            f"开始训练：{len(cfg.data.features)} 个特征，batch size {cfg.train.batch_size}，"
            f"learning rate {cfg.train.learning_rate:g}，L2 λ {cfg.train.l2_lambda:g}，{cfg.train.epochs} 个 epoch"
        )

        # 信号都连接到主窗口的方法，保证回调在界面线程执行
        self._thread = QThread(self)
        self._worker = TrainWorker(cfg)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self.sweep_panel.set_busy(True)
        self._worker.epoch.connect(self._on_epoch)
        self._worker.message.connect(self._log)
        self._worker.done.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    def _on_epoch(self, record: dict) -> None:
        self._history.append(record)
        self.progress.setValue(record["epoch"])
        self._append_history_row(record)
        text = f"epoch {record['epoch']}：训练集总损失 {record['train_loss']:.4f}"
        if not math.isnan(record["val_ce"]):
            text += f"，验证集交叉熵 {record['val_ce']:.4f}，准确率 {record['val_accuracy']:.2%}"
        self.statusBar().showMessage(text)
        if not self._redraw_timer.isActive():
            self._redraw_timer.start()

    def _live_best_epoch(self) -> int:
        """与 trainer 的选择规则一致：有验证集时取验证集交叉熵最低的 epoch，否则取训练集总损失最低的。"""
        if not self._history:
            return 0
        df = pd.DataFrame(self._history)
        column = "val_ce" if df["val_ce"].notna().any() else "train_loss"
        return int(df.loc[df[column].idxmin(), "epoch"])

    def _redraw_curves(self) -> None:
        best = self._run.metrics.get("best_epoch", 0) if self._run else self._live_best_epoch()
        self.loss_chart.update_history(self._history, best)
        self.metric_chart.update_history(self._history, best)

    def _cleanup_worker(self) -> None:
        if self._thread:
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None
        self.start_btn.setText("开始训练")
        self.start_btn.setEnabled(True)
        self.sweep_panel.set_busy(False)

    def _on_sweep_running(self, running: bool) -> None:
        """参数遍历进行中时不能同时单独训练。"""
        self.start_btn.setEnabled(not running)
        self.start_btn.setToolTip("参数遍历进行中" if running else "")

    def _on_finished(self, run_dir: Path) -> None:
        self._cleanup_worker()
        try:
            run = load_run(run_dir)
            save_run_figures(run)
        except (OSError, ValueError) as exc:
            self._log(f"读取训练结果失败：{exc}")
            return
        self._show_run(run)
        self.tabs.setCurrentWidget(self.eval_page)
        self._log(f"训练结果已保存到 {run_dir}")

    def _on_failed(self, error: str) -> None:
        self._cleanup_worker()
        self._log(f"训练失败：{error}")
        QMessageBox.critical(self, "训练失败", error)

    # ------------------------------------------------------------------ 结果展示
    def _open_run_dialog(self) -> None:
        start = self.output_edit.text() or str(PROJECT_ROOT / "runs")
        folder = QFileDialog.getExistingDirectory(self, "选择训练结果或参数遍历结果文件夹", start)
        if not folder:
            return
        if is_sweep_dir(folder):
            try:
                record = load_sweep(folder)
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, "打开失败", str(exc))
                return
            self._apply_config(record.base)
            self.sweep_panel.show_sweep(record)
            self.tabs.setCurrentWidget(self.sweep_panel)
            self._log(f"已打开参数遍历结果：{folder}（基础配置已载入左侧）")
            return
        try:
            run = load_run(folder)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "打开失败", str(exc))
            return
        self._apply_config(run.config)
        self._show_run(run)
        self._log(f"已打开训练结果：{folder}（其配置已载入左侧）")

    def _show_run_dir(self, path: Path) -> None:
        """参数遍历页中查看某个组合：在损失曲线、评价指标、评估结果、训练记录页显示它。"""
        try:
            run = load_run(path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "打开失败", str(exc))
            return
        self._show_run(run)
        self.tabs.setCurrentWidget(self.loss_chart)
        self._log(f"查看组合：{path}")

    def _apply_sweep_config(self, cfg: ExperimentConfig) -> None:
        self._apply_config(cfg)
        self._log("已把参数组合载入左侧配置")

    def _show_run(self, run: RunRecord) -> None:
        self._run = run
        self._history = run.history.to_dict("records")
        self._redraw_curves()
        self.history_table.setRowCount(0)
        best = run.metrics.get("best_epoch", 0)
        for record in self._history:
            self._append_history_row(record, highlight=record["epoch"] == best)
        self.progress.setRange(0, max(run.config.train.epochs, 1))
        self.progress.setValue(len(self._history))
        self._show_evaluation(run)

    def _append_history_row(self, record: dict, highlight: bool = False) -> None:
        row = self.history_table.rowCount()
        self.history_table.insertRow(row)
        for col, key in enumerate(HISTORY_COLUMNS):
            value = record.get(key, math.nan)
            if key == "epoch":
                text = str(int(value))
            elif value is None or math.isnan(value):
                text = ""
            else:
                text = f"{value:.2f}" if key == "seconds" else f"{value:.4g}" if key == "l2" else f"{value:.4f}"
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if highlight:
                item.setBackground(BEST_BG)
                item.setToolTip("最佳 epoch：训练结束后使用这一轮的模型参数")
            self.history_table.setItem(row, col, item)
        if not highlight:
            self.history_table.scrollToBottom()

    def _show_evaluation(self, run: RunRecord | None) -> None:
        self.summary_table.setRowCount(0)
        self.point_table.setRowCount(0)
        self.open_run_btn.setEnabled(run is not None)
        if run is None:
            self.run_label.setText("训练完成后在这里显示最佳模型在各数据集上的表现")
            self.eval_chart.update_scores({})
            return
        m = run.metrics
        state = "（手动停止）" if m.get("stopped") else "（提前停止）" if m.get("early_stopped") else ""
        best = m.get("best_epoch", 0)
        used = f"使用第 {best} 个 epoch 的模型" if best else "没有完成任何 epoch，模型为初始参数"
        self.run_label.setText(
            f"{run.path}\n共训练 {m.get('epochs_run', len(run.history))} 个 epoch{state}，{used}，"
            f"{m.get('num_parameters', 0):,} 个参数，{len(m.get('features', []))} 个输入特征"
        )
        for name, values in m.get("splits", {}).items():
            row = self.summary_table.rowCount()
            self.summary_table.insertRow(row)
            cells = [SPLIT_NAMES.get(name, name), f"{values.get('n', 0):,}", _fmt(values.get("loss"), "{:.4f}")]
            cells += [_fmt(values.get(k), "{:.4f}" if k == "auc" else "{:.2%}") for k in METRIC_NAMES]
            for col, text in enumerate(cells):
                self.summary_table.setItem(row, col, QTableWidgetItem(text))
        self.eval_chart.update_scores(split_scores(run.predictions))

        table = per_point_accuracy(run.predictions)
        if not table.empty:
            shown = table[table["split"] != "train"]
            for rec in (shown if not shown.empty else table).itertuples(index=False):
                row = self.point_table.rowCount()
                self.point_table.insertRow(row)
                cells = [
                    SPLIT_NAMES.get(rec.split, rec.split),
                    rec.scene,
                    rec.point,
                    CLASS_NAMES.get(rec.label, rec.label),
                    f"{rec.samples:,}",
                    f"{rec.accuracy:.1%}",
                    f"{rec.mean_prob:.3f}",
                ]
                for col, text in enumerate(cells):
                    item = QTableWidgetItem(str(text))
                    if rec.accuracy < 0.8:
                        item.setBackground(WARN_BG)
                    self.point_table.setItem(row, col, item)
        self.statusBar().showMessage(f"训练结果：{run.path}")

    def _open_run_folder(self) -> None:
        if self._run is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._run.path)))

    # ------------------------------------------------------------------ 其他
    def _log(self, text: str) -> None:
        self.log_view.appendPlainText(text)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._worker is not None:
            self._worker.stop()
            self._cleanup_worker()
        self.sweep_panel.shutdown()
        self._save_state()
        super().closeEvent(event)


def _fmt(value, pattern: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return pattern.format(value)
