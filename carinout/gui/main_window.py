"""主窗口：左侧数据文件树，中间车模，右侧 SLAVE→锚点 映射与导出设置，底部日志。"""

from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices
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
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..anchors import ANCHOR_SLOTS, SLOT_KEYS, AppConfig, default_names, validate_name
from ..merger import (
    COMBINED_FILENAME,
    JOIN_INNER,
    JOIN_OUTER,
    LABEL_INSIDE,
    LABEL_OUTSIDE,
    AnchorData,
    MergeOptions,
    PointResult,
    align_by_ts,
    build_dataset,
    dataset_folder,
    default_dataset_name,
    load_anchor,
    unsynced_anchors,
)
from ..parser import DEFAULT_PATTERN, group_by_point, natural_key, scan_folders, unique_source_names
from .car_view import CarView, Coverage
from .folder_list import FolderList, csv_subfolders

ROLE_POINT = Qt.ItemDataRole.UserRole
LABEL_TEXT = {LABEL_INSIDE: "1 车内", LABEL_OUTSIDE: "0 车外"}
FOLLOW_SCENE = "follow"  # 点位标签菜单中“跟随场景”选项的数据
LABEL_COLOR_OVERRIDE = QColor("#2563eb")
LABEL_COLOR_INHERIT = QColor(128, 128, 128)
WARN_BG = QColor(245, 158, 11, 60)
ERROR_BG = QColor(220, 38, 38, 70)


class ExportWorker(QObject):
    progress = Signal(int, int)
    message = Signal(str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(
        self, groups, slave_to_anchor, anchor_order, out_dir: Path, options: MergeOptions, scene_labels, point_labels
    ) -> None:
        super().__init__()
        self._args = (groups, slave_to_anchor, anchor_order, out_dir, options, scene_labels, point_labels)
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            results = build_dataset(
                *self._args,
                progress=self.progress.emit,
                log=self.message.emit,
                cancelled=lambda: self._cancel,
            )
        except Exception as exc:  # noqa: BLE001 —— 任何异常都要回报给界面，避免线程静默退出
            self.failed.emit(str(exc))
        else:
            self.done.emit(results)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("钥匙车内外识别 · 数据集构建")
        self.resize(1480, 900)

        self.config = AppConfig.load()
        # (来源, 场景, 点位) → {SLAVE: [文件]}；来源是原始数据文件夹的简称
        self.groups: dict[tuple[str, str, str], dict[str, list[Path]]] = {}
        self.slaves: list[str] = []
        self.current_point: tuple[str, str, str] | None = None
        # 当前预览点位已读取的数据：(来源, 场景, 点位), {SLAVE: AnchorData}
        self._point_cache: tuple[tuple[str, str, str], dict[str, AnchorData]] | None = None
        self._file_stats = (0, 0)  # (已识别, 未识别) 的 CSV 数
        self._thread: QThread | None = None
        self._worker: ExportWorker | None = None
        self._out_dir: Path | None = None
        self._input_dirs: list[Path] = []  # 当前已解析的原始数据文件夹
        self._sources: list[str] = []  # 与 _input_dirs 一一对应的来源简称
        self._combos: dict[str, QComboBox] = {}
        self._point_items: dict[tuple[str, str, str], QTreeWidgetItem] = {}  # (来源, 场景, 点位) → 树中的点位行

        self.car = CarView()
        self.car.set_names(self.config.anchor_names)
        self.car.anchorClicked.connect(self._on_anchor_clicked)
        self.car.anchorDoubleClicked.connect(self._rename_anchor)

        left = self._build_data_panel()
        right = self._build_mapping_panel()

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.log_view.setPlaceholderText("日志")

        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(left)
        top.addWidget(self.car)
        # 右侧内容较多，屏幕较矮时滚动显示，避免控件被压扁重叠
        right_scroll = QScrollArea()
        right_scroll.setWidget(right)
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # 和不滚动时一样，分隔条不能把右侧压得比内容还窄
        right_scroll.setMinimumWidth(right.minimumSizeHint().width() + right_scroll.verticalScrollBar().sizeHint().width())
        self._right_panel = right
        right.installEventFilter(self)  # 宽度变化时换行行数会变，重新计算所需高度
        top.addWidget(right_scroll)
        top.setStretchFactor(1, 1)
        top.setSizes([470, 640, 370])

        main = QSplitter(Qt.Orientation.Vertical)
        main.addWidget(top)
        main.addWidget(self.log_view)
        main.setStretchFactor(0, 1)
        main.setSizes([720, 160])
        container = QWidget()
        clay = QVBoxLayout(container)
        clay.setContentsMargins(8, 8, 8, 0)
        clay.addWidget(main)
        self.setCentralWidget(container)
        self.statusBar()

        self._refresh_mapping_view()
        self._update_car_title()
        self._update_target_label()
        if self.folder_list.add_folders([Path(d) for d in self.config.last_input_dirs]) == 0:
            self._update_stats()

    # ------------------------------------------------------------------ 界面搭建
    def _build_data_panel(self) -> QWidget:
        box = QGroupBox("数据文件")
        self.folder_list = FolderList()
        self.folder_list.setMaximumHeight(110)
        self.folder_list.changed.connect(lambda: self._parse_folders(auto=True))
        add_one = QPushButton("添加文件夹…")
        add_one.clicked.connect(self._add_folder)
        add_subs = QPushButton("添加子文件夹…")
        add_subs.setToolTip("选择一个上级文件夹，把其中所有含 CSV 的子文件夹一次加入")
        add_subs.clicked.connect(self._add_subfolders)
        remove = QPushButton("移除")
        remove.setToolTip("移除选中的文件夹（也可按 Delete 键）")
        remove.clicked.connect(self.folder_list.remove_selected)
        clear = QPushButton("清空")
        clear.clicked.connect(self.folder_list.clear_all)
        row = QHBoxLayout()
        for b in (add_one, add_subs, remove, clear):
            row.addWidget(b)

        self.pattern_edit = QLineEdit(self.config.filename_pattern)
        self.pattern_edit.setToolTip(
            "文件名正则，需包含命名分组 scene（场景）、point（点位）、anchor（锚点/SLAVE），不区分大小写。\n"
            "默认规则可解析 “中间座椅+脚坑-68-[COM261][SLAVE4] 2026-10-23 15-44-12.743.csv”\n"
            "（锚点取方括号内容，跳过 [COMxxx]）和 “中排座椅_53_SLAVE2.csv”。"
        )
        reset = QPushButton("默认")
        reset.setToolTip("恢复默认文件名规则")
        reset.clicked.connect(lambda: self.pattern_edit.setText(DEFAULT_PATTERN))
        parse = QPushButton("解析")
        parse.setDefault(True)
        parse.clicked.connect(self._parse_folders)
        self.pattern_edit.returnPressed.connect(self._parse_folders)
        prow = QHBoxLayout()
        prow.addWidget(self.pattern_edit, 1)
        prow.addWidget(reset)
        prow.addWidget(parse)

        self.stats_label = QLabel("尚未加载数据")
        self.stats_label.setWordWrap(True)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["场景 / 点位", "SLAVE", "文件", "标签"])
        self.tree.headerItem().setToolTip(
            3,
            "场景行：该场景所有点位的默认标签（1 = 车内，0 = 车外）。\n"
            "点位行：默认跟随场景（灰色）；点击可单独设置（蓝色加粗），多选后右键可批量设置。",
        )
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setStretchLastSection(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.currentItemChanged.connect(self._on_tree_selection)
        self.tree.itemClicked.connect(self._on_tree_clicked)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_tree_context_menu)
        tree_hint = QLabel("点位的标签默认跟随场景；点击点位的“标签”可单独修改，按住 Ctrl/Shift 多选后右键可批量修改")
        tree_hint.setWordWrap(True)
        tree_hint.setStyleSheet("color: gray")

        lay = QVBoxLayout(box)
        lay.addWidget(QLabel("原始数据文件夹（可多个，同名场景共用标签，导出为一个数据集）"))
        lay.addWidget(self.folder_list)
        lay.addLayout(row)
        lay.addWidget(QLabel("文件名规则（场景 / 点位 / 锚点）"))
        lay.addLayout(prow)
        lay.addWidget(self.stats_label)
        lay.addWidget(self.tree, 1)
        lay.addWidget(tree_hint)
        return box

    def _build_mapping_panel(self) -> QWidget:
        panel = QWidget()
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)

        mbox = QGroupBox("SLAVE → 锚点映射")
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["SLAVE", "文件数", "锚点"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.currentCellChanged.connect(self._on_table_row)

        auto = QPushButton("按编号分配")
        auto.setToolTip(
            "按 ID 末尾的编号对应车模上的锚点顺序（编号 1 → "
            + " → ".join(s.description for s in ANCHOR_SLOTS)
            + "），没有编号的 ID（如 MASTER_MCU）不分配"
        )
        auto.clicked.connect(self._auto_assign)
        clear = QPushButton("清空映射")
        clear.clicked.connect(self._clear_mapping)
        reset_names = QPushButton("恢复默认锚点名")
        reset_names.clicked.connect(self._reset_names)
        brow = QHBoxLayout()
        brow.addWidget(auto)
        brow.addWidget(clear)
        brow.addWidget(reset_names)

        self.mapping_hint = QLabel()
        self.mapping_hint.setWordWrap(True)

        mlay = QVBoxLayout(mbox)
        mlay.addWidget(self.table, 1)
        mlay.addLayout(brow)
        mlay.addWidget(self.mapping_hint)

        ebox = QGroupBox("合并导出")
        self.output_edit = QLineEdit(self.config.last_output_dir)
        self.output_edit.setPlaceholderText("默认：与原始数据文件夹同级的 merged")
        self.output_edit.textChanged.connect(self._update_target_label)
        obrowse = QPushButton("浏览…")
        obrowse.clicked.connect(self._choose_output)
        orow = QHBoxLayout()
        orow.addWidget(self.output_edit, 1)
        orow.addWidget(obrowse)
        self.name_edit = QLineEdit()
        self.name_edit.setToolTip("导出的文件夹名；留空时用原始数据文件夹名（多个时用 + 连接，超过 3 个取第一个加“等 N 个”）")
        self.name_edit.textChanged.connect(self._update_target_label)
        self.target_label = QLabel()
        self.target_label.setWordWrap(True)
        self.target_label.setStyleSheet("color: gray")

        self.join_combo = QComboBox()
        self.join_combo.addItem("只保留所有锚点都有数据的 ts", JOIN_INNER)
        self.join_combo.addItem("保留全部 ts（缺数据的锚点补空）", JOIN_OUTER)
        self.join_combo.setToolTip("各锚点筛选 errcode=0 后按 ts 列横向对齐，每行是同一 ts 下所有锚点的特征")
        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(0, 1e12)
        self.tolerance_spin.setDecimals(3)
        self.tolerance_spin.setSpecialValueText("0（精确匹配）")
        self.tolerance_spin.setToolTip(
            "0：ts 完全相同才对齐。\n"
            "大于 0：各锚点 ts 不完全同步时使用，以第一个锚点为基准，取相差不超过该值的最近一行；\n"
            "单位与 ts 列相同（例如 ts 是毫秒，填 50 表示 ±50 ms）。"
        )
        self.join_combo.currentIndexChanged.connect(self._refresh_point)
        self.tolerance_spin.valueChanged.connect(self._refresh_point)
        self.complete_check = QCheckBox("只导出已映射锚点齐全的点位")
        self.complete_check.setToolTip("勾选后，任一已映射锚点在该点位缺失文件、读取失败或没有 errcode=0 的行，整个点位都不导出")
        self.complete_check.toggled.connect(self._refresh_point)
        self.combined_check = QCheckBox(f"生成总表 {COMBINED_FILENAME}（追加 scene / point / label 等列）")
        self.combined_check.setChecked(True)
        self.combined_check.setToolTip(
            "所有有标签的点位合在一个文件里，末尾追加 source、scene、point、label（1 = 车内，0 = 车外）；\n"
            "点位的标签默认跟随场景，单独设置过的以点位为准。没有标签的点位不写入总表。"
        )
        self.encoding_combo = QComboBox()
        self.encoding_combo.addItem("UTF-8 with BOM（Excel 可直接打开）", "utf-8-sig")
        self.encoding_combo.addItem("UTF-8", "utf-8")
        self.encoding_combo.addItem("GB18030 / GBK", "gb18030")

        form = QFormLayout()
        form.addRow("ts 对齐", self.join_combo)
        form.addRow("ts 容差", self.tolerance_spin)
        form.addRow("输出编码", self.encoding_combo)

        self.export_btn = QPushButton("开始合并导出")
        self.export_btn.setMinimumHeight(34)
        self.export_btn.setToolTip("输出：场景_点位.csv，首列 ts，其余列名 = 原列名_锚点名，仅保留 errcode=0 的行")
        self.export_btn.clicked.connect(self._export_or_cancel)
        self.progress = QProgressBar()
        self.progress.setValue(0)

        elay = QVBoxLayout(ebox)
        elay.addWidget(QLabel("输出位置"))
        elay.addLayout(orow)
        nrow = QHBoxLayout()
        nrow.addWidget(QLabel("数据集名称"))
        nrow.addWidget(self.name_edit, 1)
        elay.addLayout(nrow)
        elay.addWidget(self.target_label)
        elay.addLayout(form)
        elay.addWidget(self.complete_check)
        elay.addWidget(self.combined_check)
        elay.addWidget(self.export_btn)
        elay.addWidget(self.progress)

        outer.addWidget(mbox, 1)
        outer.addWidget(ebox)
        return panel

    # ------------------------------------------------------------------ 数据加载
    def _last_folder(self) -> str:
        folders = self.folder_list.folders()
        return str(folders[-1]) if folders else ""

    def _add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "添加原始数据文件夹", self._last_folder())
        if folder and not self.folder_list.add_folders([Path(folder)]):
            self.statusBar().showMessage("该文件夹已在列表中", 5000)

    def _add_subfolders(self) -> None:
        start = str(Path(self._last_folder()).parent) if self._last_folder() else ""
        folder = QFileDialog.getExistingDirectory(self, "选择上级文件夹（加入其中所有含 CSV 的子文件夹）", start)
        if not folder:
            return
        subdirs = csv_subfolders(Path(folder))
        if not subdirs:
            QMessageBox.information(self, "提示", f"{folder}\n下没有直接包含 CSV 的子文件夹")
            return
        added = self.folder_list.add_folders(subdirs)
        self.statusBar().showMessage(f"找到 {len(subdirs)} 个子文件夹，新加入 {added} 个", 5000)

    def _choose_output(self) -> None:
        start = self.output_edit.text() or self._last_folder()
        folder = QFileDialog.getExistingDirectory(self, "选择输出文件夹", start)
        if folder:
            self.output_edit.setText(folder)

    def _target_dir(self) -> Path:
        """实际导出目录：输出位置 / 数据集名称（默认由原始数据文件夹名组成）。"""
        out_text = self.output_edit.text().strip()
        return dataset_folder(self._input_dirs, Path(out_text) if out_text else None, self.name_edit.text())

    def _update_target_label(self) -> None:
        self.name_edit.setPlaceholderText(f"默认：{default_dataset_name(self._sources)}" if self._sources else "")
        if not self._input_dirs:
            self.target_label.setText("解析数据后显示导出路径")
            self._fit_right_panel()
            return
        target = str(self._target_dir())
        # 在路径分隔符后插入零宽空格，长路径可以换行显示
        self.target_label.setText("将导出到：" + target.replace("/", "/\u200b").replace("\\", "\\\u200b"))
        tip = target
        if len(self._sources) > 1:
            tip += "\n总表 _dataset.csv 和 _summary.csv 各一个；单点位文件按原始文件夹放在同名子文件夹中"
        self.target_label.setToolTip(tip)
        self._fit_right_panel()

    def _fit_right_panel(self) -> None:
        """右侧面板最矮不低于内容所需高度，放不下时由外层滚动区域滚动，而不是把控件压扁。"""
        panel = getattr(self, "_right_panel", None)
        if panel is None:
            return
        layout = panel.layout()
        height = layout.sizeHint().height()
        if layout.hasHeightForWidth():  # 自动换行的标签：按当前宽度算实际需要的行数
            height = max(height, layout.totalHeightForWidth(panel.width()))
        if height != panel.minimumHeight():
            panel.setMinimumHeight(height)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if obj is getattr(self, "_right_panel", None) and event.type() == QEvent.Type.Resize:
            self._fit_right_panel()
        return super().eventFilter(obj, event)

    def _parse_folders(self, auto: bool = False) -> None:
        """解析列表中的所有原始数据文件夹；auto 为 True 表示由列表变化触发，空列表时不弹提示。"""
        folders = [f for f in self.folder_list.folders() if f.is_dir()]
        if not folders and not auto:
            QMessageBox.warning(self, "提示", "请先添加原始数据文件夹")
            return
        pattern = self.pattern_edit.text().strip() or DEFAULT_PATTERN
        try:
            parsed, unmatched = scan_folders(folders, pattern)
        except ValueError as exc:
            QMessageBox.warning(self, "文件名规则错误", str(exc))
            return
        except OSError as exc:
            QMessageBox.warning(self, "无法读取文件夹", str(exc))
            return

        self.config.filename_pattern = pattern
        self.config.last_input_dirs = [str(f) for f in folders]
        self._save_config()
        self._input_dirs = folders
        self._sources = unique_source_names(folders)
        self._update_target_label()

        self.groups = group_by_point(parsed)
        self.slaves = sorted({f.slave for f in parsed}, key=natural_key)
        self._point_cache = None
        self.current_point = None

        if folders:
            self._log(f"解析 {len(folders)} 个文件夹：" + "、".join(str(f) for f in folders))
            self._log(f"  识别 {len(parsed)} 个 CSV，未识别 {len(unmatched)} 个")
        for path in unmatched[:20]:
            self._log(f"  未识别：{path.name}")
        if len(unmatched) > 20:
            self._log(f"  …… 另有 {len(unmatched) - 20} 个未识别文件")
        self._log_duplicates()

        self._file_stats = (len(parsed), len(unmatched))
        self._update_stats()
        if len(self.slaves) > len(ANCHOR_SLOTS):
            self._log(f"  注意：检测到 {len(self.slaves)} 个 SLAVE，多于车模上的 {len(ANCHOR_SLOTS)} 个锚点")

        self._populate_tree()
        self._populate_table()
        self._refresh_mapping_view()
        self.car.set_coverage(None)
        self._update_car_title()

    def _point_tag(self, key: tuple[str, str, str]) -> str:
        source, scene, point = key
        return f"{source}/{scene}_{point}" if len(self._sources) > 1 else f"{scene}_{point}"

    def _log_duplicates(self) -> None:
        for key, slave_files in self.groups.items():
            for slave, paths in slave_files.items():
                if len(paths) > 1:
                    names = "、".join(p.name for p in paths)
                    self._log(f"  [{self._point_tag(key)}] {slave} 有 {len(paths)} 个文件，将按顺序纵向拼接：{names}")

    def _populate_tree(self) -> None:
        self.tree.clear()
        self._point_items.clear()
        total = len(self.slaves)
        multi = len(self._sources) > 1
        scene_items: dict[str, QTreeWidgetItem] = {}
        source_items: dict[tuple[str, str], QTreeWidgetItem] = {}  # 多个来源时：场景 → 来源 → 点位
        n_points: dict[QTreeWidgetItem, int] = {}
        # 按场景归组，同名场景的不同来源排在一起，标签只需在场景行标一次
        for key in sorted(self.groups, key=lambda k: (natural_key(k[1]), natural_key(k[0]), natural_key(k[2]))):
            source, scene, point = key
            slave_files = self.groups[key]
            parent = scene_items.get(scene)
            if parent is None:
                parent = QTreeWidgetItem([scene, "", ""])
                font = parent.font(0)
                font.setBold(True)
                parent.setFont(0, font)
                scene_items[scene] = parent
                self.tree.addTopLevelItem(parent)
            container = parent
            if multi:
                container = source_items.get((scene, source))
                if container is None:
                    container = QTreeWidgetItem([source, "", ""])
                    container.setForeground(0, QColor(110, 110, 110))
                    source_items[(scene, source)] = container
                    parent.addChild(container)
                n_points[container] = n_points.get(container, 0) + 1
            n_points[parent] = n_points.get(parent, 0) + 1
            n_files = sum(len(v) for v in slave_files.values())
            item = QTreeWidgetItem([point, f"{len(slave_files)}/{total}", str(n_files)])
            item.setData(0, ROLE_POINT, key)
            self._point_items[key] = item
            item.setTextAlignment(1, Qt.AlignmentFlag.AlignCenter)
            item.setTextAlignment(2, Qt.AlignmentFlag.AlignCenter)
            missing = [s for s in self.slaves if s not in slave_files]
            if missing:
                for col in range(3):
                    item.setBackground(col, WARN_BG)
                item.setToolTip(0, "缺少：" + "、".join(missing))
            container.addChild(item)
        for item in (*scene_items.values(), *source_items.values()):
            item.setText(2, f"{n_points[item]} 个点位")
        # 场景标签下拉框只放在场景行；点位行显示实际标签，点击单独修改
        for scene, item in scene_items.items():
            combo = QComboBox()
            combo.addItem("未标注", None)
            for label, text in LABEL_TEXT.items():
                combo.addItem(text, label)
            combo.setCurrentIndex(max(combo.findData(self.config.scene_labels.get(scene)), 0))
            combo.currentIndexChanged.connect(lambda _i, sc=scene, c=combo: self._on_label_changed(sc, c))
            self.tree.setItemWidget(item, 3, combo)
        self._refresh_point_label_cells()
        self.tree.expandAll()
        for col in (1, 2, 3):
            self.tree.resizeColumnToContents(col)

    def _refresh_point_label_cells(self) -> None:
        """点位行“标签”格：跟随场景时灰色，单独设置时蓝色加粗。"""
        for (_source, scene, point), item in self._point_items.items():
            label = self.config.label_for(scene, point)
            overridden = self.config.is_point_overridden(scene, point)
            item.setText(3, LABEL_TEXT.get(label, "未标注") + " ▾")
            font = item.font(3)
            font.setBold(overridden)
            item.setFont(3, font)
            item.setForeground(3, LABEL_COLOR_OVERRIDE if overridden else LABEL_COLOR_INHERIT)
            scene_text = LABEL_TEXT.get(self.config.scene_labels.get(scene), "未标注")
            item.setToolTip(
                3,
                f"单独设置（场景标签：{scene_text}）" if overridden else f"跟随场景（{scene_text}），点击可单独设置",
            )

    def _point_keys(self, items) -> list[tuple[str, str, str]]:
        return [tuple(k) for item in items if (k := item.data(0, ROLE_POINT))]

    def _on_tree_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        keys = self._point_keys([item])
        if column == 3 and keys:
            rect = self.tree.visualItemRect(item)
            pos = self.tree.viewport().mapToGlobal(rect.bottomLeft())
            pos.setX(self.tree.viewport().mapToGlobal(self.tree.viewport().rect().topLeft()).x()
                     + self.tree.header().sectionViewportPosition(3))
            self._show_point_label_menu(keys, pos)

    def _on_tree_context_menu(self, pos) -> None:
        item = self.tree.itemAt(pos)
        keys = self._point_keys(self.tree.selectedItems())
        if item is not None and item.data(0, ROLE_POINT) and tuple(item.data(0, ROLE_POINT)) not in keys:
            keys = self._point_keys([item])  # 右键的点位不在选中范围内时只改它
        if keys:
            self._show_point_label_menu(keys, self.tree.viewport().mapToGlobal(pos))

    def _show_point_label_menu(self, keys: list[tuple[str, str, str]], global_pos) -> None:
        pairs = sorted({(scene, point) for _source, scene, point in keys}, key=lambda p: (natural_key(p[0]), natural_key(p[1])))
        scenes = {scene for scene, _ in pairs}
        menu = QMenu(self)
        menu.addSection(f"{len(pairs)} 个点位" if len(pairs) > 1 else f"{pairs[0][0]} / 点位 {pairs[0][1]}")
        if len(scenes) == 1:
            follow_text = f"跟随场景（{LABEL_TEXT.get(self.config.scene_labels.get(next(iter(scenes))), '未标注')}）"
        else:
            follow_text = "跟随各自场景"
        current = None
        if len(pairs) == 1:
            scene, point = pairs[0]
            current = (self.config.point_labels.get(scene, {}).get(point)
                       if self.config.is_point_overridden(scene, point) else FOLLOW_SCENE)
        for text, data in ((follow_text, FOLLOW_SCENE), (LABEL_TEXT[LABEL_INSIDE], LABEL_INSIDE),
                           (LABEL_TEXT[LABEL_OUTSIDE], LABEL_OUTSIDE)):
            action = menu.addAction(text)
            action.setData(data)
            action.setCheckable(current is not None)
            action.setChecked(current == data)
        chosen = self._exec_menu(menu, global_pos)
        if chosen is not None:
            data = chosen.data()
            self._set_point_labels(pairs, None if data == FOLLOW_SCENE else data)

    def _exec_menu(self, menu: QMenu, global_pos):
        """弹出菜单并返回选中的动作（单独成方法，便于测试替换）。"""
        return menu.exec(global_pos)

    def _set_point_labels(self, pairs: list[tuple[str, str]], label: int | None) -> None:
        """单独设置若干 (场景, 点位) 的标签；label 为 None 表示恢复跟随场景。"""
        for scene, point in pairs:
            self.config.set_point_label(scene, point, label)
        what = "跟随场景" if label is None else LABEL_TEXT[label]
        shown = "、".join(f"{scene}/{point}" for scene, point in pairs[:5]) + (" 等" if len(pairs) > 5 else "")
        self._log(f"点位标签 → {what}：{shown}（共 {len(pairs)} 个）")
        scenes = {scene for scene, _ in pairs}
        self._after_labels_changed(next(iter(scenes)) if len(scenes) == 1 else None)

    def _after_labels_changed(self, scene: str | None = None) -> None:
        """场景或点位标签改变后：保存、刷新点位标签与统计；scene 为空表示可能影响多个场景。"""
        self._save_config()
        self._refresh_point_label_cells()
        self._update_stats()
        if self.current_point and (scene is None or self.current_point[1] == scene):
            self._show_point(self.current_point)

    def _scenes(self) -> list[str]:
        return sorted({scene for _, scene, _ in self.groups}, key=natural_key)

    def _on_label_changed(self, scene: str, combo: QComboBox) -> None:
        label = combo.currentData()
        if label is None:
            self.config.scene_labels.pop(scene, None)
        else:
            self.config.scene_labels[scene] = label
        self._after_labels_changed(scene)

    def _update_stats(self) -> None:
        if not self._input_dirs:
            self.stats_label.setText("尚未加载数据")
            return
        n_parsed, n_unmatched = self._file_stats
        scenes = self._scenes()
        inside = sum(self.config.scene_labels.get(s) == LABEL_INSIDE for s in scenes)
        outside = sum(self.config.scene_labels.get(s) == LABEL_OUTSIDE for s in scenes)
        unlabeled = len(scenes) - inside - outside
        text = (
            f"文件夹 {len(self._input_dirs)} 个，CSV {n_parsed + n_unmatched} 个：已识别 {n_parsed}，未识别 {n_unmatched}\n"
            f"场景 {len(scenes)} 个，点位 {len(self.groups)} 个，SLAVE {len(self.slaves)} 个\n"
            f"标签：车内 {inside}，车外 {outside}"
        )
        if unlabeled:
            text += f"，<span style='color:#b45309'>未标注 {unlabeled}</span>"
        pairs = {(scene, point) for _, scene, point in self.groups}
        overridden = sum(self.config.is_point_overridden(scene, point) for scene, point in pairs)
        if overridden:
            text += f"；单独设置 {overridden} 个点位"
        self.stats_label.setText(text.replace("\n", "<br>"))

    def _populate_table(self) -> None:
        counts = {s: 0 for s in self.slaves}
        for slave_files in self.groups.values():
            for slave, paths in slave_files.items():
                counts[slave] += len(paths)

        self.table.setRowCount(0)  # 先清空：替换下来的旧下拉框会立即隐藏，不会残留在表格上
        self.table.setRowCount(len(self.slaves))
        self._combos.clear()
        for row, slave in enumerate(self.slaves):
            item = QTableWidgetItem(slave)
            self.table.setItem(row, 0, item)
            cnt = QTableWidgetItem(str(counts[slave]))
            cnt.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 1, cnt)
            combo = QComboBox()
            combo.addItem("— 未分配（忽略）—", "")
            for slot in ANCHOR_SLOTS:
                combo.addItem("", slot.key)
            current = self.config.slave_mapping.get(slave, "")
            combo.setCurrentIndex(max(combo.findData(current), 0))
            combo.currentIndexChanged.connect(lambda _i, s=slave, c=combo: self._on_combo_changed(s, c))
            self.table.setCellWidget(row, 2, combo)
            self._combos[slave] = combo
        self._refresh_combo_texts()

    def _refresh_combo_texts(self) -> None:
        for combo in self._combos.values():
            for i, slot in enumerate(ANCHOR_SLOTS, start=1):
                combo.setItemText(i, f"{self.config.anchor_names[slot.key]}（{slot.description}）")

    # ------------------------------------------------------------------ 映射
    def _on_combo_changed(self, slave: str, combo: QComboBox) -> None:
        key = combo.currentData()
        if key:
            self.config.slave_mapping[slave] = key
        else:
            self.config.slave_mapping.pop(slave, None)
        self._save_config()
        self._refresh_mapping_view()
        self.car.set_selected(key or None)
        if self.current_point:
            self._show_point(self.current_point)

    def _slaves_by_key(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for slave in self.slaves:
            key = self.config.slave_mapping.get(slave)
            if key:
                result.setdefault(key, []).append(slave)
        return result

    def _refresh_mapping_view(self) -> None:
        by_key = self._slaves_by_key()
        self.car.set_assignment(by_key)
        conflicts = {s for slaves in by_key.values() if len(slaves) > 1 for s in slaves}
        unassigned = [s for s in self.slaves if not self.config.slave_mapping.get(s)]
        for row, slave in enumerate(self.slaves):
            bg = ERROR_BG if slave in conflicts else WARN_BG if slave in unassigned else QColor(0, 0, 0, 0)
            for col in (0, 1):
                self.table.item(row, col).setBackground(bg)

        parts = []
        if conflicts:
            parts.append(f"<span style='color:#dc2626'>冲突：{'、'.join(sorted(conflicts, key=natural_key))} 映射到了同一锚点</span>")
        if unassigned:
            parts.append(f"<span style='color:#b45309'>未分配（导出时忽略）：{'、'.join(unassigned)}</span>")
        if not self.slaves:
            parts.append("加载数据后在此为每个 SLAVE 选择锚点")
        elif not parts:
            parts.append("<span style='color:#16a34a'>全部 SLAVE 已分配</span>")
        parts.append("<span style='color:gray'>提示：双击车模上的锚点可修改名称（名称即列名后缀）</span>")
        self.mapping_hint.setText("<br>".join(parts))

    def _auto_assign(self) -> None:
        if not self.slaves:
            return
        if any(self.config.slave_mapping.get(s) for s in self.slaves):
            ok = QMessageBox.question(self, "按编号分配", "将覆盖当前的映射，是否继续？")
            if ok != QMessageBox.StandardButton.Yes:
                return
        taken: set[int] = set()
        for slave in self.slaves:
            m = re.search(r"(\d+)$", slave)
            n = int(m.group(1)) if m else 0
            if 1 <= n <= len(SLOT_KEYS) and n not in taken:
                self.config.slave_mapping[slave] = SLOT_KEYS[n - 1]
                taken.add(n)
            else:
                self.config.slave_mapping.pop(slave, None)
        self._sync_combos()

    def _clear_mapping(self) -> None:
        for slave in self.slaves:
            self.config.slave_mapping.pop(slave, None)
        self._sync_combos()

    def _sync_combos(self) -> None:
        for slave, combo in self._combos.items():
            combo.blockSignals(True)
            combo.setCurrentIndex(max(combo.findData(self.config.slave_mapping.get(slave, "")), 0))
            combo.blockSignals(False)
        self._save_config()
        self._refresh_mapping_view()
        if self.current_point:
            self._show_point(self.current_point)

    def _rename_anchor(self, key: str) -> None:
        slot = next(s for s in ANCHOR_SLOTS if s.key == key)
        names = self.config.anchor_names
        text, ok = QInputDialog.getText(
            self, "重命名锚点", f"{slot.description}的名称（用作列名后缀）：", QLineEdit.EchoMode.Normal, names[key]
        )
        if not ok or text.strip() == names[key]:
            return
        error = validate_name(text, names, key)
        if error:
            QMessageBox.warning(self, "名称无效", error)
            return
        self._log(f"锚点 {slot.description} 改名：{names[key]} → {text.strip()}")
        names[key] = text.strip()
        self._apply_names()

    def _reset_names(self) -> None:
        self.config.anchor_names = default_names()
        self._apply_names()

    def _apply_names(self) -> None:
        self._save_config()
        self.car.set_names(self.config.anchor_names)
        self._refresh_combo_texts()

    def _on_anchor_clicked(self, key: str) -> None:
        self.car.set_selected(key)
        slaves = self._slaves_by_key().get(key, [])
        name = self.config.anchor_names[key]
        if slaves:
            row = self.slaves.index(slaves[0])
            self.table.selectRow(row)
            self.table.scrollToItem(self.table.item(row, 0))
            self.statusBar().showMessage(f"锚点 {name} ← {' / '.join(slaves)}", 5000)
        else:
            self.table.clearSelection()
            self.statusBar().showMessage(f"锚点 {name} 尚未分配 SLAVE，请在右侧列表中选择", 5000)

    def _on_table_row(self, row: int, *_args) -> None:
        if 0 <= row < len(self.slaves):
            self.car.set_selected(self.config.slave_mapping.get(self.slaves[row]) or None)

    # ------------------------------------------------------------------ 点位预览
    def _on_tree_selection(self, item: QTreeWidgetItem | None, _prev) -> None:
        point = item.data(0, ROLE_POINT) if item else None
        self.current_point = tuple(point) if point else None
        if self.current_point:
            self._show_point(self.current_point)
        else:
            self.car.set_coverage(None)
            self._update_car_title(item.text(0) if item else "")

    def _refresh_point(self) -> None:
        if self.current_point:
            self._show_point(self.current_point)

    def _show_point(self, point: tuple[str, str, str]) -> None:
        if self._point_cache is None or self._point_cache[0] != point:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                loaded = {slave: load_anchor(paths) for slave, paths in self.groups[point].items()}
            finally:
                QApplication.restoreOverrideCursor()
            self._point_cache = (point, loaded)
        loaded = self._point_cache[1]

        coverage: dict[str, Coverage] = {}
        frames = []
        for key in SLOT_KEYS:  # 与导出时的锚点顺序一致
            slaves = [s for s in loaded if self.config.slave_mapping.get(s) == key]
            if len(slaves) != 1:
                continue
            data = loaded[slaves[0]]
            if data.error:
                coverage[key] = Coverage(False, "读取失败")
            else:
                coverage[key] = Coverage(data.valid_rows > 0, f"有效 {data.valid_rows}/{data.total_rows} 行")
                if data.valid_rows > 0:  # 与导出一致：没有有效数据的锚点不参与对齐
                    frames.append((self.config.anchor_names[key], data.valid))
        self.car.set_coverage(coverage)

        source, scene, pt = point
        title = [f"{scene} / 点位 {pt}"]
        if len(self._sources) > 1:
            title.append(source)
        label_text = LABEL_TEXT.get(self.config.label_for(scene, pt), "未标注")
        title.append(label_text + ("（单独设置）" if self.config.is_point_overridden(scene, pt) else ""))
        if frames:
            try:
                tolerance = self.tolerance_spin.value()
                rows = len(align_by_ts(frames, self.join_combo.currentData(), tolerance))
                title.append(f"ts 对齐后 {rows} 行")
                unsynced = unsynced_anchors(frames) if tolerance == 0 else []
                if unsynced:
                    title.append(f"⚠ {','.join(unsynced)} 的 ts 对不上")
            except ValueError as exc:
                title.append(f"ts 对齐失败：{exc}")
        expected = {self.config.slave_mapping.get(s) for s in self.slaves} - {None}
        lacking = [
            self.config.anchor_names[k]
            for k in SLOT_KEYS
            if k in expected and not (k in coverage and coverage[k].ok)
        ]
        if lacking and self.complete_check.isChecked():
            title.append(f"⚠ 导出时会跳过：{','.join(lacking)} 缺失或无有效数据")
        self._update_car_title("  ·  ".join(title))

        lines = [f"[{self._point_tag(point)}]"]
        for slave, data in loaded.items():
            key = self.config.slave_mapping.get(slave)
            target = self.config.anchor_names[key] if key else "未分配"
            lines.append(
                f"{slave}→{target}: "
                + (f"读取失败（{data.error}）" if data.error else f"{data.valid_rows}/{data.total_rows}")
            )
        self.statusBar().showMessage("  ".join(lines))

    def _update_car_title(self, text: str = "") -> None:
        self.car.set_title(text or "SLAVE ↔ 锚点映射总览")

    # ------------------------------------------------------------------ 导出
    def _export_or_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.export_btn.setEnabled(False)
            return
        self._start_export()

    def _start_export(self) -> None:
        if not self.groups:
            QMessageBox.warning(self, "提示", "没有可合并的数据，请先选择文件夹并解析")
            return
        by_key = self._slaves_by_key()
        conflicts = {k: v for k, v in by_key.items() if len(v) > 1}
        if conflicts:
            detail = "\n".join(f"{self.config.anchor_names[k]}：{'、'.join(v)}" for k, v in conflicts.items())
            QMessageBox.warning(self, "映射冲突", "以下锚点被多个 SLAVE 选中，请先修正：\n" + detail)
            return
        if not by_key:
            QMessageBox.warning(self, "提示", "还没有为任何 SLAVE 分配锚点")
            return
        unassigned = [s for s in self.slaves if not self.config.slave_mapping.get(s)]
        if unassigned:
            ok = QMessageBox.question(
                self, "存在未分配的 SLAVE", f"{'、'.join(unassigned)} 未分配锚点，其数据将被忽略。是否继续？"
            )
            if ok != QMessageBox.StandardButton.Yes:
                return

        scene_labels = point_labels = None
        if self.combined_check.isChecked():
            scenes = self._scenes()
            pairs = sorted({(scene, point) for _, scene, point in self.groups})
            scene_labels = {sc: self.config.scene_labels[sc] for sc in scenes if sc in self.config.scene_labels}
            point_labels = {
                (scene, point): self.config.point_labels[scene][point]
                for scene, point in pairs
                if self.config.is_point_overridden(scene, point)
            }
            unlabeled: dict[str, int] = {}
            for scene, point in pairs:
                if self.config.label_for(scene, point) is None:
                    unlabeled[scene] = unlabeled.get(scene, 0) + 1
            if len(unlabeled) == len(scenes) and sum(unlabeled.values()) == len(pairs):
                QMessageBox.warning(self, "提示", "生成总表需要先在左侧为场景或点位选择标签（1 车内 / 0 车外）")
                return
            if unlabeled:
                detail = "、".join(f"{sc}（{n} 个点位）" for sc, n in unlabeled.items())
                ok = QMessageBox.question(
                    self,
                    "存在未标注的点位",
                    f"{detail} 没有标签，不会写入总表（单点位文件照常导出）。\n\n是否继续？",
                )
                if ok != QMessageBox.StandardButton.Yes:
                    return

        out_dir = self._target_dir()
        if out_dir.is_dir() and any(out_dir.iterdir()):
            ok = QMessageBox.question(
                self,
                "输出文件夹已存在",
                f"{out_dir}\n已存在且不为空，其中的同名文件会被覆盖，其他文件保留。是否继续？",
            )
            if ok != QMessageBox.StandardButton.Yes:
                return
        self.config.last_output_dir = self.output_edit.text().strip()
        self._save_config()

        names = self.config.anchor_names
        slave_to_anchor = {s: names[k] for s, k in self.config.slave_mapping.items() if s in self.slaves}
        anchor_order = [names[k] for k in SLOT_KEYS]
        options = MergeOptions(
            join=self.join_combo.currentData(),
            ts_tolerance=self.tolerance_spin.value(),
            require_complete=self.complete_check.isChecked(),
            output_encoding=self.encoding_combo.currentData(),
        )
        self._log(
            f"开始导出到 {out_dir}，映射："
            + "，".join(f"{s}→{slave_to_anchor[s]}" for s in self.slaves if s in slave_to_anchor)
        )

        self.progress.setRange(0, len(self.groups))
        self.progress.setValue(0)
        self.export_btn.setText("取消")

        # 信号都连接到主窗口的方法，保证回调在界面线程执行
        self._out_dir = out_dir
        self._thread = QThread(self)
        self._worker = ExportWorker(
            self.groups, slave_to_anchor, anchor_order, out_dir, options, scene_labels, point_labels
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._export_progress)
        self._worker.message.connect(self._log)
        self._worker.done.connect(self._export_finished)
        self._worker.failed.connect(self._export_failed)
        self._thread.start()

    def _export_progress(self, done: int, _total: int) -> None:
        self.progress.setValue(done)

    def _cleanup_worker(self) -> None:
        if self._thread:
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None
        self.export_btn.setText("开始合并导出")
        self.export_btn.setEnabled(True)

    def _export_finished(self, results: list[PointResult]) -> None:
        self._cleanup_worker()
        out_dir = self._out_dir
        exported = [r for r in results if r.output]
        skipped = len(results) - len(exported)
        rows = sum(r.rows for r in exported)
        msg = f"完成：导出 {len(exported)} 个点位，共 {rows} 行；跳过 {skipped} 个。汇总见 _summary.csv"
        if skipped:
            msg += "（跳过原因见日志或 _summary.csv 的“状态”列）"
        if self.combined_check.isChecked():
            msg += f"，总表见 {COMBINED_FILENAME}"
        if len(results) < len(self.groups):
            msg = f"已取消：处理了 {len(results)}/{len(self.groups)} 个点位，导出 {len(exported)} 个"
        self._log(msg)
        box = QMessageBox(QMessageBox.Icon.Information, "合并导出", msg, parent=self)
        open_btn = box.addButton("打开输出文件夹", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Close)
        box.exec()
        if box.clickedButton() is open_btn:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(out_dir)))

    def _export_failed(self, error: str) -> None:
        self._cleanup_worker()
        self._log(f"导出失败：{error}")
        QMessageBox.critical(self, "导出失败", error)

    # ------------------------------------------------------------------ 其他
    def _log(self, text: str) -> None:
        self.log_view.appendPlainText(text)

    def _save_config(self) -> None:
        try:
            self.config.save()
        except OSError as exc:
            self.statusBar().showMessage(f"配置保存失败：{exc}", 5000)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._worker is not None:
            self._worker.cancel()
            self._cleanup_worker()
        self._save_config()
        super().closeEvent(event)
