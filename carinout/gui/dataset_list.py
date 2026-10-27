"""可放入多个文件夹的数据集列表，独立验证集、独立测试集共用。

每项是含 _dataset.csv 的文件夹（或 csv 文件）。选择上级文件夹（如数据集构建工具的 merged）时，
添加其中所有含 _dataset.csv 的子文件夹。也可以从资源管理器把一个或多个文件夹拖进列表。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..merger import COMBINED_FILENAME
from ..training.data import dataset_name, resolve_datasets

MISSING_FG = QColor(208, 59, 59)


class _DropList(QListWidget):
    dropped = Signal(list)

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            event.acceptProposedAction()
            self.dropped.emit(paths)


class DatasetList(QWidget):
    changed = Signal()

    def __init__(
        self,
        empty_text: str,
        filled_text: str,
        start_dir: Callable[[], str],
        exclude: Callable[[], dict[str, str]],
        parent=None,
    ) -> None:
        """empty_text / filled_text：列表为空 / 非空时的说明（filled_text 中的 {n} 为文件夹数）。
        exclude 返回不能添加的数据 {路径: 原因}，如训练数据、已在另一个列表中的文件夹。"""
        super().__init__(parent)
        self._empty_text = empty_text
        self._filled_text = filled_text
        self._start_dir = start_dir
        self._exclude = exclude

        self.list = _DropList()
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        # 路径过长时省略中间部分，保留开头的名称和路径末尾
        self.list.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setMaximumHeight(96)
        self.list.setToolTip(f"每项是一个含 {COMBINED_FILENAME} 的文件夹，可以从资源管理器拖入一个或多个文件夹")
        self.list.dropped.connect(self.add_paths)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color: gray")

        add = QPushButton("添加文件夹…")
        add.setToolTip(
            f"选择含 {COMBINED_FILENAME} 的文件夹；\n"
            f"选择上级文件夹（如 merged）时，添加其中所有含 {COMBINED_FILENAME} 的子文件夹"
        )
        add.clicked.connect(self._choose)
        remove = QPushButton("移除")
        remove.clicked.connect(self._remove_selected)
        clear = QPushButton("清空")
        clear.clicked.connect(lambda: self.set_paths([], notify=True))
        buttons = QHBoxLayout()
        buttons.addWidget(add)
        buttons.addWidget(remove)
        buttons.addWidget(clear)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.list)
        lay.addLayout(buttons)
        lay.addWidget(self.summary)
        self._refresh()

    def paths(self) -> list[str]:
        return [self.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.list.count())]

    def set_paths(self, paths: list[str], notify: bool = False) -> None:
        """显示配置中的路径（不检查；找不到的项标红，开始训练时会报错）。"""
        self.list.clear()
        for path in paths:
            self._append(path)
        self._refresh()
        if notify:
            self.changed.emit()

    def add_paths(self, paths: list[str]) -> None:
        """添加文件夹或 csv；上级文件夹展开为其中含 _dataset.csv 的各子文件夹，已有的不重复添加。"""
        existing = {f.resolve() for f in _resolve_quietly(self.paths())}
        excluded = {}
        for path, reason in self._exclude().items():
            excluded.update({f.resolve(): reason for f in _resolve_quietly([path])})
        errors, skipped = [], []
        for path in paths:
            try:
                files = resolve_datasets([path])
            except ValueError as exc:
                errors.append(str(exc))
                continue
            for file in files:
                key = file.resolve()
                if key in excluded:
                    skipped.append(f"{dataset_name(file)}（{excluded[key]}）")
                elif key not in existing:
                    existing.add(key)
                    self._append(str(file.parent if file.name == COMBINED_FILENAME else file))
        self._refresh()
        self.changed.emit()
        if skipped:
            errors.append("已跳过：" + "、".join(skipped))
        if errors:
            QMessageBox.warning(self, "部分文件夹未添加", "\n".join(errors))

    def _append(self, entry: str) -> None:
        item = QListWidgetItem(entry)
        item.setData(Qt.ItemDataRole.UserRole, entry)
        try:
            files = resolve_datasets([entry])
        except ValueError as exc:
            item.setForeground(MISSING_FG)
            item.setToolTip(str(exc))
        else:
            item.setToolTip("\n".join(str(f) for f in files))
            item.setText(f"{dataset_name(files[0])}    {entry}" if len(files) == 1 else entry)
        self.list.addItem(item)

    def _choose(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择数据集文件夹（或其上级文件夹）", self._start_dir())
        if folder:
            self.add_paths([folder])

    def _remove_selected(self) -> None:
        for item in self.list.selectedItems():
            self.list.takeItem(self.list.row(item))
        self._refresh()
        self.changed.emit()

    def _refresh(self) -> None:
        n = self.list.count()
        self.summary.setText(self._filled_text.format(n=n) if n else self._empty_text)


def _resolve_quietly(paths: list[str]) -> list[Path]:
    """展开路径，忽略找不到的项。"""
    files = []
    for path in paths:
        try:
            files.extend(resolve_datasets([path]))
        except ValueError:
            pass
    return files
