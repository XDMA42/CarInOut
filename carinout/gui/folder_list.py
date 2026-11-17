"""原始数据文件夹列表：可添加多个，支持从资源管理器拖入。"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QAbstractItemView, QListWidget, QListWidgetItem

from ..parser import natural_key

ROLE_PATH = Qt.ItemDataRole.UserRole


def csv_subfolders(parent: Path) -> list[Path]:
    """parent 下直接包含 csv 文件的子文件夹（不递归），按名称自然排序。"""
    subdirs = [d for d in parent.iterdir() if d.is_dir() and any(_csv_files(d))]
    return sorted(subdirs, key=lambda d: natural_key(d.name))


def _csv_files(folder: Path):
    return (f for f in folder.iterdir() if f.is_file() and f.suffix.lower() == ".csv")


class FolderList(QListWidget):
    changed = Signal()

    PLACEHOLDER = "点击下方按钮添加，或把文件夹拖到这里（可多个）"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

    def folders(self) -> list[Path]:
        return [Path(self.item(i).data(ROLE_PATH)) for i in range(self.count())]

    def add_folders(self, paths: list[Path]) -> int:
        """添加文件夹（忽略不存在的和已在列表中的），返回实际添加的个数。"""
        existing = {p.resolve() for p in self.folders()}
        added = 0
        for path in paths:
            path = Path(path)
            if not path.is_dir() or path.resolve() in existing:
                continue
            existing.add(path.resolve())
            item = QListWidgetItem(path.name or str(path))
            item.setData(ROLE_PATH, str(path))
            item.setToolTip(str(path))
            self.addItem(item)
            added += 1
        if added:
            self.changed.emit()
        return added

    def remove_selected(self) -> None:
        rows = sorted((self.row(item) for item in self.selectedItems()), reverse=True)
        for row in rows:
            self.takeItem(row)
        if rows:
            self.changed.emit()

    def clear_all(self) -> None:
        if self.count():
            self.clear()
            self.changed.emit()

    # ---- 拖入文件夹 ----
    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802
        if not event.mimeData().hasUrls():
            super().dropEvent(event)
            return
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        self.add_folders([p for p in paths if p.is_dir()])
        event.acceptProposedAction()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.remove_selected()
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        if self.count() == 0:
            p = QPainter(self.viewport())
            p.setPen(QColor(128, 128, 128))
            p.drawText(self.viewport().rect().adjusted(8, 0, -8, 0),
                       Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.PLACEHOLDER)
            p.end()
