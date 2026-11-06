"""小场景类别管理：新建、重命名、删除。"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..anchors import AppConfig


def ask_new_sub_category(parent: QWidget, config: AppConfig) -> int | None:
    """弹窗输入名称并新建类别，返回编号；取消或名称无效时返回 None。"""
    name, ok = QInputDialog.getText(parent, "新建小场景类别", "类别名称：")
    if not ok:
        return None
    error = config.validate_sub_category_name(name)
    if error:
        QMessageBox.warning(parent, "名称无效", error)
        return None
    return config.add_sub_category(name)


class SubCategoryDialog(QDialog):
    """直接修改传入的 config；关闭后由调用方保存并刷新界面。"""

    def __init__(self, config: AppConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("小场景类别")
        self.resize(360, 380)
        self.config = config

        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(lambda _item: self._rename())
        add = QPushButton("新建…")
        add.clicked.connect(self._add)
        rename = QPushButton("重命名…")
        rename.clicked.connect(self._rename)
        remove = QPushButton("删除")
        remove.clicked.connect(self._remove)
        buttons = QVBoxLayout()
        for b in (add, rename, remove):
            buttons.addWidget(b)
        buttons.addStretch(1)

        row = QHBoxLayout()
        row.addWidget(self.list, 1)
        row.addLayout(buttons)

        hint = QLabel("编号写入总表的 sub_label 列，创建后固定不变；删除的编号不会再分配给新类别。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray")
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.accept)

        lay = QVBoxLayout(self)
        lay.addLayout(row, 1)
        lay.addWidget(hint)
        lay.addWidget(close)
        self._refresh()

    def _refresh(self, select: int | None = None) -> None:
        self.list.clear()
        usage: dict[int, int] = {}
        for cid in self.config.scene_subs.values():
            usage[cid] = usage.get(cid, 0) + 1
        for cid, name in self.config.sub_categories.items():
            item = QListWidgetItem(f"{cid}  {name}（{usage.get(cid, 0)} 个场景）")
            item.setData(Qt.ItemDataRole.UserRole, cid)
            self.list.addItem(item)
            if cid == select:
                self.list.setCurrentItem(item)

    def _current(self) -> int | None:
        item = self.list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _add(self) -> None:
        cid = ask_new_sub_category(self, self.config)
        if cid is not None:
            self._refresh(select=cid)

    def _rename(self) -> None:
        cid = self._current()
        if cid is None:
            return
        old = self.config.sub_categories[cid]
        name, ok = QInputDialog.getText(self, "重命名小场景类别", f"类别 {cid} 的名称：", QLineEdit.EchoMode.Normal, old)
        if not ok or name.strip() == old:
            return
        error = self.config.validate_sub_category_name(name, cid)
        if error:
            QMessageBox.warning(self, "名称无效", error)
            return
        self.config.sub_categories[cid] = name.strip()
        self._refresh(select=cid)

    def _remove(self) -> None:
        cid = self._current()
        if cid is None:
            return
        used = [scene for scene, c in self.config.scene_subs.items() if c == cid]
        text = f"删除类别 {cid}“{self.config.sub_categories[cid]}”？"
        if used:
            text += f"\n\n以下 {len(used)} 个场景的小场景标注会被清除：\n" + "、".join(used)
        if QMessageBox.question(self, "删除小场景类别", text) == QMessageBox.StandardButton.Yes:
            self.config.remove_sub_category(cid)
            self._refresh()
