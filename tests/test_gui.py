"""主窗口的基本交互（无界面运行）。"""

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(app, tmp_path: Path, monkeypatch):
    import carinout.anchors as anchors
    from carinout.gui.main_window import MainWindow

    monkeypatch.setattr(anchors, "CONFIG_PATH", tmp_path / "config.json")
    w = MainWindow()
    yield w
    w.close()


def make_folder(path: Path, scenes: list[str]) -> Path:
    path.mkdir(parents=True)
    for scene in scenes:
        for slave in ("SLAVE1", "SLAVE2"):
            (path / f"{scene}-1-[{slave}].csv").write_text("ts,rssi,errcode\n1,-50,0\n")
    return path


def scene_rows(w) -> dict:
    return {w.tree.topLevelItem(i).text(0): w.tree.topLevelItem(i) for i in range(w.tree.topLevelItemCount())}


@pytest.mark.parametrize("n_folders", [1, 2])
def test_label_dropdowns_write_to_their_own_scene(window, tmp_path: Path, n_folders: int):
    folders = [make_folder(tmp_path / "1023采集", ["座椅", "车外", "后备箱"])]
    if n_folders == 2:
        folders.append(make_folder(tmp_path / "1024采集", ["座椅", "脚坑"]))
    window.folder_list.add_folders(folders)

    rows = scene_rows(window)
    assert sorted(rows) == sorted({"座椅", "车外", "后备箱"} | ({"脚坑"} if n_folders == 2 else set()))
    wanted = {scene: i % 2 for i, scene in enumerate(rows)}
    for scene, label in wanted.items():
        combo = window.tree.itemWidget(rows[scene], 3)
        combo.setCurrentIndex(combo.findData(label))
    assert window.config.scene_labels == wanted

    if n_folders == 2:
        # 同名场景在两个文件夹中各有一个来源行；来源行没有标签下拉框
        seat = rows["座椅"]
        assert [seat.child(i).text(0) for i in range(seat.childCount())] == ["1023采集", "1024采集"]
        assert window.tree.itemWidget(seat.child(0), 3) is None


def test_folder_list_ignores_duplicates_and_files(window, tmp_path: Path):
    raw = make_folder(tmp_path / "raw", ["座椅"])
    assert window.folder_list.add_folders([raw, raw, raw / "座椅-1-[SLAVE1].csv"]) == 1
    assert window.folder_list.add_folders([raw]) == 0
    assert window._target_dir() == tmp_path / "merged" / "raw"


def point_rows(w) -> dict:
    return {key[1:]: item for key, item in w._point_items.items()}


def test_point_label_override_and_display(window, tmp_path: Path):
    from carinout.gui.main_window import LABEL_COLOR_INHERIT, LABEL_COLOR_OVERRIDE

    raw = tmp_path / "raw"
    raw.mkdir()
    for point in ("1", "2", "3"):
        for slave in ("SLAVE1", "SLAVE2"):
            (raw / f"座椅-{point}-[{slave}].csv").write_text("ts,rssi,errcode\n1,-50,0\n")
    window.folder_list.add_folders([raw])
    scene_combo = window.tree.itemWidget(scene_rows(window)["座椅"], 3)
    scene_combo.setCurrentIndex(scene_combo.findData(1))

    window._set_point_labels([("座椅", "2")], 0)
    rows = point_rows(window)
    assert [rows[("座椅", p)].text(3) for p in ("1", "2", "3")] == ["1 车内 ▾", "0 车外 ▾", "1 车内 ▾"]
    assert rows[("座椅", "1")].foreground(3).color() == LABEL_COLOR_INHERIT
    assert rows[("座椅", "2")].foreground(3).color() == LABEL_COLOR_OVERRIDE and rows[("座椅", "2")].font(3).bold()

    # 改场景标签：跟随场景的点位跟着变，单独设置的不变
    scene_combo.setCurrentIndex(scene_combo.findData(0))
    rows = point_rows(window)
    assert [rows[("座椅", p)].text(3) for p in ("1", "2", "3")] == ["0 车外 ▾", "0 车外 ▾", "0 车外 ▾"]
    assert window.config.label_for("座椅", "2") == 0 and window.config.is_point_overridden("座椅", "2")

    window._set_point_labels([("座椅", "2")], None)
    assert not window.config.point_labels


def test_point_label_menu_applies_to_selected_points(window, tmp_path: Path, monkeypatch):
    folders = [make_folder(tmp_path / "1023采集", ["座椅", "车外"]), make_folder(tmp_path / "1024采集", ["座椅"])]
    window.folder_list.add_folders(folders)
    shown = {}

    def pick(text):
        def fake_exec(menu, _pos):
            shown["actions"] = [a.text() for a in menu.actions() if not a.isSeparator() and a.text()]
            return next(a for a in menu.actions() if a.text() == text)
        return fake_exec

    # 多选两个文件夹里的 “座椅/1” 和 “车外/1”，右键设为车内
    window.tree.clearSelection()
    for key, item in window._point_items.items():
        item.setSelected(True)
    monkeypatch.setattr(window, "_exec_menu", pick("1 车内"))
    window._on_tree_context_menu(window.tree.visualItemRect(next(iter(window._point_items.values()))).center())
    assert window.config.point_labels == {"座椅": {"1": 1}, "车外": {"1": 1}}
    assert shown["actions"][-3:] == ["跟随各自场景", "1 车内", "0 车外"]

    # 同一点位在两个文件夹里共用单独设置：两行都显示为单独设置
    seat_rows = [item for key, item in window._point_items.items() if key[1:] == ("座椅", "1")]
    assert len(seat_rows) == 2 and all(r.font(3).bold() for r in seat_rows)

    # 点击某个点位的标签格：只改这一个点位
    monkeypatch.setattr(window, "_exec_menu", pick("跟随场景（未标注）"))
    window._on_tree_clicked(seat_rows[0], 3)
    assert window.config.point_labels == {"车外": {"1": 1}}
