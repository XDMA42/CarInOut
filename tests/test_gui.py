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


def test_sub_scene_dropdown_writes_to_its_own_scene(window, tmp_path: Path):
    window.config.add_sub_category("座椅区")
    window.config.add_sub_category("车外")
    window.folder_list.add_folders([make_folder(tmp_path / "raw", ["座椅", "车外"])])
    rows = scene_rows(window)
    for scene, cid in (("座椅", 0), ("车外", 1)):
        combo = window.tree.itemWidget(rows[scene], 4)
        combo.setCurrentIndex(combo.findData(cid))
    assert window.config.scene_subs == {"座椅": 0, "车外": 1}


def test_folder_list_ignores_duplicates_and_files(window, tmp_path: Path):
    raw = make_folder(tmp_path / "raw", ["座椅"])
    assert window.folder_list.add_folders([raw, raw, raw / "座椅-1-[SLAVE1].csv"]) == 1
    assert window.folder_list.add_folders([raw]) == 0
    assert window._target_dir() == tmp_path / "merged" / "raw"
