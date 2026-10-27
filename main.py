"""启动数据集构建工具：python main.py"""

import sys

from PySide6.QtWidgets import QApplication

from carinout.gui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
