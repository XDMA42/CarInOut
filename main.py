"""启动数据集构建工具：python main.py"""

import os
import sys

# 必须在导入 numpy/pandas 之前设置。本工具不做矩阵运算，但 numpy 自带的 OpenBLAS
# 加载时会按 CPU 核数为每个线程预留内存，核数多或内存紧张时会报
# “OpenBLAS error: Memory allocation still failed after 10 retries” 并直接退出。
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

from PySide6.QtWidgets import QApplication  # noqa: E402

from carinout.gui.main_window import MainWindow  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
