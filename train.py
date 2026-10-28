"""启动模型训练平台：python train.py
命令行训练：python train.py --config 配置.json（python train.py --help 查看全部参数）
"""

import os
import sys

# 必须在导入 numpy 之前设置，原因见 main.py。只限制 numpy 自带的 OpenBLAS，不影响 torch 的线程数。
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def main() -> int:
    # Windows 上需先于 PySide6 导入 torch，否则可能报 “OSError: [WinError 1114] ... c10.dll” 初始化失败。
    # 放在 main() 里而不是模块顶层：参数遍历的子进程会重新导入本文件，它们在 sweep_worker 中自行导入 torch，
    # 导入失败（如虚拟内存不足）时才能把原因报告给界面。
    import torch  # noqa: F401

    if len(sys.argv) > 1:
        from carinout.training.cli import main as cli_main

        return cli_main(sys.argv[1:])

    from PySide6.QtWidgets import QApplication

    from carinout.gui.train_window import TrainWindow

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = TrainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
