"""画出数据集中各锚点特征的分布与相关性：python plot_features.py 数据集文件夹 [--features dis_sm,dis_main_sm,rssi_init]
python plot_features.py --help 查看全部参数。
"""

import os
import sys

# 必须在导入 numpy 之前设置，原因见 main.py。
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from carinout.training.feature_plots import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
