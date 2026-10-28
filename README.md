# CarInOut

基于 UWB 测距（各锚点的首径距离、主径距离、RSSI）的车内外识别。目标平台：128 MHz CPU（带 FPU），模型 RAM ≤ 8 KB。

## 模型对比脚本

`training/compare_models.py` 在同一份数据上对比以下 MCU 友好模型，自动做超参搜索，并报告验证集准确率和部署占用估算：

| 名称 | 模型 |
|---|---|
| `logreg` | 逻辑回归（baseline） |
| `tree` | 单棵决策树 |
| `rf` | 小随机森林 |
| `lgbm` | LightGBM 梯度提升树 |
| `mlp` | 小型全连接网络 |

### 安装

```bash
pip install -r requirements.txt
```

### 数据格式

训练集、验证集各一个 CSV，列名一致：

```
label,scene,A1_fp,A1_mp,A1_rssi,A2_fp,A2_mp,A2_rssi,...
车内,garage_01,1.23,1.51,-62.0,2.80,3.42,-71.5,...
车左,street_03,2.10,2.13,-58.5,-1,-1,-1,...
```

- `label`：标签。可以是车内/车外二分类，也可以是 车内/车前/车左/车右/车后 这样的多分类。
- `scene`：采集场景（地点、车型、采集批次等）。用于分组交叉验证和分场景统计。
- 其余列默认都是特征。不是特征的列（时间戳、ID 等）用 `--drop-cols` 排除。
- 缺失值可以留空，也可以用哨兵值（如 `-1`）表示，后者需要加 `--missing-values -1`。
- 支持 UTF-8（含 Excel 的 BOM）和 GBK 编码。

### 运行

```bash
python training/compare_models.py --train data/train.csv --val data/val.csv --inside-label 车内 --missing-values -1
```

常用参数（完整列表见 `--help`）：

| 参数 | 说明 |
|---|---|
| `--inside-label 车内` | 代表车内的标签值。指定后报告车内/车外准确率；标签是多分类时，用 P(车内) 合并判断，并自动额外训练一组直接做车内/车外二分类的对照（`task=inout`） |
| `--models lgbm,mlp` | 只跑部分模型 |
| `--search random --n-iter 40` | 每个模型随机抽 40 组超参（默认）；`--search grid` 遍历全部组合 |
| `--cv group` | 在训练集上按 `scene` 分组交叉验证来挑超参（默认）。`stratified` 不分组；`val` 直接在验证集上挑（快，但验证集分数偏乐观） |
| `--max-flash-kb 32` | 只在估算 Flash 不超过 32 KB 的组合中挑选 |
| `--threshold 0.5` | P(车内) ≥ 阈值判为车内 |

### 输出（默认 `results/`）

| 文件 | 内容 |
|---|---|
| `summary.csv` | 每个模型的最优结果：交叉验证分数、验证集准确率、误判率、Flash/RAM 估算 |
| `search_<模型>_<任务>.csv` | 每组超参的交叉验证结果，按分数从高到低排序 |
| `per_scene.csv` | 验证集分场景结果，用来找哪个场景差 |
| `confusion_<模型>_<任务>.csv` | 验证集混淆矩阵 |
| `best_params.json`、`models/*.joblib` | 最优超参和训练好的模型 |

指标说明：
- `inout_acc`：车内/车外准确率。
- `false_inside`：车外样本被判为车内的比例，是安全问题。
- `false_outside`：车内样本被判为车外的比例，是体验问题。
- `val_acc`：原始多分类标签的准确率。
- `task=label`：用原始标签训练；`task=inout`：直接训练车内/车外二分类。

`flash_kb`、`ram_b` 是按 float32 实现粗略估算的模型常量和推理工作内存，不包括原始数据的滑动窗口。实际占用以导出成 C 后在板子上的实测为准。

## 调参方法

搜索空间在 `compare_models.py` 顶部的 `SEARCH_SPACES` 里，直接改列表即可。建议流程：

1. 先用默认的随机搜索粗扫一遍。
2. 打开 `search_*.csv`，看排名靠前的组合集中在哪些取值，把范围缩小到那附近，再用 `--search grid` 精调。
3. 分数接近时选 Flash 更小的组合（脚本在分数相同时已经优先选小的）。也可以用 `--max-flash-kb` 直接限制。
4. 看交叉验证分数和验证集分数的差距。交叉验证高、验证集低，说明对新场景泛化差：优先补充场景数据或加强正则，而不是加大模型。

各模型的关键超参：

| 模型 | 超参 | 作用 |
|---|---|---|
| logreg | `C` | 正则强度的倒数，越小正则越强 |
| tree | `max_depth` | 主要控制大小和过拟合。节点数最多 2^depth |
| tree | `min_samples_leaf` | 叶子最少样本数，越大越平滑 |
| rf | `n_estimators` × `max_depth` | 决定 Flash 占用 |
| rf | `max_features` | 越小树之间差异越大 |
| lgbm | `n_estimators` × `num_leaves` | 决定 Flash 占用。**多分类时还要再乘以类别数** |
| lgbm | `learning_rate` | 与 `n_estimators` 反向配合：学习率小就需要更多树 |
| lgbm | `min_child_samples`、`reg_lambda` | 防过拟合 |
| mlp | `hidden_layer_sizes` | 决定 Flash（参数量 × 4 B）和 RAM（相邻两层宽度之和 × 4 B） |
| mlp | `alpha` | L2 正则 |
| mlp | `learning_rate_init` | 训练不收敛或分数抖动时调小 |
