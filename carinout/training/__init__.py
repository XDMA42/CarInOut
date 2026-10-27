"""车内外识别模型训练：配置 → 数据准备 → 模型（可替换）→ 训练 → 评估与可视化 → 保存结果。

- config.py   训练配置（JSON）
- data.py     读取数据集构建工具导出的总表、选择特征、缺失值处理、划分与标准化
- models/     模型注册表，内置 MLP；新增模型只需注册一个 ModelSpec
- losses.py   交叉熵 + L2 正则
- trainer.py  训练循环与评估
- metrics.py  评价指标
- plots.py    损失曲线、指标曲线、混淆矩阵、ROC
- runs.py     训练结果目录的保存 / 读取，Predictor 推理
- cli.py      命令行入口
"""
