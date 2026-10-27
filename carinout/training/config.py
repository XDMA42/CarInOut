"""训练配置：数据、模型、优化三部分，保存为 JSON。

本模块不依赖 torch，界面和命令行都用它读写配置；模型自身的超参数（如 MLP 的隐藏层节点数）
放在 model.params 里，由对应模型的 ModelSpec 负责校验。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from ..merger import LABEL_COLUMN

MISSING_DROP = "drop"  # 丢弃含缺失值的行
MISSING_MEAN = "mean"  # 用训练集均值填充
MISSING_CONSTANT = "constant"  # 用固定值填充（如 RSSI 缺失填 -120）
MISSING_CHOICES = (MISSING_DROP, MISSING_MEAN, MISSING_CONSTANT)

SPLIT_POINT = "point"  # 按 (场景, 点位) 整组划分，同一点位的数据不会同时出现在训练集和验证/测试集
SPLIT_ROW = "row"  # 按行随机划分
SPLIT_CHOICES = (SPLIT_POINT, SPLIT_ROW)

OPTIMIZER_CHOICES = ("adam", "sgd")
CLASS_WEIGHT_CHOICES = ("none", "balanced")
DEVICE_CHOICES = ("auto", "cpu", "cuda")


@dataclass
class DataConfig:
    dataset: str = ""  # 数据集构建工具导出的总表 _dataset.csv
    # 可选：独立验证集 / 测试集，可以有多个。每项是含 _dataset.csv 的文件夹、其上级文件夹（取各子文件夹中的
    # _dataset.csv）或 csv 文件；指定后不再从 dataset 中划分验证集 / 测试集
    val_datasets: list[str] = field(default_factory=list)
    test_datasets: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)  # 为空时使用全部非常数的数值特征列
    label_column: str = LABEL_COLUMN
    missing: str = MISSING_DROP
    fill_value: float = 0.0  # missing == constant 时使用
    standardize: bool = True  # 按训练集的均值、标准差做 z-score 标准化
    split: str = SPLIT_POINT
    val_ratio: float = 0.2  # 指定了 val_datasets 时忽略
    test_ratio: float = 0.1  # 指定了 test_datasets 时忽略


@dataclass
class ModelConfig:
    name: str = "mlp"
    params: dict[str, Any] = field(default_factory=dict)  # 未给出的参数取模型默认值


@dataclass
class OptimConfig:
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 100
    l2_lambda: float = 1e-4  # 总损失 = 交叉熵 + l2_lambda × Σ‖W‖²（只对权重，不含偏置）
    optimizer: str = "adam"
    class_weight: str = "none"  # balanced：按类别样本数的倒数给交叉熵加权
    early_stopping: int = 0  # 验证集交叉熵连续这么多个 epoch 没有改善就停止；0 表示不提前停止
    seed: int = 42  # 同时用于数据划分和模型初始化
    device: str = "auto"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: OptimConfig = field(default_factory=OptimConfig)
    output_dir: str = "runs"  # 每次训练在其中新建一个子文件夹
    run_name: str = ""  # 子文件夹名的后缀，为空时用模型名

    def validate(self) -> None:
        """检查取值范围，有问题时抛出 ValueError。模型参数由模型自己校验。"""
        d, t = self.data, self.train
        errors = []
        if d.missing not in MISSING_CHOICES:
            errors.append(f"data.missing 应为 {'/'.join(MISSING_CHOICES)}")
        if d.split not in SPLIT_CHOICES:
            errors.append(f"data.split 应为 {'/'.join(SPLIT_CHOICES)}")
        if not 0 <= d.val_ratio < 1 or not 0 <= d.test_ratio < 1:
            errors.append("验证集、测试集比例应在 0 ~ 1 之间")
        elif (0 if d.val_datasets else d.val_ratio) + (0 if d.test_datasets else d.test_ratio) >= 1:
            errors.append("验证集与测试集比例之和应小于 1，训练集才有数据")
        if t.batch_size < 1:
            errors.append("batch size 至少为 1")
        if not t.learning_rate > 0:
            errors.append("learning rate 应大于 0")
        if t.epochs < 1:
            errors.append("epoch 数至少为 1")
        if t.l2_lambda < 0:
            errors.append("L2 系数不能为负")
        if t.optimizer not in OPTIMIZER_CHOICES:
            errors.append(f"train.optimizer 应为 {'/'.join(OPTIMIZER_CHOICES)}")
        if t.class_weight not in CLASS_WEIGHT_CHOICES:
            errors.append(f"train.class_weight 应为 {'/'.join(CLASS_WEIGHT_CHOICES)}")
        if t.early_stopping < 0:
            errors.append("提前停止的 epoch 数不能为负")
        if t.device not in DEVICE_CHOICES:
            errors.append(f"train.device 应为 {'/'.join(DEVICE_CHOICES)}")
        if errors:
            raise ValueError("；".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        """从字典构建；未知的键会报错（避免 batchsize 之类的拼写错误被静默忽略）。"""
        sections = {"data": DataConfig, "model": ModelConfig, "train": OptimConfig}
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key in sections:
                if not isinstance(value, dict):
                    raise ValueError(f"配置项 {key} 应为对象")
                if key == "data":
                    value = _migrate_data(value)
                kwargs[key] = _build_section(sections[key], value, key)
            elif key in ("output_dir", "run_name"):
                kwargs[key] = str(value)
            else:
                raise ValueError(f"未知的配置项：{key}")
        return cls(**kwargs)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        except ValueError as exc:
            raise ValueError(f"配置文件不是有效的 JSON：{exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("配置文件应为 JSON 对象")
        return cls.from_dict(data)


def _migrate_data(values: dict[str, Any]) -> dict[str, Any]:
    """旧版本只有一个独立测试集 test_dataset（字符串），转为 test_datasets 列表。"""
    if "test_dataset" not in values:
        return values
    values = dict(values)
    legacy = values.pop("test_dataset")
    if legacy and "test_datasets" not in values:
        values["test_datasets"] = [legacy]
    return values


def _build_section(cls, values: dict[str, Any], prefix: str):
    defaults = cls()
    known = {f.name for f in fields(cls)}
    kwargs = {}
    for key, value in values.items():
        if key not in known:
            raise ValueError(f"未知的配置项：{prefix}.{key}")
        kwargs[key] = _coerce(f"{prefix}.{key}", value, getattr(defaults, key))
    return cls(**kwargs)


def _coerce(name: str, value: Any, default: Any) -> Any:
    """按默认值的类型转换（JSON 里 1e-3 与 0.001、64 与 64.0 都能接受）。"""
    try:
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise TypeError
            return value
        if isinstance(default, int):
            if isinstance(value, bool) or float(value) != int(float(value)):
                raise TypeError
            return int(float(value))
        if isinstance(default, float):
            if isinstance(value, bool):
                raise TypeError
            return float(value)
        if isinstance(default, str):
            if not isinstance(value, str):
                raise TypeError
            return value
        if isinstance(default, list):
            if not isinstance(value, list):
                raise TypeError
            return [str(v) for v in value]
        if isinstance(default, dict):
            if not isinstance(value, dict):
                raise TypeError
            return dict(value)
    except (TypeError, ValueError):
        raise ValueError(f"配置项 {name} 的值 {value!r} 类型不正确，应与 {default!r} 同类型") from None
    return value
