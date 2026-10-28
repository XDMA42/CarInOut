"""模型注册表。

新增模型：在本目录新建模块，实现一个 nn.Module（输入 (batch, 特征数)，输出 (batch, 类别数) 的 logits），
定义 ModelSpec 并在下方 _BUILTIN 中加入即可；界面会自动列出该模型并按 ParamSpec 生成参数输入框，
训练、评估、可视化流程无需修改。也可以在外部代码中调用 register() 注册。
"""

from __future__ import annotations

from typing import Any

from torch import nn

from . import mlp
from .base import ModelSpec, ParamSpec

__all__ = ["ModelSpec", "ParamSpec", "available_models", "build_model", "get_spec", "register"]

_REGISTRY: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> ModelSpec:
    if spec.name in _REGISTRY:
        raise ValueError(f"模型 {spec.name} 已注册")
    _REGISTRY[spec.name] = spec
    return spec


def get_spec(name: str) -> ModelSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"未知的模型：{name}（可选：{'、'.join(_REGISTRY)}）") from None


def available_models() -> list[ModelSpec]:
    return list(_REGISTRY.values())


def build_model(name: str, in_features: int, num_classes: int, params: dict[str, Any] | None = None) -> nn.Module:
    return get_spec(name).build(in_features, num_classes, params)


_BUILTIN = (mlp.SPEC,)
for _spec in _BUILTIN:
    register(_spec)
