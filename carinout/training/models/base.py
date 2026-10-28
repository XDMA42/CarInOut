"""模型描述：每种模型声明自己的超参数，训练流程和界面只通过 ModelSpec 使用模型。

约定：模型是一个 torch.nn.Module，输入 (batch, 特征数) 的 float32 张量，输出 (batch, 类别数) 的 logits。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torch import nn

KIND_INT = "int"
KIND_FLOAT = "float"
KIND_INT_LIST = "int_list"  # 如隐藏层节点数 “64, 32”
KIND_CHOICE = "choice"
KIND_BOOL = "bool"


@dataclass(frozen=True)
class ParamSpec:
    """一个模型超参数。界面会按 kind 生成对应的输入控件。"""

    name: str
    label: str
    kind: str
    default: Any
    help: str = ""
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()
    short: str = ""  # 参数遍历时组合名称中的缩写，如 hidden_sizes → h
    example: str = ""  # 参数遍历时的取值示例，如 “0, 0.2, 0.5”；为空时按当前值生成

    def parse(self, value: Any) -> Any:
        """把配置文件或界面中的值转换为正确类型并检查范围，非法时抛出 ValueError。"""
        try:
            if self.kind == KIND_INT_LIST:
                if isinstance(value, str):
                    parts = [p for p in value.replace("，", ",").replace(" ", ",").split(",") if p]
                else:
                    parts = list(value)
                result = [_as_int(p) for p in parts]
                if not result:
                    raise ValueError
            elif self.kind == KIND_INT:
                result = _as_int(value)
            elif self.kind == KIND_FLOAT:
                result = float(value)
            elif self.kind == KIND_BOOL:
                if not isinstance(value, bool):
                    raise ValueError
                return value
            elif self.kind == KIND_CHOICE:
                if value not in self.choices:
                    raise ValueError
                return value
            else:
                raise TypeError(f"未知的参数类型 {self.kind}")
        except (TypeError, ValueError):
            raise ValueError(f"参数 “{self.label}” 的值 {value!r} 无效") from None
        for v in result if isinstance(result, list) else [result]:
            self._check_range(v)
        return result

    def format(self, value: Any) -> str:
        if self.kind == KIND_INT_LIST:
            return ", ".join(str(v) for v in value)
        return str(value)

    def _check_range(self, value: float) -> None:
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"参数 “{self.label}” 不能小于 {self.minimum}")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"参数 “{self.label}” 不能大于 {self.maximum}")


@dataclass(frozen=True)
class ModelSpec:
    name: str  # 写入配置文件的标识
    label: str  # 界面显示的名称
    description: str
    params: tuple[ParamSpec, ...]
    factory: Callable[..., nn.Module]  # factory(in_features, num_classes, **params)
    # 可选：根据输入维度和参数给出结构说明，如 “8 → 64 → 32 → 2”
    describe: Callable[[int, int, dict[str, Any]], str] | None = None

    def resolve_params(self, params: dict[str, Any] | None) -> dict[str, Any]:
        """补全默认值并校验；出现未声明的参数时报错。"""
        params = dict(params or {})
        unknown = set(params) - {p.name for p in self.params}
        if unknown:
            raise ValueError(f"模型 {self.label} 没有参数：{'、'.join(sorted(unknown))}")
        return {p.name: p.parse(params.get(p.name, p.default)) for p in self.params}

    def build(self, in_features: int, num_classes: int, params: dict[str, Any] | None = None) -> nn.Module:
        return self.factory(in_features, num_classes, **self.resolve_params(params))

    def summary(self, in_features: int, num_classes: int, params: dict[str, Any] | None = None) -> str:
        resolved = self.resolve_params(params)
        model = self.factory(in_features, num_classes, **resolved)
        n_params = sum(p.numel() for p in model.parameters())
        text = self.describe(in_features, num_classes, resolved) if self.describe else self.label
        return f"{text}，共 {n_params:,} 个参数"


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError
    number = float(str(value).strip())
    if number != int(number):
        raise ValueError
    return int(number)
