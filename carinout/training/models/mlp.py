"""多层感知机：输入 → [Linear → ReLU] × 隐藏层数 → Linear（输出各类别 logits）。

默认两层隐藏层，加上输出层共 3 个全连接层。
"""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from .base import KIND_INT_LIST, ModelSpec, ParamSpec


class MLP(nn.Module):
    def __init__(self, in_features: int, num_classes: int, hidden_sizes: Sequence[int] = (64, 32)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for size in hidden_sizes:
            layers += [nn.Linear(prev, size), nn.ReLU()]
            prev = size
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _describe(in_features: int, num_classes: int, params: dict) -> str:
    sizes = [in_features, *params["hidden_sizes"], num_classes]
    return f"{len(sizes) - 1} 层全连接：" + " → ".join(map(str, sizes)) + "（隐藏层 ReLU）"


SPEC = ModelSpec(
    name="mlp",
    label="MLP 多层感知机",
    description="全连接网络，隐藏层激活函数为 ReLU",
    params=(
        ParamSpec(
            "hidden_sizes",
            "隐藏层节点数",
            KIND_INT_LIST,
            [64, 32],
            help="逗号分隔，每个数字是一层隐藏层的节点数。\n"
            "“64, 32” 即 输入 → 64 → 32 → 输出，共 3 层全连接（两层隐藏层 + 输出层）。",
            minimum=1,
            short="h",
        ),
    ),
    factory=lambda in_features, num_classes, hidden_sizes: MLP(in_features, num_classes, hidden_sizes),
    describe=_describe,
)
