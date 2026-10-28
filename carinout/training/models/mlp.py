"""多层感知机：输入 → [Linear → ReLU → Dropout] × 隐藏层数 → Linear（输出各类别 logits）。

默认两层隐藏层，加上输出层共 3 个全连接层；Dropout 默认为 0（不使用）。
"""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from .base import KIND_FLOAT, KIND_INT_LIST, ModelSpec, ParamSpec


class MLP(nn.Module):
    def __init__(
        self, in_features: int, num_classes: int, hidden_sizes: Sequence[int] = (64, 32), dropout: float = 0.0
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for size in hidden_sizes:
            layers += [nn.Linear(prev, size), nn.ReLU()]
            if dropout > 0:  # 为 0 时不加 Dropout 层，层的编号与旧版本一致，以前保存的 model.pt 仍能加载
                layers.append(nn.Dropout(dropout))
            prev = size
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _describe(in_features: int, num_classes: int, params: dict) -> str:
    sizes = [in_features, *params["hidden_sizes"], num_classes]
    extra = f"，Dropout {params['dropout']:g}" if params["dropout"] > 0 else ""
    return f"{len(sizes) - 1} 层全连接：" + " → ".join(map(str, sizes)) + f"（隐藏层 ReLU{extra}）"


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
        ParamSpec(
            "dropout",
            "Dropout 比例",
            KIND_FLOAT,
            0.0,
            help="训练时每个隐藏层的输出按这个概率随机置 0，用于减轻过拟合；评估和推理时不生效。\n"
            "0 表示不使用，常用 0.1 ~ 0.5。",
            minimum=0.0,
            maximum=0.9,
            short="do",
            example="0, 0.2, 0.5",
        ),
    ),
    factory=MLP,
    describe=_describe,
)
