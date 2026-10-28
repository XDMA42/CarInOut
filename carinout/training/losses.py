"""损失函数：总损失 = 交叉熵 + λ × L2 正则项（Σ‖W‖²）。

L2 项单独计算而不是交给优化器的 weight_decay，这样两部分可以分别记录、画在损失曲线上。
只对维度 > 1 的参数（全连接 / 卷积的权重）计算 L2，不含偏置和归一化层参数。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def l2_penalty(model: nn.Module) -> Tensor:
    weights = [p for p in model.parameters() if p.ndim > 1]
    if not weights:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.stack([w.pow(2).sum() for w in weights]).sum()


class Objective:
    def __init__(self, l2_lambda: float, class_weights: Tensor | None = None) -> None:
        self.l2_lambda = l2_lambda
        self.class_weights = class_weights

    def cross_entropy(self, logits: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
        return F.cross_entropy(logits, target, weight=self.class_weights, reduction=reduction)

    def l2_term(self, model: nn.Module) -> Tensor:
        if self.l2_lambda == 0:
            return torch.zeros((), device=next(model.parameters()).device)
        return self.l2_lambda * l2_penalty(model)

    def __call__(self, model: nn.Module, logits: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """返回 (总损失, 交叉熵, L2 项)。"""
        ce = self.cross_entropy(logits, target)
        l2 = self.l2_term(model)
        return ce + l2, ce, l2


def class_weights(labels, num_classes: int, mode: str) -> Tensor | None:
    """mode == "balanced" 时权重为 样本总数 / (类别数 × 该类样本数)，与 sklearn 一致。"""
    if mode != "balanced":
        return None
    counts = torch.bincount(torch.as_tensor(labels, dtype=torch.long), minlength=num_classes).float()
    return counts.sum() / (num_classes * counts.clamp(min=1))
