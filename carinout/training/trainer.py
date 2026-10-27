"""训练循环：小批量训练、每个 epoch 在验证集上评估、保留验证集交叉熵最低的模型，结束后评估各数据集。"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor, nn

from .config import ExperimentConfig
from .data import NUM_CLASSES, SPLITS, PreparedData
from .losses import Objective, class_weights
from .metrics import binary_metrics
from .models import get_spec

# history 中每个 epoch 记录的字段；val_* 在没有验证集时为 NaN
HISTORY_COLUMNS = (
    "epoch",
    "train_loss",  # 训练集总损失（各 batch 平均）= train_ce + l2
    "train_ce",
    "l2",  # λ × Σ‖W‖²
    "val_loss",  # 验证集交叉熵 + epoch 结束时的 L2 项
    "val_ce",
    "train_accuracy",
    "val_accuracy",
    "val_precision",
    "val_recall",
    "val_f1",
    "val_auc",
    "seconds",
)
EVAL_BATCH = 8192


@dataclass
class EvalResult:
    loss: float  # 交叉熵
    metrics: dict  # binary_metrics 的结果
    probs: np.ndarray  # 车内概率


@dataclass
class TrainResult:
    model: nn.Module
    history: list[dict] = field(default_factory=list)
    best_epoch: int = 0
    stopped: bool = False  # 手动停止
    early_stopped: bool = False
    device: str = "cpu"
    num_parameters: int = 0
    evals: dict[str, EvalResult] = field(default_factory=dict)  # train / val / test，空数据集不含


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("没有可用的 CUDA 显卡，请把设备改为 auto 或 cpu")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def make_optimizer(name: str, params, lr: float) -> torch.optim.Optimizer:
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"未知的优化器：{name}")


@torch.no_grad()
def evaluate(model: nn.Module, objective: Objective, x: Tensor, y: Tensor) -> EvalResult:
    """整个数据集上的交叉熵（与训练时同样的类别加权）和车内概率。"""
    model.eval()
    ce_sum, weight_sum, probs = 0.0, 0.0, []
    for start in range(0, len(y), EVAL_BATCH):
        logits = model(x[start : start + EVAL_BATCH])
        target = y[start : start + EVAL_BATCH]
        ce_sum += objective.cross_entropy(logits, target, reduction="sum").item()
        w = objective.class_weights
        weight_sum += w[target].sum().item() if w is not None else len(target)
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu())
    prob = torch.cat(probs).numpy().astype(np.float64) if probs else np.zeros(0)
    labels = y.cpu().numpy()
    loss = ce_sum / weight_sum if weight_sum else math.nan
    return EvalResult(loss, binary_metrics(labels, prob), prob)


def fit(
    cfg: ExperimentConfig,
    data: PreparedData,
    on_epoch: Callable[[dict], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    log: Callable[[str], None] | None = None,
) -> TrainResult:
    """按配置训练模型。on_epoch 在每个 epoch 结束时收到一行 history 记录；should_stop 返回 True 时尽快停止。"""
    log = log or (lambda _msg: None)
    should_stop = should_stop or (lambda: False)
    opt = cfg.train
    device = resolve_device(opt.device)
    torch.manual_seed(opt.seed)

    spec = get_spec(cfg.model.name)
    model = spec.build(len(data.features), NUM_CLASSES, cfg.model.params).to(device)
    weights = class_weights(data.train.y, NUM_CLASSES, opt.class_weight)
    objective = Objective(opt.l2_lambda, weights.to(device) if weights is not None else None)
    optimizer = make_optimizer(opt.optimizer, model.parameters(), opt.learning_rate)
    result = TrainResult(model, device=str(device), num_parameters=sum(p.numel() for p in model.parameters()))
    log(f"模型：{spec.summary(len(data.features), NUM_CLASSES, cfg.model.params)}；设备：{device}")
    if weights is not None:
        log("类别权重：" + "，".join(f"{w:.3f}" for w in weights.tolist()))

    tensors = {
        name: (torch.from_numpy(data.split(name).x).to(device), torch.from_numpy(data.split(name).y).to(device))
        for name in SPLITS
    }
    x_train, y_train = tensors["train"]
    x_val, y_val = tensors["val"]
    has_val = len(y_val) > 0
    generator = torch.Generator().manual_seed(opt.seed)

    best_score, best_state = math.inf, None
    for epoch in range(1, opt.epochs + 1):
        started = time.perf_counter()
        model.train()
        ce_sum = torch.zeros((), device=device)
        l2_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device, dtype=torch.long)
        order = torch.randperm(len(y_train), generator=generator).to(device)
        for start in range(0, len(order), opt.batch_size):
            if should_stop():
                result.stopped = True
                break
            idx = order[start : start + opt.batch_size]
            logits = model(x_train[idx])
            total, ce, l2 = objective(model, logits, y_train[idx])
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            ce_sum += ce.detach() * len(idx)
            l2_sum += l2.detach() * len(idx)
            correct += (logits.argmax(dim=1) == y_train[idx]).sum()
        if result.stopped:  # 不完整的 epoch 不记录
            break

        n = len(y_train)
        train_ce, l2_mean = ce_sum.item() / n, l2_sum.item() / n
        l2_now = objective.l2_term(model).item()
        record = {
            "epoch": epoch,
            "train_loss": train_ce + l2_mean,
            "train_ce": train_ce,
            "l2": l2_mean,
            "train_accuracy": correct.item() / n,
        }
        if has_val:
            val = evaluate(model, objective, x_val, y_val)
            record.update(val_loss=val.loss + l2_now, val_ce=val.loss)
            record.update({f"val_{k}": val.metrics[k] for k in ("accuracy", "precision", "recall", "f1", "auc")})
        record["seconds"] = time.perf_counter() - started
        record = {key: record.get(key, math.nan) for key in HISTORY_COLUMNS}
        result.history.append(record)

        # 有验证集时按验证集交叉熵选最佳 epoch（衡量泛化能力，不受 L2 项影响），否则按训练集总损失
        score = record["val_ce"] if has_val else record["train_loss"]
        if score < best_score:
            best_score, result.best_epoch = score, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if on_epoch:
            on_epoch(record)
        if opt.early_stopping and epoch - result.best_epoch >= opt.early_stopping:
            result.early_stopped = True
            monitored = "验证集交叉熵" if has_val else "训练集损失"
            log(f"{monitored}已连续 {opt.early_stopping} 个 epoch 没有改善，提前停止")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        log(f"使用第 {result.best_epoch} 个 epoch 的模型参数（{'验证集交叉熵' if has_val else '训练集损失'}最低）")
    for name in SPLITS:
        x, y = tensors[name]
        if len(y):
            result.evals[name] = evaluate(model, objective, x, y)
    model.cpu()
    return result
