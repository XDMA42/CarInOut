"""二分类评价指标（正类 = 1 车内），只依赖 numpy。"""

from __future__ import annotations

import numpy as np

METRIC_NAMES = {
    "accuracy": "准确率",
    "precision": "精确率",
    "recall": "召回率",
    "f1": "F1",
    "auc": "AUC",
}


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """2×2 混淆矩阵，行 = 真实类别，列 = 预测类别：[[TN, FP], [FN, TP]]。"""
    cm = np.zeros((2, 2), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)), 1)
    return cm


def roc_curve(y_true: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回 (假正率, 真正率)，从 (0, 0) 开始；分数相同的样本作为同一个阈值处理。"""
    y_true = np.asarray(y_true)
    score = np.asarray(score, dtype=np.float64)
    order = np.argsort(-score, kind="mergesort")
    score, y_true = score[order], y_true[order]
    # 每个不同分数的最后一个位置作为阈值点
    last = np.r_[np.flatnonzero(np.diff(score)), len(score) - 1] if len(score) else np.array([], int)
    tps = np.cumsum(y_true == 1)[last]
    fps = (last + 1) - tps
    pos, neg = int((y_true == 1).sum()), int((y_true != 1).sum())
    tpr = np.r_[0.0, tps / pos] if pos else np.full(len(last) + 1, np.nan)
    fpr = np.r_[0.0, fps / neg] if neg else np.full(len(last) + 1, np.nan)
    return fpr, tpr


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """ROC 曲线下面积；只有一个类别时为 NaN。"""
    fpr, tpr = roc_curve(y_true, score)
    if len(fpr) < 2 or np.isnan(fpr).any() or np.isnan(tpr).any():
        return float("nan")
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))  # 梯形法


def binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    """prob 为正类（车内）概率。没有样本时各指标为 NaN；分母为 0 的精确率 / 召回率记为 0。"""
    y_true = np.asarray(y_true, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float64)
    y_pred = (prob >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred)
    (tn, fp), (fn, tp) = cm
    n = int(cm.sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    nan = float("nan")
    return {
        "n": n,
        "accuracy": float((tp + tn) / n) if n else nan,
        "precision": float(precision) if n else nan,
        "recall": float(recall) if n else nan,
        "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else (0.0 if n else nan),
        "auc": roc_auc(y_true, prob),
        "confusion": cm.tolist(),
    }
