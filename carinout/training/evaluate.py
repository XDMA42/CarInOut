"""用训练好的模型评估新的数据（独立验证集 / 测试集），不再训练。

模型会按训练时的方式选取特征、填充缺失值、标准化。结果保存在模型所在的训练结果文件夹里，
例如 runs/20261027-153012_mlp/eval_20261028-101500/：
  evaluation.json   模型、所用数据，各集合的指标与混淆矩阵，按文件夹、按场景的准确率
  predictions.csv   每个样本的 split、source、ts、scene、point、label、车内概率 prob、预测 pred（格式同训练结果）
  evaluation.png    混淆矩阵与 ROC 曲线
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..merger import LABEL_COLUMN, POINT_COLUMN, SCENE_COLUMN, TS_COLUMN
from .config import ExperimentConfig
from .data import SOURCE_COLUMN, SPLIT_NAMES, SPLITS, dataset_name, load_table, parse_labels, resolve_datasets
from .metrics import binary_metrics
from .runs import (
    CONFIG_FILE,
    MODEL_FILE,
    PRED_COLUMN,
    PREDICTIONS_FILE,
    PROB_COLUMN,
    SPLIT_COLUMN,
    TRUE_COLUMN,
    Predictor,
    json_safe,
    per_scene_accuracy,
    per_source_metrics,
    read_predictions,
    split_scores,
    unique_dir,
)

EVALUATION_FILE = "evaluation.json"
FIGURE_FILE = "evaluation.png"
EVAL_DIR_PREFIX = "eval_"


@dataclass
class EvaluationRecord:
    path: Path  # 评估结果文件夹
    model_dir: Path  # 被评估模型的训练结果文件夹
    metrics: dict  # 与训练结果的 metrics.json 同样的 splits / sources / scenes，另有 model、datasets、notes
    predictions: pd.DataFrame


def evaluate_model(
    model_dir: str | Path,
    val_datasets: list[str],
    test_datasets: list[str],
    log: Callable[[str], None] | None = None,
    threshold: float | None = None,
) -> EvaluationRecord:
    """用 model_dir 中的模型评估独立验证集、测试集（各自可有多个文件夹），保存并返回结果。
    threshold 为判定阈值，默认使用模型训练时设定的阈值。"""
    log = log or (lambda _msg: None)
    model_dir = Path(model_dir)
    if model_dir.is_file():  # 也可以直接给 model.pt
        model_dir = model_dir.parent
    if not (model_dir / MODEL_FILE).is_file():
        raise ValueError(f"{model_dir} 中没有模型文件 {MODEL_FILE}，请选择训练结果文件夹")
    files = {"val": resolve_datasets(val_datasets), "test": resolve_datasets(test_datasets)}
    if not files["val"] and not files["test"]:
        raise ValueError("请至少指定一个验证集或测试集文件夹")
    both = {f.resolve() for f in files["val"]} & {f.resolve() for f in files["test"]}
    if both:
        raise ValueError(f"{'、'.join(sorted(dataset_name(p) for p in both))} 同时出现在验证集和测试集中")

    predictor = Predictor.load(model_dir)
    threshold = predictor.threshold if threshold is None else threshold
    if not 0 < threshold < 1:
        raise ValueError("判定阈值应在 0 ~ 1 之间（不含 0 和 1）")
    label_column, train_files = _training_info(model_dir)
    notes: list[str] = []
    used_in_training = [f for paths in files.values() for f in paths if f.resolve() in train_files]
    if used_in_training:
        names = "、".join(dataset_name(f) for f in used_in_training)
        notes.append(f"注意：{names} 是该模型的训练数据，指标会偏乐观")
    n_features = len(predictor.features)
    log(f"评估模型：{model_dir}（{predictor.describe()}，{n_features} 个输入特征，判定阈值 {threshold:g}）")

    frames = [
        _predict_file(predictor, path, split, label_column, threshold, notes)
        for split, paths in files.items()
        for path in paths
    ]
    predictions = pd.concat(frames, ignore_index=True)
    if predictions.empty:
        raise ValueError("没有可以评估的数据行")
    splits = {}
    for split in SPLITS:
        part = predictions[predictions[SPLIT_COLUMN] == split]
        if len(part):
            labels, probs = part[TRUE_COLUMN].to_numpy(), part[PROB_COLUMN].to_numpy()
            splits[split] = {"loss": _cross_entropy(labels, probs), **binary_metrics(labels, probs, threshold)}
    metrics = {
        "model": str(model_dir),
        "model_description": predictor.describe(),
        "features": list(predictor.features),
        "threshold": threshold,
        "datasets": {split: [str(f.resolve()) for f in paths] for split, paths in files.items() if paths},
        "splits": splits,
        "sources": per_source_metrics(predictions, threshold),
        "scenes": per_scene_accuracy(predictions).to_dict("records"),
        "notes": notes,
    }

    out = unique_dir(model_dir / f"{EVAL_DIR_PREFIX}{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    predictions.to_csv(out / PREDICTIONS_FILE, index=False, encoding="utf-8-sig")
    (out / EVALUATION_FILE).write_text(json.dumps(json_safe(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    for note in notes:
        log(note)
    for split, values in splits.items():
        log(
            f"  {SPLIT_NAMES[split]}：{values['n']:,} 行，准确率 {values['accuracy']:.2%}，"
            f"平衡准确率 {values['balanced_accuracy']:.2%}，AUC {values['auc']:.4f}"
        )
    log(f"评估结果已保存到 {out}")
    return EvaluationRecord(out, model_dir, metrics, predictions)


def save_evaluation_figure(record: EvaluationRecord) -> None:
    from . import plots  # matplotlib 只在画图时需要

    fig = plots.new_figure(13, 4.2, dpi=120)
    plots.draw_evaluation(fig, split_scores(record.predictions), record.metrics.get("threshold", 0.5))
    fig.savefig(record.path / FIGURE_FILE, facecolor=plots.SURFACE)


def load_evaluation(path: str | Path) -> EvaluationRecord:
    path = Path(path)
    if not is_evaluation_dir(path):
        raise ValueError(f"{path} 不是模型评估结果文件夹（缺少 {EVALUATION_FILE}）")
    metrics = json.loads((path / EVALUATION_FILE).read_text(encoding="utf-8"))
    model_dir = Path(metrics.get("model", path.parent))
    return EvaluationRecord(path, model_dir, metrics, read_predictions(path / PREDICTIONS_FILE))


def is_evaluation_dir(path: str | Path) -> bool:
    return (Path(path) / EVALUATION_FILE).is_file()


def _training_info(model_dir: Path) -> tuple[str, set[Path]]:
    """从训练配置中取标签列名和训练用的数据文件（用于提示评估数据是否用过）。"""
    try:
        cfg = ExperimentConfig.load(model_dir / CONFIG_FILE)
    except (OSError, ValueError):
        return LABEL_COLUMN, set()
    train_files: set[Path] = set()
    for entry in cfg.data.datasets:
        try:
            train_files.update(f.resolve() for f in resolve_datasets([entry]))
        except ValueError:  # 训练数据已移动或删除
            pass
    return cfg.data.label_column, train_files


def _predict_file(
    predictor: Predictor, path: Path, split: str, label_column: str, threshold: float, notes: list[str]
) -> pd.DataFrame:
    name = dataset_name(path)
    df = load_table(path)
    try:
        labels = parse_labels(df, label_column)
        prob = predictor.predict_proba(df)
    except ValueError as exc:
        raise ValueError(f"{name}：{exc}") from None
    keep = (labels.notna() & ~np.isnan(prob)).to_numpy()
    skipped = int((~keep).sum())
    if skipped:
        notes.append(f"{name}：{skipped} 行没有标签，或特征有缺失值（模型训练时选择了丢弃缺失值），未参与评估")
    frame = df.loc[keep, [c for c in (TS_COLUMN, SCENE_COLUMN, POINT_COLUMN) if c in df.columns]].copy()
    frame.insert(0, SOURCE_COLUMN, name)
    frame.insert(0, SPLIT_COLUMN, split)
    frame[TRUE_COLUMN] = labels[keep].astype(np.int64).to_numpy()
    frame[PROB_COLUMN] = prob[keep]
    frame[PRED_COLUMN] = (prob[keep] >= threshold).astype(np.int64)
    return frame.reset_index(drop=True)


def _cross_entropy(labels: np.ndarray, probs: np.ndarray) -> float:
    """由车内概率算平均交叉熵（不加类别权重）。"""
    if not len(labels):
        return math.nan
    p_true = np.clip(np.where(labels == 1, probs, 1 - probs), 1e-12, 1.0)
    return float(-np.log(p_true).mean())
