"""训练结果目录的保存与读取，以及用保存的模型做推理。

每次训练在 output_dir 下新建一个文件夹，例如 runs/20261027-153012_mlp/：
  config.json       完整配置（features 为实际使用的特征），可直接加载后复现
  history.csv       每个 epoch 的损失与验证集指标
  metrics.json      最佳模型在训练 / 验证 / 测试集上的指标与混淆矩阵
  predictions.csv   每个样本的 split、scene、point、ts、label、车内概率 prob、预测 pred
  model.pt          模型参数 + 特征列表 + 缺失值填充值 + 标准化参数（Predictor 加载）
  loss.png / metrics.png / evaluation.png
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..merger import POINT_COLUMN, SCENE_COLUMN
from .config import ExperimentConfig
from .data import META_COLUMNS, NUM_CLASSES, SPLITS, PreparedData, to_numeric
from .models import get_spec
from .trainer import HISTORY_COLUMNS, TrainResult

CONFIG_FILE = "config.json"
HISTORY_FILE = "history.csv"
METRICS_FILE = "metrics.json"
PREDICTIONS_FILE = "predictions.csv"
MODEL_FILE = "model.pt"

PROB_COLUMN = "prob"
PRED_COLUMN = "pred"
SPLIT_COLUMN = "split"
TRUE_COLUMN = "label"


def new_run_dir(cfg: ExperimentConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = re.sub(r'[\\/:*?"<>|\s]+', "_", cfg.run_name.strip()) or cfg.model.name
    base = Path(cfg.output_dir) / f"{stamp}_{suffix}"
    path, i = base, 1
    while path.exists():
        i += 1
        path = base.with_name(f"{base.name}_{i}")
    path.mkdir(parents=True)
    return path


def predictions_frame(data: PreparedData, result: TrainResult) -> pd.DataFrame:
    frames = []
    for name in SPLITS:
        if name not in result.evals:
            continue
        split = data.split(name)
        prob = result.evals[name].probs
        frame = split.meta.copy()
        frame.insert(0, SPLIT_COLUMN, name)
        frame[TRUE_COLUMN] = split.y
        frame[PROB_COLUMN] = prob
        frame[PRED_COLUMN] = (prob >= 0.5).astype(np.int64)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def save_run(
    run_dir: Path, cfg: ExperimentConfig, data: PreparedData, result: TrainResult, figures: bool = True
) -> None:
    """写出结果文件。figures=False 时不画 PNG（界面在后台线程训练，图表改由界面线程调用 save_run_figures）。"""
    run_dir = Path(run_dir)
    saved = ExperimentConfig.from_dict(cfg.to_dict())
    saved.data.features = list(data.features)
    spec = get_spec(cfg.model.name)
    saved.model.params = spec.resolve_params(cfg.model.params)
    saved.save(run_dir / CONFIG_FILE)

    history = pd.DataFrame(result.history, columns=list(HISTORY_COLUMNS))
    history.to_csv(run_dir / HISTORY_FILE, index=False)
    predictions = predictions_frame(data, result)
    predictions.to_csv(run_dir / PREDICTIONS_FILE, index=False, encoding="utf-8-sig")

    metrics = {
        "best_epoch": result.best_epoch,
        "epochs_run": len(result.history),
        "stopped": result.stopped,
        "early_stopped": result.early_stopped,
        "device": result.device,
        "num_parameters": result.num_parameters,
        "features": list(data.features),
        "splits": {name: {"loss": ev.loss, **ev.metrics} for name, ev in result.evals.items()},
        "notes": data.notes,
    }
    (run_dir / METRICS_FILE).write_text(
        json.dumps(_json_safe(metrics), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    torch.save(
        {
            "model": cfg.model.name,
            "params": saved.model.params,
            "num_classes": NUM_CLASSES,
            "features": list(data.features),
            "fill_values": None if data.fill_values is None else data.fill_values.tolist(),
            "scaler_mean": None if data.scaler is None else data.scaler.mean.tolist(),
            "scaler_std": None if data.scaler is None else data.scaler.std.tolist(),
            "state_dict": result.model.state_dict(),
        },
        run_dir / MODEL_FILE,
    )
    if figures:
        save_run_figures(load_run(run_dir))


@dataclass
class RunRecord:
    path: Path
    config: ExperimentConfig
    history: pd.DataFrame
    metrics: dict
    predictions: pd.DataFrame


def load_run(path: str | Path) -> RunRecord:
    path = Path(path)
    if not (path / HISTORY_FILE).is_file():
        raise ValueError(f"{path} 不是训练结果文件夹（缺少 {HISTORY_FILE}）")
    config = ExperimentConfig.load(path / CONFIG_FILE)
    history = pd.read_csv(path / HISTORY_FILE)
    metrics = json.loads((path / METRICS_FILE).read_text(encoding="utf-8")) if (path / METRICS_FILE).is_file() else {}
    pred_path = path / PREDICTIONS_FILE
    predictions = (
        pd.read_csv(pred_path, encoding="utf-8-sig", dtype={c: str for c in META_COLUMNS}, keep_default_na=False)
        if pred_path.is_file()
        else pd.DataFrame()
    )
    return RunRecord(path, config, history, metrics, predictions)


def save_run_figures(run: RunRecord) -> None:
    from . import plots  # matplotlib 只在画图时需要

    plots.save_figures(run.path, run.history, run.metrics.get("best_epoch", 0), split_scores(run.predictions))


def split_scores(predictions: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """split → (真实标签, 车内概率)。"""
    if predictions.empty:
        return {}
    return {
        name: (part[TRUE_COLUMN].to_numpy(dtype=np.int64), part[PROB_COLUMN].to_numpy(dtype=np.float64))
        for name in SPLITS
        if not (part := predictions[predictions[SPLIT_COLUMN] == name]).empty
    }


def per_point_accuracy(predictions: pd.DataFrame) -> pd.DataFrame:
    """按 (split, scene, point) 统计样本数和准确率，准确率低的在前，用于找出容易判错的点位。"""
    if predictions.empty or not {SCENE_COLUMN, POINT_COLUMN} <= set(predictions.columns):
        return pd.DataFrame()
    df = predictions.assign(correct=predictions[PRED_COLUMN] == predictions[TRUE_COLUMN])
    table = (
        df.groupby([SPLIT_COLUMN, SCENE_COLUMN, POINT_COLUMN, TRUE_COLUMN], sort=False)
        .agg(samples=("correct", "size"), accuracy=("correct", "mean"), mean_prob=(PROB_COLUMN, "mean"))
        .reset_index()
    )
    table["_order"] = table[SPLIT_COLUMN].map({name: i for i, name in enumerate(SPLITS)})
    return table.sort_values(["accuracy", "_order"], kind="stable").drop(columns="_order").reset_index(drop=True)


class Predictor:
    """加载 model.pt，对总表格式的数据输出车内概率（与训练时相同的特征、填充和标准化）。"""

    def __init__(self, checkpoint: dict) -> None:
        self.features: list[str] = list(checkpoint["features"])
        self.fill_values = _array(checkpoint["fill_values"])
        self.mean = _array(checkpoint["scaler_mean"])
        self.std = _array(checkpoint["scaler_std"])
        spec = get_spec(checkpoint["model"])
        self.model = spec.build(len(self.features), checkpoint["num_classes"], checkpoint["params"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    @classmethod
    def load(cls, path: str | Path) -> "Predictor":
        """path 可以是 model.pt，也可以是训练结果文件夹。"""
        path = Path(path)
        if path.is_dir():
            path = path / MODEL_FILE
        return cls(torch.load(path, map_location="cpu", weights_only=True))

    @torch.no_grad()
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """每行的车内概率；不填充缺失值（训练时选择了丢弃）且该行有缺失时为 NaN。"""
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise ValueError(f"数据中没有特征列：{'、'.join(missing)}")
        x = np.column_stack([to_numeric(df[f]).to_numpy(dtype=np.float64) for f in self.features])
        if self.fill_values is not None:
            x = np.where(np.isnan(x), self.fill_values, x)
        if self.mean is not None:
            x = (x - self.mean) / self.std
        valid = ~np.isnan(x).any(axis=1)
        prob = np.full(len(x), np.nan)
        if valid.any():
            logits = self.model(torch.from_numpy(x[valid].astype(np.float32)))
            prob[valid] = torch.softmax(logits, dim=1)[:, 1].numpy()
        return prob


def _array(values) -> np.ndarray | None:
    return None if values is None else np.asarray(values, dtype=np.float64)


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    return value
