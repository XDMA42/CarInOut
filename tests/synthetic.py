"""生成与数据集构建工具总表 _dataset.csv 同格式的模拟数据，供测试使用。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from carinout.anchors import ANCHOR_SLOTS

CAR_W, CAR_L = 1.9, 4.8  # 米


def make_dataset(
    path: Path,
    n_inside: int = 8,
    n_outside: int = 8,
    rows_per_point: int = 30,
    seed: int = 0,
    missing_rate: float = 0.0,
    noise: float = 3.0,
    body_loss: float = 6.0,
) -> Path:
    """车内点位 label=1，车外点位 label=0；RSSI 按距离衰减，信号穿过车身时额外衰减。"""
    rng = np.random.default_rng(seed)
    anchors = [(s.default_name, s.x * CAR_W, s.y * CAR_L, s.inside) for s in ANCHOR_SLOTS]
    frames = []
    for scene, label, count in (("车内座椅", 1, n_inside), ("车外一圈", 0, n_outside)):
        for k in range(count):
            if label:
                px, py = rng.uniform(0.2, CAR_W - 0.2), rng.uniform(0.6, CAR_L - 0.6)
            else:
                angle, dist = rng.uniform(0, 2 * np.pi), rng.uniform(0.3, 2.5)
                px = CAR_W / 2 + np.cos(angle) * (CAR_W / 2 + dist)
                py = CAR_L / 2 + np.sin(angle) * (CAR_L / 2 + dist)
            ts = np.arange(rows_per_point) * 100 + 1000 * k
            row = {"ts": ts.astype(str)}
            for name, ax, ay, inside in anchors:
                d = np.hypot(px - ax, py - ay)
                body = body_loss if bool(label) != inside else 0.0
                rssi = -45 - 20 * np.log10(d + 0.3) - body + rng.normal(0, noise, rows_per_point)
                text = np.char.mod("%.1f", rssi).astype(object)
                if missing_rate:
                    text[rng.random(rows_per_point) < missing_rate] = ""
                row[f"rssi_{name}"] = text
                row[f"errcode_{name}"] = "0"
            frame = pd.DataFrame(row)
            frame["scene"], frame["point"], frame["label"] = scene, str(k + 1), str(label)
            frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")
    return path
