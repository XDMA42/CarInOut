"""读取数据集构建工具导出的总表，选择特征、处理缺失值、划分训练 / 验证 / 测试集并标准化。

总表格式（见 merger.build_dataset）：ts, <原列名>_<锚点名>..., scene, point, label。
ts / ts_* / errcode_* 以及 scene、point、label 不作为特征；其余能解析为数值的列都是候选特征。

验证集、测试集可以从训练数据中划分，也可以指定独立的数据（多个文件夹，每个文件夹一个 _dataset.csv），
每行都记录来自哪个文件夹（source 列），便于分别统计各次采集的表现。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..merger import (
    COMBINED_FILENAME,
    LABEL_INSIDE,
    LABEL_OUTSIDE,
    POINT_COLUMN,
    SCENE_COLUMN,
    TS_COLUMN,
    CsvReadError,
    read_csv_text,
)
from ..parser import natural_key
from .config import MISSING_CONSTANT, MISSING_DROP, MISSING_MEAN, SPLIT_POINT, DataConfig

NUM_CLASSES = 2
CLASS_NAMES = {LABEL_OUTSIDE: "车外", LABEL_INSIDE: "车内"}
SPLITS = ("train", "val", "test")
SPLIT_NAMES = {"train": "训练集", "val": "验证集", "test": "测试集"}

SOURCE_COLUMN = "source"  # 数据来自哪个文件夹（_dataset.csv 所在文件夹名）
META_COLUMNS = (SOURCE_COLUMN, TS_COLUMN, SCENE_COLUMN, POINT_COLUMN)  # 随预测结果一起保存，便于按点位分析
EXCLUDED_PREFIXES = (f"{TS_COLUMN}_", "errcode_")


@dataclass(frozen=True)
class FeatureInfo:
    name: str
    missing_ratio: float  # 空值或非数值所占比例
    constant: bool  # 所有非空值都相同，对模型没有信息量


@dataclass
class Split:
    x: np.ndarray  # (样本数, 特征数) float32，已填充缺失值并标准化
    y: np.ndarray  # (样本数,) int64
    meta: pd.DataFrame  # source / ts / scene / point（存在时），与 x 逐行对应

    def __len__(self) -> int:
        return len(self.y)


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        std = x.std(axis=0)
        return cls(x.mean(axis=0), np.where(std > 0, std, 1.0))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


@dataclass
class PreparedData:
    features: list[str]
    train: Split
    val: Split
    test: Split
    fill_values: np.ndarray | None  # 每个特征的缺失填充值；missing == drop 时为 None
    scaler: Standardizer | None
    notes: list[str] = field(default_factory=list)  # 划分情况与提示，显示在日志中
    val_files: list[str] = field(default_factory=list)  # 独立验证集的文件（已展开文件夹）
    test_files: list[str] = field(default_factory=list)

    def split(self, name: str) -> Split:
        return getattr(self, name)


def resolve_datasets(entries: list[str]) -> list[Path]:
    """把独立验证集 / 测试集的各项展开为 csv 文件列表（去重）。

    含 _dataset.csv 的文件夹 → 该文件；不含的文件夹 → 各子文件夹中的 _dataset.csv（如数据集构建工具的
    merged 文件夹）；csv 文件 → 本身。
    """
    files: list[Path] = []
    for entry in entries:
        path = Path(entry)
        if path.is_file():
            found = [path]
        elif (path / COMBINED_FILENAME).is_file():
            found = [path / COMBINED_FILENAME]
        elif path.is_dir():
            subdirs = sorted((d for d in path.iterdir() if d.is_dir()), key=lambda d: natural_key(d.name))
            found = [d / COMBINED_FILENAME for d in subdirs if (d / COMBINED_FILENAME).is_file()]
            if not found:
                raise ValueError(f"{path} 及其子文件夹中都没有 {COMBINED_FILENAME}")
        else:
            raise ValueError(f"找不到 {path}")
        seen = {f.resolve() for f in files}
        files.extend(f for f in found if f.resolve() not in seen)
    return files


def dataset_name(path: str | Path) -> str:
    """数据的简称：_dataset.csv 取所在文件夹名（如 “1023采集”），其他 csv 取文件名。"""
    path = Path(path)
    return path.parent.name if path.name == COMBINED_FILENAME else path.stem


def load_table(path: str | Path) -> pd.DataFrame:
    """以文本方式读取 CSV（UTF-8 / GB18030），所有值保持字符串。"""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"找不到数据文件：{path}")
    try:
        return read_csv_text(path)
    except CsvReadError as exc:
        raise ValueError(f"读取 {path.name} 失败：{exc}") from exc


def to_numeric(series: pd.Series) -> pd.Series:
    """文本列转为 float，空值和无法解析的值为 NaN。"""
    return pd.to_numeric(series.astype(str).str.strip().replace("", np.nan), errors="coerce")


def feature_candidates(df: pd.DataFrame, label_column: str) -> list[FeatureInfo]:
    """列出可作为特征的数值列（按列顺序）。整列为空或含非数值文本的列不算。"""
    excluded = {c.lower() for c in (*META_COLUMNS, label_column)}
    result = []
    for col in df.columns:
        if col.lower() in excluded or col.lower().startswith(EXCLUDED_PREFIXES):
            continue
        text = df[col].astype(str).str.strip()
        filled = text != ""
        values = to_numeric(df[col])
        if not filled.any() or values[filled].isna().any():
            continue
        result.append(FeatureInfo(col, float(values.isna().mean()), values.nunique(dropna=True) <= 1))
    return result


def default_features(infos: list[FeatureInfo]) -> list[str]:
    return [f.name for f in infos if not f.constant]


def parse_labels(df: pd.DataFrame, label_column: str) -> pd.Series:
    if label_column not in df.columns:
        raise ValueError(f"数据中没有标签列 “{label_column}”")
    labels = to_numeric(df[label_column])
    bad = labels.notna() & ~labels.isin(list(CLASS_NAMES))
    if bad.any():
        sample = df[label_column][bad].iat[0]
        raise ValueError(f"标签列 “{label_column}” 只能是 0（车外）或 1（车内），发现 “{sample}”")
    return labels


def split_indices(
    labels: np.ndarray,
    groups: np.ndarray | None,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按标签分层划分，返回 (训练, 验证, 测试) 的行号。

    groups 为 None 时按行随机划分；否则同一组（点位）的行整体分到同一个集合：
    组随机排序后依次排开，每组按其中点落在 [0, 测试比例) / [测试比例, 测试+验证比例) / 其余 的位置归入
    测试 / 验证 / 训练集，使各集合的行数比例接近设定值。每个标签至少留一组给训练集。
    """
    rng = np.random.default_rng(seed)
    parts: tuple[list, list, list] = ([], [], [])
    for label in np.unique(labels):
        rows = np.flatnonzero(labels == label)
        if groups is None:
            rows = rng.permutation(rows)
            n_test = int(round(len(rows) * test_ratio))
            n_val = int(round(len(rows) * val_ratio))
            if n_test + n_val >= len(rows):  # 样本极少时保证训练集至少有一行
                n_val = max(0, len(rows) - 1 - n_test)
                n_test = min(n_test, len(rows) - 1)
            parts[2].append(rows[:n_test])
            parts[1].append(rows[n_test : n_test + n_val])
            parts[0].append(rows[n_test + n_val :])
            continue
        names, inverse = np.unique(groups[rows], return_inverse=True)
        order = rng.permutation(len(names))
        sizes = np.bincount(inverse, minlength=len(names))
        target = np.empty(len(names), dtype=int)  # 0 训练 / 1 验证 / 2 测试
        done = 0
        for g in order:
            mid = (done + sizes[g] / 2) / len(rows)
            target[g] = 2 if mid < test_ratio else 1 if mid < test_ratio + val_ratio else 0
            done += sizes[g]
        if not (target == 0).any():
            target[order[-1]] = 0
        for k in range(3):
            parts[k].append(rows[np.isin(inverse, np.flatnonzero(target == k))])
    return tuple(np.sort(np.concatenate(p)).astype(np.int64) if p else np.array([], np.int64) for p in parts)


def prepare_data(cfg: DataConfig, seed: int, log: Callable[[str], None] | None = None) -> PreparedData:
    """读取、清洗、划分、标准化。所有统计量（均值填充、标准化）只用训练集计算。"""
    log = log or (lambda _msg: None)
    if not cfg.dataset:
        raise ValueError("请先选择数据集文件（数据集构建工具导出的 _dataset.csv）")
    notes: list[str] = []
    main_path = Path(cfg.dataset)
    extra = {"val": resolve_datasets(cfg.val_datasets), "test": resolve_datasets(cfg.test_datasets)}
    _check_overlap(main_path, extra)
    df = load_table(main_path)
    parse_labels(df, cfg.label_column)  # 先检查标签列，报错更直接
    infos = feature_candidates(df, cfg.label_column)
    features = list(cfg.features) or default_features(infos)
    if not features:
        raise ValueError("数据中没有可用的数值特征列")
    _check_features(df, features, dataset_name(main_path))

    frame = _clean(df, features, cfg, notes, main_path)
    labels = frame["__label"].to_numpy(dtype=np.int64)
    use_groups = cfg.split == SPLIT_POINT
    if use_groups and not {SCENE_COLUMN, POINT_COLUMN} <= set(frame.columns):
        notes.append(f"数据中缺少 {SCENE_COLUMN} / {POINT_COLUMN} 列，改为按行随机划分")
        use_groups = False
    groups = (frame[SCENE_COLUMN] + "\x1f" + frame[POINT_COLUMN]).to_numpy() if use_groups else None
    val_ratio = 0.0 if extra["val"] else cfg.val_ratio
    test_ratio = 0.0 if extra["test"] else cfg.test_ratio
    idx_train, idx_val, idx_test = split_indices(labels, groups, val_ratio, test_ratio, seed)

    frames = {"train": frame.iloc[idx_train], "val": frame.iloc[idx_val], "test": frame.iloc[idx_test]}
    for name, files in extra.items():
        if files:
            parts = []
            for path in files:
                extra_df = load_table(path)
                _check_features(extra_df, features, dataset_name(path))
                parts.append(_clean(extra_df, features, cfg, notes, path))
            frames[name] = pd.concat(parts, ignore_index=True)

    train_x = frames["train"][features].to_numpy(dtype=np.float64)
    if len(np.unique(frames["train"]["__label"])) < NUM_CLASSES:
        raise ValueError("训练集中只有一个类别，请检查数据集的 label 列或调整划分比例")

    fill_values = None
    if cfg.missing == MISSING_MEAN:
        means = np.nanmean(np.where(np.isnan(train_x).all(axis=0), 0.0, train_x), axis=0)
        fill_values = np.nan_to_num(means)
    elif cfg.missing == MISSING_CONSTANT:
        fill_values = np.full(len(features), cfg.fill_value, dtype=np.float64)
    scaler = Standardizer.fit(_fill(train_x, fill_values)) if cfg.standardize else None

    splits = {}
    for name, part in frames.items():
        x = _fill(part[features].to_numpy(dtype=np.float64), fill_values)
        if scaler is not None:
            x = scaler.transform(x)
        meta = part[[c for c in META_COLUMNS if c in part.columns]].reset_index(drop=True)
        labels = part["__label"].to_numpy(dtype=np.int64, copy=True)  # torch 需要可写数组
        splits[name] = Split(x.astype(np.float32), labels, meta)

    notes.extend(_describe_splits(splits, groups is not None, {k for k, files in extra.items() if files}))
    for note in notes:
        log(note)
    return PreparedData(
        features, splits["train"], splits["val"], splits["test"], fill_values, scaler, notes,
        val_files=[str(p) for p in extra["val"]], test_files=[str(p) for p in extra["test"]],
    )


def _check_overlap(main_path: Path, extra: dict[str, list[Path]]) -> None:
    """同一份数据不能既用于训练又用于评估，也不能同时出现在验证集和测试集中。"""
    main = main_path.resolve()
    for name, files in extra.items():
        if main in {f.resolve() for f in files}:
            raise ValueError(f"{SPLIT_NAMES[name]}中包含训练用的数据集 {dataset_name(main_path)}，评估结果会偏乐观")
    both = {f.resolve() for f in extra["val"]} & {f.resolve() for f in extra["test"]}
    if both:
        names = "、".join(sorted(dataset_name(p) for p in both))
        raise ValueError(f"{names} 同时出现在验证集和测试集中")


def _check_features(df: pd.DataFrame, features: list[str], source: str) -> None:
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"{source} 中没有特征列：{'、'.join(missing)}")
    # 整列为空是允许的（如独立测试集当天缺某个锚点），按缺失值处理
    bad = [f for f in features if (to_numeric(df[f]).isna() & (df[f].astype(str).str.strip() != "")).any()]
    if bad:
        raise ValueError(f"{source} 中以下列不是数值，不能作为特征：{'、'.join(bad)}")


def _clean(df: pd.DataFrame, features: list[str], cfg: DataConfig, notes: list[str], path: Path) -> pd.DataFrame:
    """取出特征（float）、标签与元信息列，丢弃无标签的行；missing == drop 时丢弃含缺失值的行。"""
    source = dataset_name(path)
    labels = parse_labels(df, cfg.label_column)
    out = pd.DataFrame({f: to_numeric(df[f]) for f in features})
    out["__label"] = labels
    for col in META_COLUMNS:
        if col in df.columns and col not in out.columns:
            out[col] = df[col].astype(str)
    out[SOURCE_COLUMN] = source
    unlabeled = int(labels.isna().sum())
    if unlabeled:
        notes.append(f"{source}：{unlabeled} 行没有标签，已忽略")
    out = out[labels.notna()]
    if cfg.missing == MISSING_DROP:
        has_nan = out[features].isna().any(axis=1)
        if has_nan.any():
            notes.append(f"{source}：{int(has_nan.sum())} 行特征有缺失值，已丢弃（可改为按均值或常数填充）")
        out = out[~has_nan]
    if out.empty:
        raise ValueError(f"{source} 清洗后没有可用的数据行")
    out["__label"] = out["__label"].astype(np.int64)
    return out.reset_index(drop=True)


def _fill(x: np.ndarray, fill_values: np.ndarray | None) -> np.ndarray:
    if fill_values is None:
        return x
    return np.where(np.isnan(x), fill_values, x)


def _describe_splits(splits: dict[str, Split], by_group: bool, independent: set[str]) -> list[str]:
    total = sum(len(s) for s in splits.values()) or 1
    lines = ["数据划分：" + ("按点位整组划分" if by_group else "按行随机划分")]
    for name, s in splits.items():
        origin = "，独立数据" if name in independent else ""
        lines.append(f"  {SPLIT_NAMES[name]}：{_counts(s.meta, s.y)}（占 {len(s) / total:.0%}{origin}）")
        if SOURCE_COLUMN in s.meta and s.meta[SOURCE_COLUMN].nunique() > 1:
            for source, rows in s.meta.groupby(SOURCE_COLUMN, sort=False).indices.items():
                lines.append(f"    {source}：{_counts(s.meta.iloc[rows], s.y[rows])}")
        if name != "train" and len(s) and len(np.unique(s.y)) < NUM_CLASSES:
            lines.append(f"  注意：{SPLIT_NAMES[name]}只有一个类别，AUC 等指标无法计算（点位太少时可改为按行划分）")
    return lines


def _counts(meta: pd.DataFrame, y: np.ndarray) -> str:
    text = f"{len(y)} 行，" + "，".join(f"{CLASS_NAMES[c]} {int((y == c).sum())}" for c in CLASS_NAMES)
    keys = [c for c in (SOURCE_COLUMN, SCENE_COLUMN, POINT_COLUMN) if c in meta.columns]
    if {SCENE_COLUMN, POINT_COLUMN} <= set(keys) and len(y):
        text += f"，{meta.groupby(keys).ngroups} 个点位"
    return text
