#!/usr/bin/env python3
"""车内外识别：多种 MCU 友好模型的超参搜索与准确率对比。

输入训练集、验证集两个 CSV（特征列 + label + scene），对每种模型：
  1. 在训练集上做超参搜索（默认按 scene 分组交叉验证，避免同一场景的数据同时出现在训练折和验证折）；
  2. 用最优超参在整个训练集上重训，在验证集上报告准确率；
  3. 粗略估算部署到 MCU（float32 实现）时的 Flash / RAM 占用。

用法见 README.md。
"""
import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    ParameterGrid,
    ParameterSampler,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

MODEL_NAMES = ["logreg", "tree", "rf", "lgbm", "mlp"]

# 搜索空间：按需增删。取值已按 MCU 部署（Flash / RAM）收窄，调参方法见 README。
SEARCH_SPACES = {
    "logreg": {
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__class_weight": [None, "balanced"],
    },
    "tree": {
        "clf__max_depth": [3, 4, 5, 6, 8, 10],
        "clf__min_samples_leaf": [1, 5, 20, 50],
        "clf__class_weight": [None, "balanced"],
    },
    "rf": {
        "clf__n_estimators": [10, 20, 50],
        "clf__max_depth": [4, 6, 8],
        "clf__min_samples_leaf": [1, 5, 20],
        "clf__max_features": ["sqrt", 0.5, 1.0],
    },
    "lgbm": {
        "clf__n_estimators": [30, 50, 100, 200],
        "clf__learning_rate": [0.05, 0.1, 0.2],
        "clf__max_depth": [3, 4, 5],
        "clf__num_leaves": [7, 15],
        "clf__min_child_samples": [10, 30, 100],
        "clf__colsample_bytree": [0.7, 1.0],
        "clf__reg_lambda": [0.0, 1.0],
    },
    "mlp": {
        "clf__hidden_layer_sizes": [(8,), (16,), (32,), (16, 8), (16, 16), (32, 16)],
        "clf__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
        "clf__learning_rate_init": [1e-3, 3e-3, 1e-2],
    },
}

FLOAT_BYTES = 4
# 树按数组存储时每个节点的估算大小：float32 阈值/叶子值 + uint8 特征下标 + 2 个 int16 子节点下标
BYTES_PER_TREE_NODE = 10


@dataclass
class Task:
    name: str  # "label"：用原始标签训练；"inout"：直接训练 车内/车外 二分类（对照组）
    classes: list
    y_train: np.ndarray
    y_val: np.ndarray
    inside_idx: int | None  # P(车内) 所在的列；None 表示没有指定车内标签


def build_model(name, seed):
    # 缺失值统一用中位数填充并追加缺失标志列，MCU 上按同样规则处理即可
    imp = ("imp", SimpleImputer(strategy="median", add_indicator=True))
    if name == "logreg":
        steps = [imp, ("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=5000))]
    elif name == "tree":
        steps = [imp, ("clf", DecisionTreeClassifier(random_state=seed))]
    elif name == "rf":
        steps = [imp, ("clf", RandomForestClassifier(random_state=seed, n_jobs=1))]
    elif name == "lgbm":
        steps = [imp, ("clf", LGBMClassifier(random_state=seed, n_jobs=1, verbose=-1))]
    elif name == "mlp":
        steps = [
            imp,
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(max_iter=1000, early_stopping=True, random_state=seed)),
        ]
    else:
        raise ValueError(name)
    return Pipeline(steps)


def estimate_footprint(model, n_raw_features):
    """粗略估算 float32 实现的 Flash / RAM（字节）。RAM 只算推理工作内存，不含原始数据滑窗。"""
    n_in = model[:-1].transform(np.zeros((1, n_raw_features))).shape[1]
    clf = model[-1]
    flash = n_raw_features * FLOAT_BYTES  # 缺失值填充用的中位数
    if "scale" in model.named_steps:
        flash += 2 * n_in * FLOAT_BYTES
    ram = n_in * FLOAT_BYTES  # 输入特征向量
    if isinstance(clf, LogisticRegression):
        flash += (clf.coef_.size + clf.intercept_.size) * FLOAT_BYTES
        ram += clf.coef_.shape[0] * FLOAT_BYTES
    elif isinstance(clf, MLPClassifier):
        flash += sum(w.size for w in clf.coefs_ + clf.intercepts_) * FLOAT_BYTES
        # 逐层计算只需同时保留相邻两层的激活值（输入向量已包含在第一层里）
        widths = [n_in] + [w.shape[1] for w in clf.coefs_]
        ram = max(a + b for a, b in zip(widths, widths[1:])) * FLOAT_BYTES
    elif isinstance(clf, DecisionTreeClassifier):
        flash += clf.tree_.node_count * BYTES_PER_TREE_NODE
    elif isinstance(clf, RandomForestClassifier):
        flash += sum(t.tree_.node_count for t in clf.estimators_) * BYTES_PER_TREE_NODE
        ram += len(clf.classes_) * FLOAT_BYTES  # 各类投票累加
    else:  # LightGBM；多分类时每轮迭代每个类各一棵树
        trees = clf.booster_.dump_model()["tree_info"]
        flash += sum(2 * t["num_leaves"] - 1 for t in trees) * BYTES_PER_TREE_NODE
        ram += len(clf.classes_) * FLOAT_BYTES  # 各类得分累加
    return {"flash_bytes": int(flash), "ram_bytes": int(ram)}


def predict_proba_full(model, X, n_classes):
    # 某折训练数据可能缺少个别类别，按 classes_ 把概率放回完整的列
    proba = np.zeros((len(X), n_classes))
    proba[:, model.classes_] = model.predict_proba(X)
    return proba


def compute_metrics(proba, y_true, inside_idx, threshold):
    metrics = {"acc": float(np.mean(proba.argmax(axis=1) == y_true))}
    if inside_idx is not None:
        # 用 P(车内) 与阈值比较判断车内/车外（多分类时 P(车外) 即其余各类之和），不要 argmax 后再映射
        pred_in = proba[:, inside_idx] >= threshold
        true_in = y_true == inside_idx
        metrics["inout_acc"] = float(np.mean(pred_in == true_in))
        # 车外误判为车内：安全问题；车内误判为车外：体验问题
        metrics["false_inside"] = float(np.mean(pred_in[~true_in])) if (~true_in).any() else np.nan
        metrics["false_outside"] = float(np.mean(~pred_in[true_in])) if true_in.any() else np.nan
    return metrics


def fit(base, params, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return clone(base).set_params(**params).fit(X, y)


def fit_and_score(base, params, X_fit, y_fit, X_eval, y_eval, n_classes, inside_idx, threshold):
    model = fit(base, params, X_fit, y_fit)
    proba = predict_proba_full(model, X_eval, n_classes)
    metrics = compute_metrics(proba, y_eval, inside_idx, threshold)
    metrics.update(estimate_footprint(model, X_fit.shape[1]))
    return metrics


def candidate_params(space, args):
    grid = ParameterGrid(space)
    if args.search == "grid" or len(grid) <= args.n_iter:
        return list(grid)
    return list(ParameterSampler(space, n_iter=args.n_iter, random_state=args.seed))


def run_search(name, task, X_tr, X_val, splits, metric, args):
    base = build_model(name, args.seed)
    cands = candidate_params(SEARCH_SPACES[name], args)
    if splits is None:  # --cv val：直接在验证集上挑参数
        folds = [(X_tr, task.y_train, X_val, task.y_val)]
    else:
        folds = [(X_tr[a], task.y_train[a], X_tr[b], task.y_train[b]) for a, b in splits]
    print(f"[{name}/{task.name}] {len(cands)} 组参数 x {len(folds)} 折 ...", flush=True)

    t0 = time.time()
    jobs = [
        delayed(fit_and_score)(base, p, *f, len(task.classes), task.inside_idx, args.threshold)
        for p in cands
        for f in folds
    ]
    results = Parallel(n_jobs=args.n_jobs)(jobs)

    rows = []
    for i, params in enumerate(cands):
        fold_res = pd.DataFrame(results[i * len(folds) : (i + 1) * len(folds)])
        row = {k.removeprefix("clf__"): str(v) for k, v in params.items()}
        row.update({f"search_{c}": v for c, v in fold_res.mean().items()})
        row["search_score_std"] = fold_res[metric].std(ddof=0)
        row["_params"] = params
        rows.append(row)
    log = pd.DataFrame(rows).sort_values(
        [f"search_{metric}", "search_flash_bytes"], ascending=[False, True]
    )
    if args.max_flash_kb is not None:
        log = log[log["search_flash_bytes"] <= args.max_flash_kb * 1024]
        if log.empty:
            print(f"  所有参数组合的 Flash 都超过 {args.max_flash_kb} KB，跳过（可在搜索空间里减小树的数量/深度或网络宽度）")
            return None

    best = log.iloc[0]
    model = fit(base, best["_params"], X_tr, task.y_train)
    proba = predict_proba_full(model, X_val, len(task.classes))
    return {
        "model": model,
        "log": log.drop(columns="_params"),
        "best_params": best["_params"],
        "n_cand": len(cands),
        "search_score": best[f"search_{metric}"],
        "search_score_std": best["search_score_std"],
        "proba": proba,
        "val": compute_metrics(proba, task.y_val, task.inside_idx, args.threshold),
        "footprint": estimate_footprint(model, X_tr.shape[1]),
        "time_s": time.time() - t0,
    }


def read_csv(path):
    for enc in ("utf-8-sig", "gbk"):  # 兼容 Excel 导出的带 BOM UTF-8 和 GBK
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"无法识别 {path} 的编码，请转成 UTF-8")


def to_feature_matrix(df, features, missing_values, name):
    absent = [c for c in features if c not in df.columns]
    if absent:
        raise SystemExit(f"{name}缺少特征列：{absent}")
    bad = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    if bad:
        raise SystemExit(
            f"{name}中这些特征列不是数值：{bad}。用 --drop-cols 排除，或用 --feature-cols 指定特征列"
        )
    X = df[features].to_numpy(dtype=np.float64)
    for v in missing_values:
        X[X == v] = np.nan
    return X


def load_data(args):
    train, val = read_csv(args.train), read_csv(args.val)
    for name, df in (("训练集", train), ("验证集", val)):
        if args.label_col not in df.columns:
            raise SystemExit(f"{name}缺少标签列 {args.label_col!r}")
        if df[args.label_col].isna().any():
            raise SystemExit(f"{name}的标签列有空值")

    if args.feature_cols:
        features = args.feature_cols
    else:
        exclude = {args.label_col, args.scene_col, *args.drop_cols}
        features = [c for c in train.columns if c not in exclude]
    X_tr = to_feature_matrix(train, features, args.missing_values, "训练集")
    X_val = to_feature_matrix(val, features, args.missing_values, "验证集")

    y_tr_raw = train[args.label_col].astype(str).str.strip().to_numpy()
    y_val_raw = val[args.label_col].astype(str).str.strip().to_numpy()
    classes = sorted(set(y_tr_raw))
    unknown = set(y_val_raw) - set(classes)
    if unknown:
        raise SystemExit(f"验证集中有训练集没有的标签：{sorted(unknown)}")
    code = {c: i for i, c in enumerate(classes)}
    y_tr = np.array([code[c] for c in y_tr_raw])
    y_val = np.array([code[c] for c in y_val_raw])

    if args.scene_col in train.columns and args.scene_col in val.columns:
        g_tr = train[args.scene_col].astype(str).to_numpy()
        g_val = val[args.scene_col].astype(str).to_numpy()
    else:
        print(f"警告：没有场景列 {args.scene_col!r}，交叉验证改为不分组的分层划分，也不输出分场景准确率")
        g_tr = g_val = None

    inside_idx = None
    if args.inside_label is not None:
        if args.inside_label not in code:
            raise SystemExit(f"--inside-label {args.inside_label!r} 不在训练集标签 {classes} 中")
        inside_idx = code[args.inside_label]
    tasks = [Task("label", classes, y_tr, y_val, inside_idx)]
    if inside_idx is not None and len(classes) > 2 and not args.no_inout_baseline:
        tasks.append(
            Task(
                "inout",
                ["outside", args.inside_label],
                (y_tr == inside_idx).astype(int),
                (y_val == inside_idx).astype(int),
                1,
            )
        )

    print(f"训练集 {len(X_tr)} 条，验证集 {len(X_val)} 条，特征 {len(features)} 个")
    print(f"缺失值比例：训练集 {np.isnan(X_tr).mean():.1%}，验证集 {np.isnan(X_val).mean():.1%}")
    counts = pd.DataFrame(
        {"train": pd.Series(y_tr_raw).value_counts(), "val": pd.Series(y_val_raw).value_counts()}
    )
    print("标签分布：\n" + counts.fillna(0).astype(int).to_string())
    if g_tr is not None:
        print(f"场景：训练集 {len(set(g_tr))} 个，验证集 {len(set(g_val))} 个")
    return X_tr, X_val, y_tr, g_tr, g_val, tasks


def make_splits(args, y, groups):
    if args.cv == "val":
        print("注意：--cv val 直接在验证集上挑参数，报告的验证集准确率会偏乐观")
        return None
    if args.cv == "group" and groups is not None:
        k = min(args.cv_folds, len(set(groups)))
        if k < 2:
            raise SystemExit("训练集只有 1 个场景，无法分组交叉验证，请改用 --cv stratified 或 --cv val")
        if k < args.cv_folds:
            print(f"训练集只有 {k} 个场景，分组交叉验证折数降为 {k}")
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=args.seed)
        return list(cv.split(np.zeros(len(y)), y, groups))
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    return list(cv.split(np.zeros(len(y)), y))


def per_scene_metrics(res, task, groups, threshold):
    rows = []
    for scene in sorted(set(groups)):
        mask = groups == scene
        m = compute_metrics(res["proba"][mask], task.y_val[mask], task.inside_idx, threshold)
        rows.append({"scene": scene, "n": int(mask.sum()), **m})
    return rows


def parse_args():
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()]
    p = argparse.ArgumentParser(description="车内外识别：多模型超参搜索与准确率对比")
    p.add_argument("--train", required=True, help="训练集 CSV")
    p.add_argument("--val", required=True, help="验证集 CSV")
    p.add_argument("--label-col", default="label")
    p.add_argument("--scene-col", default="scene")
    p.add_argument(
        "--inside-label",
        help="代表车内的标签值，例如 车内。指定后报告车内/车外准确率；多分类时按 P(车内) 合并",
    )
    p.add_argument("--feature-cols", type=split, default=[], help="逗号分隔；默认用除标签和场景外的所有列")
    p.add_argument("--drop-cols", type=split, default=[], help="逗号分隔，不作为特征的列（如时间戳、ID）")
    p.add_argument(
        "--missing-values", type=float, nargs="*", default=[], help="特征中表示缺失的哨兵值，如 -1"
    )
    p.add_argument("--models", type=split, default=MODEL_NAMES, help=f"逗号分隔，可选 {MODEL_NAMES}")
    p.add_argument(
        "--search", choices=["random", "grid"], default="random", help="random：随机抽 --n-iter 组；grid：全部遍历"
    )
    p.add_argument("--n-iter", type=int, default=40)
    p.add_argument(
        "--cv",
        choices=["group", "stratified", "val"],
        default="group",
        help="group：训练集按场景分组交叉验证（推荐）；stratified：不分组；val：直接用验证集挑参数",
    )
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument(
        "--select-metric",
        choices=["auto", "inout_acc", "acc"],
        default="auto",
        help="挑参数用的指标；auto：指定了 --inside-label 时用 inout_acc，否则用 acc",
    )
    p.add_argument("--threshold", type=float, default=0.5, help="P(车内) >= 阈值 判为车内")
    p.add_argument("--max-flash-kb", type=float, help="只在估算 Flash 不超过该值的参数组合中挑选")
    p.add_argument("--no-inout-baseline", action="store_true", help="多分类标签时，不额外训练车内/车外二分类对照组")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="results")
    return p.parse_args()


def main():
    args = parse_args()
    unknown = set(args.models) - set(MODEL_NAMES)
    if unknown:
        raise SystemExit(f"未知模型：{sorted(unknown)}，可选 {MODEL_NAMES}")
    if "lgbm" in args.models and LGBMClassifier is None:
        print("警告：未安装 lightgbm，跳过 lgbm（pip install lightgbm）")
        args.models = [m for m in args.models if m != "lgbm"]

    X_tr, X_val, y_tr, g_tr, g_val, tasks = load_data(args)
    if args.select_metric == "auto":
        metric = "inout_acc" if tasks[0].inside_idx is not None else "acc"
    elif args.select_metric == "inout_acc" and tasks[0].inside_idx is None:
        raise SystemExit("--select-metric inout_acc 需要同时指定 --inside-label")
    else:
        metric = args.select_metric
    splits = make_splits(args, y_tr, g_tr)  # 两种任务共用同一套划分，保证对比公平

    out = Path(args.out_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    summary, scene_rows, best_params, confusions = [], [], {}, {}
    for task in tasks:
        for name in args.models:
            res = run_search(name, task, X_tr, X_val, splits, metric, args)
            if res is None:
                continue
            key = f"{name}/{task.name}"
            val = res["val"]
            summary.append(
                {
                    "model": name,
                    "task": task.name,
                    "n_cand": res["n_cand"],
                    f"search_{metric}": res["search_score"],
                    "search_std": res["search_score_std"],
                    "val_acc": val["acc"] if task.name == "label" else np.nan,
                    "val_inout_acc": val.get("inout_acc", np.nan),
                    "val_false_inside": val.get("false_inside", np.nan),
                    "val_false_outside": val.get("false_outside", np.nan),
                    "flash_kb": res["footprint"]["flash_bytes"] / 1024,
                    "ram_b": res["footprint"]["ram_bytes"],
                    "time_s": res["time_s"],
                }
            )
            best_params[key] = {k.removeprefix("clf__"): v for k, v in res["best_params"].items()}
            print(f"  最优参数 {best_params[key]}  验证集 {metric}={val[metric]:.4f}")

            res["log"].round(4).to_csv(out / f"search_{name}_{task.name}.csv", index=False)
            joblib.dump(res["model"], out / "models" / f"{name}_{task.name}.joblib")
            pred = res["proba"].argmax(axis=1)
            cm = pd.crosstab(
                pd.Series(np.array(task.classes)[task.y_val], name="true"),
                pd.Series(np.array(task.classes)[pred], name="pred"),
            )
            cm.to_csv(out / f"confusion_{name}_{task.name}.csv")
            confusions[key] = cm
            if g_val is not None:
                for row in per_scene_metrics(res, task, g_val, args.threshold):
                    scene_rows.append({"model": name, "task": task.name, **row})

    if not summary:
        raise SystemExit("没有可用的模型结果")
    sort_col = "val_inout_acc" if metric == "inout_acc" else "val_acc"
    summary = pd.DataFrame(summary).sort_values(sort_col, ascending=False)
    summary.round(4).to_csv(out / "summary.csv", index=False)
    (out / "best_params.json").write_text(
        json.dumps(best_params, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    with pd.option_context("display.width", 200, "display.max_columns", None):
        print("\n==== 验证集结果（按 %s 排序）====" % sort_col)
        print(summary.dropna(axis=1, how="all").to_string(index=False, float_format="%.4f"))
        if scene_rows:
            scenes = pd.DataFrame(scene_rows)
            scenes.round(4).to_csv(out / "per_scene.csv", index=False)
            scenes["model_task"] = scenes["model"] + "/" + scenes["task"]
            pivot = scenes.pivot_table(index=["scene", "n"], columns="model_task", values=metric)
            print(f"\n==== 验证集分场景 {metric} ====")
            print(pivot.to_string(float_format="%.4f"))
        top = summary.iloc[0]
        print(f"\n==== 最优模型 {top['model']}/{top['task']} 的验证集混淆矩阵 ====")
        print(confusions[f"{top['model']}/{top['task']}"].to_string())
    print(f"\n结果已保存到 {out}/")


if __name__ == "__main__":
    main()
