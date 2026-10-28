import math

import numpy as np

from carinout.training.metrics import binary_metrics, confusion_matrix, roc_auc


def test_confusion_and_scores():
    y = np.array([0, 0, 0, 1, 1, 1, 1])
    prob = np.array([0.1, 0.6, 0.2, 0.9, 0.4, 0.8, 0.7])
    m = binary_metrics(y, prob)
    assert m["confusion"] == [[2, 1], [1, 3]]
    assert m["accuracy"] == 5 / 7
    assert m["precision"] == 3 / 4 and m["recall"] == 3 / 4 and m["f1"] == 3 / 4
    assert confusion_matrix([1, 0], [0, 0]).tolist() == [[1, 0], [1, 0]]


def test_auc_matches_pairwise_definition():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    score = np.round(rng.random(200), 1)  # 大量并列分数
    pos, neg = score[y == 1], score[y == 0]
    pairs = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
    assert math.isclose(roc_auc(y, score), pairs)
    assert roc_auc([0, 1], [0.2, 0.8]) == 1.0
    assert roc_auc([0, 1], [0.8, 0.2]) == 0.0
    assert roc_auc([1, 0], [0.5, 0.5]) == 0.5


def test_degenerate_cases():
    assert math.isnan(roc_auc([1, 1], [0.2, 0.8]))
    m = binary_metrics(np.array([0, 0]), np.array([0.1, 0.2]))
    assert m["accuracy"] == 1.0 and m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0
    empty = binary_metrics(np.array([], dtype=int), np.array([]))
    assert empty["n"] == 0 and math.isnan(empty["accuracy"])
