import numpy as np
from sklearn.metrics import roc_auc_score

from neuroprobe_eval.base_runner import BaseRunner


def test_compute_metrics_binary_uses_positive_class_probability():
    classes = np.array([2, 4], dtype=np.int64)
    y_true = np.array([2, 4, 2, 4], dtype=np.int64)
    y_proba = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.7, 0.3],
            [0.1, 0.9],
        ],
        dtype=np.float64,
    )

    accuracy, roc_auc = BaseRunner._compute_metrics(y_true, y_proba, classes)

    expected_auc = roc_auc_score((y_true == 4).astype(np.int64), y_proba[:, 1])
    assert accuracy == 1.0
    assert np.isclose(roc_auc, expected_auc)


def test_compute_metrics_handles_binary_subset_of_declared_classes():
    classes = np.array([0, 1, 2], dtype=np.int64)
    y_true = np.array([0, 2, 0, 2], dtype=np.int64)
    y_proba = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.2, 0.1, 0.7],
            [0.8, 0.1, 0.1],
            [0.1, 0.2, 0.7],
        ],
        dtype=np.float64,
    )

    accuracy, roc_auc = BaseRunner._compute_metrics(y_true, y_proba, classes)

    expected_auc = roc_auc_score((y_true == 2).astype(np.int64), y_proba[:, 2])
    assert accuracy == 1.0
    assert np.isclose(roc_auc, expected_auc)


def test_compute_metrics_returns_nan_auc_for_single_class_truth():
    classes = np.array([0, 1], dtype=np.int64)
    y_true = np.array([1, 1, 1], dtype=np.int64)
    y_proba = np.array(
        [
            [0.1, 0.9],
            [0.2, 0.8],
            [0.3, 0.7],
        ],
        dtype=np.float64,
    )

    accuracy, roc_auc = BaseRunner._compute_metrics(y_true, y_proba, classes)

    assert accuracy == 1.0
    assert np.isnan(roc_auc)
