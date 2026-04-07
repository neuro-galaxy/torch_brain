"""
Base runner utilities shared across sklearn and PyTorch runners.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


class BaseRunner:
    """Shared functionality for model runners."""

    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    def _compute_metrics(y_true, y_proba, classes):
        """Compute accuracy and ROC-AUC for a set of predictions."""
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)
        classes = np.asarray(classes)

        if y_proba.ndim != 2:
            raise ValueError(
                f"y_proba must be 2D [n_samples, n_classes], got shape {y_proba.shape}."
            )
        if y_proba.shape[1] != len(classes):
            raise ValueError(
                "y_proba column count must match classes length, got "
                f"{y_proba.shape[1]} and {len(classes)}."
            )

        predictions = classes[np.argmax(y_proba, axis=1)]
        accuracy = np.mean(predictions == y_true)

        valid_class_mask = np.isin(y_true, classes)
        y_filtered = y_true[valid_class_mask]
        y_proba_filtered = y_proba[valid_class_mask]
        if y_filtered.size == 0:
            return accuracy, float("nan")

        class_to_idx = {int(label): idx for idx, label in enumerate(classes)}
        y_indices = np.asarray([class_to_idx[int(label)] for label in y_filtered])
        present_indices = np.unique(y_indices)

        # roc_auc_score requires at least two classes in y_true.
        if present_indices.size < 2:
            return accuracy, float("nan")

        if present_indices.size == 2:
            negative_idx, positive_idx = int(present_indices[0]), int(present_indices[1])
            y_binary = (y_indices == positive_idx).astype(np.int64)
            y_score = y_proba_filtered[:, positive_idx]
            roc_auc = roc_auc_score(y_binary, y_score)
            return accuracy, roc_auc

        index_remap = {int(idx): remapped for remapped, idx in enumerate(present_indices)}
        y_multiclass = np.asarray([index_remap[int(idx)] for idx in y_indices])
        y_score = y_proba_filtered[:, present_indices]
        row_sums = y_score.sum(axis=1, keepdims=True)
        safe_row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
        y_score = y_score / safe_row_sums
        roc_auc = roc_auc_score(y_multiclass, y_score, multi_class="ovr", average="macro")

        return accuracy, roc_auc
