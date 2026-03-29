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
        predictions = classes[np.argmax(y_proba, axis=1)]
        accuracy = np.mean(predictions == y_true)

        valid_class_mask = np.isin(y_true, classes)
        y_filtered = y_true[valid_class_mask]
        y_proba_filtered = y_proba[valid_class_mask]

        n_classes = len(classes)
        y_onehot = np.zeros((len(y_filtered), n_classes))
        for i, label in enumerate(y_filtered):
            class_idx = np.where(classes == label)[0][0]
            y_onehot[i, class_idx] = 1

        if n_classes > 2:
            roc_auc = roc_auc_score(
                y_onehot, y_proba_filtered, multi_class="ovr", average="macro"
            )
        else:
            roc_auc = roc_auc_score(y_onehot, y_proba_filtered)

        return accuracy, roc_auc
