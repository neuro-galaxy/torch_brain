"""
Metrics computation utilities.
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def compute_roc_auc(y_true, y_proba, classes):
    """
    Compute ROC AUC score, handling both binary and multiclass cases.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities (n_samples, n_classes)
        classes: Array of unique class labels
    
    Returns:
        ROC AUC score
    """
    # Filter to only include classes that were in training
    valid_class_mask = np.isin(y_true, classes)
    y_filtered = y_true[valid_class_mask]
    y_proba_filtered = y_proba[valid_class_mask]
    
    # Convert to one-hot encoding
    n_classes = len(classes)
    y_onehot = np.zeros((len(y_filtered), n_classes))
    for i, label in enumerate(y_filtered):
        class_idx = np.where(classes == label)[0][0]
        y_onehot[i, class_idx] = 1
    
    # Calculate ROC AUC
    if n_classes > 2:
        return roc_auc_score(y_onehot, y_proba_filtered, multi_class='ovr', average='macro')
    else:
        return roc_auc_score(y_onehot, y_proba_filtered)

