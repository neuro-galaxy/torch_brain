"""
Runner for sklearn models (like PopulationTransformer's Runner pattern).
"""

import gc
from neuroprobe_eval.base_runner import BaseRunner


class SKLearnRunner(BaseRunner):
    """Runner for sklearn models."""

    def __init__(self, cfg):
        super().__init__(cfg)

    def run_fold(self, model, X_train, y_train, X_test, y_test):
        """
        Train and evaluate a single fold.

        Args:
            model: sklearn model instance
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with train_accuracy, train_roc_auc, test_accuracy, test_roc_auc
        """
        # Data is already standardized in preprocessing pipeline
        gc.collect()

        # Train
        model.fit(X_train, y_train)

        # Evaluate
        train_acc, train_auc = self._evaluate(model, X_train, y_train)
        test_acc, test_auc = self._evaluate(model, X_test, y_test)

        # Clean up
        gc.collect()

        return {
            "train_accuracy": float(train_acc),
            "train_roc_auc": float(train_auc),
            "test_accuracy": float(test_acc),
            "test_roc_auc": float(test_auc),
        }

    def _evaluate(self, model, X, y):
        """Evaluate model and return (accuracy, roc_auc)."""
        # Get predictions
        y_proba = model.predict_proba(X)
        return self._compute_metrics(y, y_proba, model.classes_)
