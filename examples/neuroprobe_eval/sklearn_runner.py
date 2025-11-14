"""
Runner for sklearn models (like PopulationTransformer's Runner pattern).
"""
import numpy as np
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


class SKLearnRunner:
    """Runner for sklearn models."""
    
    def __init__(self, cfg):
        self.cfg = cfg
    
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
        # Standardize
        scaler = StandardScaler(copy=False)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        gc.collect()
        
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_acc, train_auc = self._evaluate(model, X_train_scaled, y_train)
        test_acc, test_auc = self._evaluate(model, X_test_scaled, y_test)
        
        # Clean up
        del scaler
        gc.collect()
        
        return {
            "train_accuracy": float(train_acc),
            "train_roc_auc": float(train_auc),
            "test_accuracy": float(test_acc),
            "test_roc_auc": float(test_auc)
        }
    
    def _evaluate(self, model, X, y):
        """Evaluate model and return (accuracy, roc_auc)."""
        accuracy = model.score(X, y)
        
        # Get predictions
        y_proba = model.predict_proba(X)
        
        # Filter to only include classes that were in training
        valid_class_mask = np.isin(y, model.classes_)
        y_filtered = y[valid_class_mask]
        y_proba_filtered = y_proba[valid_class_mask]
        
        # Convert to one-hot encoding
        n_classes = len(model.classes_)
        y_onehot = np.zeros((len(y_filtered), n_classes))
        for i, label in enumerate(y_filtered):
            class_idx = np.where(model.classes_ == label)[0][0]
            y_onehot[i, class_idx] = 1
        
        # Calculate ROC AUC
        if n_classes > 2:
            roc_auc = roc_auc_score(y_onehot, y_proba_filtered, multi_class='ovr', average='macro')
        else:
            roc_auc = roc_auc_score(y_onehot, y_proba_filtered)
        
        return accuracy, roc_auc

