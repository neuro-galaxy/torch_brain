"""
Base model interface that all models must implement.
This provides a unified interface for both sklearn and PyTorch models.
"""
import numpy as np
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Base class for all models (sklearn and PyTorch)."""
    
    def __init__(self):
        self.classes_ = None
    
    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features
            y_val: Optional validation labels
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Array of shape (n_samples, n_classes) with class probabilities
        """
        pass
    
    def predict(self, X):
        """
        Predict class labels.
        
        Args:
            X: Input features
            
        Returns:
            Array of shape (n_samples,) with predicted class labels
        """
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
    
    def score(self, X, y):
        """
        Compute accuracy score.
        
        Args:
            X: Input features
            y: True labels
            
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def get_classes(self):
        """Return unique classes."""
        return self.classes_

