"""
Logistic Regression model wrapper for sklearn.
"""
from sklearn.linear_model import LogisticRegression
from omegaconf import DictConfig
from .base_model import BaseModel
from . import register_model


@register_model("logistic")
class LogisticModel(BaseModel):
    """Wrapper for sklearn LogisticRegression."""
    
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = LogisticRegression(
            max_iter=cfg.get("max_iter", 10000),
            tol=cfg.get("tol", 1e-3),
            random_state=cfg.get("random_state", 42)
        )
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the logistic regression model."""
        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.model.predict_proba(X)
    
    def predict(self, X):
        """Predict class labels."""
        return self.model.predict(X)
    
    def score(self, X, y):
        """Compute accuracy score."""
        return self.model.score(X, y)

