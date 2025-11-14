"""
Runner for PyTorch models.
"""
import torch
import torch.nn as nn
import numpy as np
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


class TorchRunner:
    """Runner for PyTorch models."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = self._get_device(cfg)
    
    def _get_device(self, cfg):
        """Get device from config or auto-detect."""
        device_str = cfg.model.get("device", "auto")
        
        if device_str == "auto":
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device_str == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            return torch.device('cuda')
        elif device_str == "cpu":
            return torch.device('cpu')
        else:
            # Allow specific device strings like 'cuda:0'
            return torch.device(device_str)
    
    def run_fold(self, model, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        Train and evaluate a single fold with early stopping.
        
        Args:
            model: PyTorch model instance
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary with train_accuracy, train_roc_auc, test_accuracy, test_roc_auc
        """
        # Standardize
        scaler = StandardScaler(copy=False)
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        gc.collect()
        
        # Convert to torch tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        y_val_tensor = torch.LongTensor(y_val)
        
        # Get unique classes
        classes = np.unique(y_train)
        n_classes = len(classes)
        model.classes_ = classes
        
        # Build model if not already built
        if model.model is None:
            input_shape = X_train_scaled.shape[1:]
            model.build_model(input_shape, n_classes)
        
        # Train with early stopping using provided validation set
        self._train_with_early_stopping(
            model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, n_classes, classes
        )
        
        # Evaluate
        train_acc, train_auc = self._evaluate(model, X_train_scaled, y_train, n_classes, classes)
        test_acc, test_auc = self._evaluate(model, X_test_scaled, y_test, n_classes, classes)
        
        # Clean up
        del scaler
        torch.cuda.empty_cache()
        gc.collect()
        
        return {
            "train_accuracy": float(train_acc),
            "train_roc_auc": float(train_auc),
            "test_accuracy": float(test_acc),
            "test_roc_auc": float(test_auc)
        }
    
    def _train_with_early_stopping(self, model, X_train, y_train, X_val, y_val, n_classes, classes):
        """Train model with early stopping."""
        criterion = nn.CrossEntropyLoss()
        learning_rate = self.cfg.model.get("learning_rate", 0.001)
        optimizer = torch.optim.Adam(model.model.parameters(), lr=learning_rate)
        
        max_iter = self.cfg.model.get("max_iter", 100)
        batch_size = self.cfg.model.get("batch_size", 64)
        patience = self.cfg.model.get("patience", 10)
        tol = self.cfg.model.get("tol", 1e-4)
        
        best_val_auroc = 0.0
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(max_iter):
            model.model.train()
            train_loss = 0.0
            train_total = 0
            
            # Training batches
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size].to(self.device)
                batch_y = y_train[i:i+batch_size].to(self.device)
                
                optimizer.zero_grad()
                outputs = model.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
                train_total += batch_y.size(0)
            
            # Validation
            model.model.eval()
            with torch.no_grad():
                val_outputs = model.model(X_val.to(self.device))
                val_probs = torch.nn.functional.softmax(val_outputs, dim=1).cpu().numpy()
                y_val_np = y_val.numpy()
                
                # Convert to one-hot encoding
                y_val_onehot = np.zeros((len(y_val_np), n_classes))
                for i, label in enumerate(y_val_np):
                    class_idx = np.where(classes == label)[0][0]
                    y_val_onehot[i, class_idx] = 1
                
                if n_classes > 2:
                    val_auroc = roc_auc_score(y_val_onehot, val_probs, multi_class='ovr', average='macro')
                else:
                    val_auroc = roc_auc_score(y_val_onehot, val_probs)
                
                # Check if validation AUROC improved
                if val_auroc > best_val_auroc + tol:
                    best_val_auroc = val_auroc
                    best_model_state = {k: v.cpu().clone() for k, v in model.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
        
        # Load best model
        if best_model_state is not None:
            model.model.load_state_dict(best_model_state)
    
    def _evaluate(self, model, X, y, n_classes, classes):
        """Evaluate model and return (accuracy, roc_auc)."""
        model.model.eval()
        all_probs = []
        batch_size = self.cfg.model.get("batch_size", 64)
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size].to(self.device)
                outputs = model.model(batch_X)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
        
        y_proba = np.concatenate(all_probs, axis=0)
        
        # Calculate accuracy
        predictions = classes[np.argmax(y_proba, axis=1)]
        accuracy = np.mean(predictions == y)
        
        # Filter to only include classes that were in training
        valid_class_mask = np.isin(y, classes)
        y_filtered = y[valid_class_mask]
        y_proba_filtered = y_proba[valid_class_mask]
        
        # Convert to one-hot encoding
        y_onehot = np.zeros((len(y_filtered), n_classes))
        for i, label in enumerate(y_filtered):
            class_idx = np.where(classes == label)[0][0]
            y_onehot[i, class_idx] = 1
        
        # Calculate ROC AUC
        if n_classes > 2:
            roc_auc = roc_auc_score(y_onehot, y_proba_filtered, multi_class='ovr', average='macro')
        else:
            roc_auc = roc_auc_score(y_onehot, y_proba_filtered)
        
        return accuracy, roc_auc

