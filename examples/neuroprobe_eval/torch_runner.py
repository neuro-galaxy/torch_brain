"""
Runner for PyTorch models.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import gc
import inspect
from torch.utils.data import DataLoader
from neuroprobe_eval.utils.logging_utils import log
from neuroprobe_eval.base_runner import BaseRunner


class TorchRunner(BaseRunner):
    """Runner for PyTorch models."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.device = self._get_device(cfg)
        self.deterministic = self._configure_determinism()
        self.wandb_run = None  # Will be set via set_wandb_run() if wandb is enabled

    def set_wandb_run(self, wandb_run):
        """Set wandb run object for logging."""
        self.wandb_run = wandb_run

    def _configure_determinism(self):
        """Configure PyTorch deterministic settings based on config."""
        model_cfg = getattr(self.cfg, "model", None)
        deterministic = (
            bool(model_cfg.get("deterministic", False)) if model_cfg else False
        )

        if deterministic:
            log("[TorchRunner] Deterministic mode enabled", priority=0)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True)
            except (AttributeError, RuntimeError) as exc:
                log(
                    f"[TorchRunner] Unable to enforce deterministic algorithms: {exc}",
                    priority=1,
                )
            if torch.cuda.is_available():
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        return deterministic

    def _get_device(self, cfg):
        """Get device from config or auto-detect."""
        device_str = cfg.model.get("device", "auto")

        if device_str == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_str == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            device = torch.device("cuda")
        elif device_str == "cpu":
            device = torch.device("cpu")
        else:
            # Allow specific device strings like 'cuda:0'
            device = torch.device(device_str)

        # Verify device is valid and log
        if device.type == "cuda":
            if device.index is not None:
                if device.index >= torch.cuda.device_count():
                    import warnings

                    warnings.warn(
                        f"GPU {device.index} not available (only {torch.cuda.device_count()} GPUs). Falling back to CPU."
                    )
                    device = torch.device("cpu")
                else:
                    # Set the current device to ensure it's accessible
                    torch.cuda.set_device(device.index)

        log(
            f"[TorchRunner] Device configured: {device_str} -> {device} (CUDA available: {torch.cuda.is_available()}, GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0})",
            priority=0,
        )
        return device

    def run_fold(
        self,
        model,
        *,
        train_loader: DataLoader | None = None,
        val_loader: DataLoader | None = None,
        test_loader: DataLoader | None = None,
        fold_idx=None,
    ):
        """
        Train and evaluate a single fold from split DataLoaders.

        Args:
            model: PyTorch model instance
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            test_loader: Test DataLoader
            fold_idx: Optional fold index for wandb logging.

        Returns:
            Dictionary with train_accuracy, train_roc_auc, val_accuracy, val_roc_auc, test_accuracy, test_roc_auc
        """
        if train_loader is None or val_loader is None or test_loader is None:
            raise ValueError(
                "run_fold requires train_loader, val_loader, and test_loader."
            )
        if not isinstance(train_loader, DataLoader):
            raise TypeError("train_loader must be a torch.utils.data.DataLoader.")
        if not isinstance(val_loader, DataLoader):
            raise TypeError("val_loader must be a torch.utils.data.DataLoader.")
        if not isinstance(test_loader, DataLoader):
            raise TypeError("test_loader must be a torch.utils.data.DataLoader.")

        return self._run_fold_with_loaders(
            model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            fold_idx=fold_idx,
        )

    def _run_fold_with_loaders(
        self,
        model,
        *,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        fold_idx=None,
    ):
        """Train/evaluate a fold from split-specific DataLoaders."""
        classes = self._infer_classes_from_loader(train_loader)
        n_classes = len(classes)
        model.classes_ = classes

        if model.model is None:
            try:
                example_raw = next(iter(train_loader))
            except StopIteration as exc:
                raise ValueError(
                    "train_loader must contain at least one batch."
                ) from exc

            example_batch = self._prepare_model_batch(model, example_raw)
            example_inputs, _, _, _, _ = self._extract_batch_tensors(example_batch)
            input_shape = tuple(example_inputs.shape[1:])

            model.build_model(input_shape, n_classes, device=self.device)
            _log_model_summary(model, input_shape, n_classes)

        training_mode = self.cfg.model.get("training_mode", "epoch_based")
        if training_mode == "steps_based":
            self._train_steps_based_loader(
                model,
                train_loader=train_loader,
                val_loader=val_loader,
                n_classes=n_classes,
                classes=classes,
                fold_idx=fold_idx,
            )
        else:
            self._train_with_early_stopping_loader(
                model,
                train_loader=train_loader,
                val_loader=val_loader,
                n_classes=n_classes,
                classes=classes,
                fold_idx=fold_idx,
            )

        train_acc, train_auc, _ = self._evaluate_loader(
            model,
            train_loader,
            n_classes=n_classes,
            classes=classes,
            criterion=None,
        )
        val_acc, val_auc, _ = self._evaluate_loader(
            model,
            val_loader,
            n_classes=n_classes,
            classes=classes,
            criterion=None,
        )
        test_acc, test_auc, _ = self._evaluate_loader(
            model,
            test_loader,
            n_classes=n_classes,
            classes=classes,
            criterion=None,
        )

        if self.wandb_run is not None and fold_idx is not None:
            self.wandb_run.log(
                {
                    f"fold_{fold_idx}/train_accuracy": float(train_acc),
                    f"fold_{fold_idx}/train_roc_auc": float(train_auc),
                    f"fold_{fold_idx}/val_accuracy": float(val_acc),
                    f"fold_{fold_idx}/val_roc_auc": float(val_auc),
                    f"fold_{fold_idx}/test_accuracy": float(test_acc),
                    f"fold_{fold_idx}/test_roc_auc": float(test_auc),
                }
            )

        torch.cuda.empty_cache()
        gc.collect()
        return {
            "train_accuracy": float(train_acc),
            "train_roc_auc": float(train_auc),
            "val_accuracy": float(val_acc),
            "val_roc_auc": float(val_auc),
            "test_accuracy": float(test_acc),
            "test_roc_auc": float(test_auc),
        }

    def _infer_classes_from_loader(self, loader: DataLoader) -> np.ndarray:
        """Infer sorted unique class ids from DataLoader batches."""
        labels: set[int] = set()
        for raw_batch in loader:
            if not isinstance(raw_batch, dict) or "y" not in raw_batch:
                raise ValueError(
                    "Unable to infer class labels from DataLoader batch. "
                    "Expected dict batches with key 'y'."
                )

            batch_y = raw_batch["y"]
            if torch.is_tensor(batch_y):
                batch_y = batch_y.detach().cpu().numpy()
            else:
                batch_y = np.asarray(batch_y)

            for value in np.asarray(batch_y).reshape(-1):
                if isinstance(value, (bool, np.bool_)):
                    raise TypeError("Class labels must be integers, got bool.")
                if not isinstance(value, (int, np.integer)):
                    raise TypeError(
                        "Class labels must be integers, got " f"{type(value).__name__}."
                    )
                labels.add(int(value))

        if not labels:
            raise ValueError("Training loader is empty; cannot infer classes.")
        return np.asarray(sorted(labels), dtype=np.int64)

    def _prepare_model_batch(
        self,
        model,
        raw_batch: dict,
    ) -> dict:
        """Apply model-specific prepare_batch hook to one collated batch dict."""
        batch = raw_batch
        prepare = getattr(model, "prepare_batch", None)
        if callable(prepare):
            batch = prepare(
                batch,
                runner_cfg=self.cfg.get("runner", {}),
                device=self.device,
            )
        return batch

    def _extract_batch_tensors(self, batch: dict):
        """Extract model tensors from one prepared batch dict."""
        if not isinstance(batch, dict):
            raise TypeError(
                f"prepare_batch must return dict, got {type(batch).__name__}."
            )
        if "x" not in batch or "y" not in batch:
            raise KeyError("prepared batch must contain keys 'x' and 'y'.")

        x = batch["x"]
        y = batch["y"]
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        else:
            x = x.float()
        if not torch.is_tensor(y):
            y = torch.as_tensor(y, dtype=torch.long)
        else:
            y = y.long()

        coords = batch.get("channel_coords_lip")
        seq_id = batch.get("seq_id")
        model_kwargs = batch.get("model_kwargs", None)
        if model_kwargs is not None and not isinstance(model_kwargs, dict):
            raise TypeError("batch['model_kwargs'] must be a dict when provided.")
        return x, y, coords, seq_id, model_kwargs

    def _evaluate_loader(
        self,
        model,
        loader: DataLoader,
        *,
        n_classes: int,
        classes: np.ndarray,
        criterion=None,
    ):
        """Evaluate a model on a split DataLoader."""
        model.model.eval()
        all_probs = []
        all_targets = []
        running_loss = 0.0
        n_samples = 0

        with torch.no_grad():
            for raw_batch in loader:
                batch = self._prepare_model_batch(model, raw_batch)
                batch_x, batch_y, batch_coords, batch_seq_id, model_kwargs = (
                    self._extract_batch_tensors(batch)
                )
                batch_x = batch_x.to(self.device)
                batch_y_device = batch_y.to(self.device)
                outputs = self._forward_model(
                    model.model,
                    batch_x,
                    batch_coords,
                    batch_seq_id,
                    accepts_coords=getattr(model, "accepts_coords", False),
                    model_kwargs=model_kwargs,
                )
                probs = torch.nn.functional.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())

                if criterion is not None:
                    running_loss += float(
                        criterion(outputs, batch_y_device).item()
                    ) * batch_y.size(0)
                    n_samples += batch_y.size(0)

        if not all_probs:
            raise ValueError("Cannot evaluate on an empty loader.")

        y_proba = np.concatenate(all_probs, axis=0)
        y_true = np.concatenate(all_targets, axis=0)
        accuracy, roc_auc = self._compute_metrics(y_true, y_proba, classes)
        avg_loss = (
            (running_loss / n_samples)
            if criterion is not None and n_samples > 0
            else None
        )
        return accuracy, roc_auc, avg_loss

    def _compute_metrics_from_train_batches(
        self,
        *,
        prob_chunks: list[np.ndarray],
        target_chunks: list[np.ndarray],
        classes: np.ndarray,
    ) -> tuple[float, float]:
        """Compute metrics from cached train-loop predictions/targets."""
        if not prob_chunks or not target_chunks:
            raise ValueError("Cannot compute train metrics from empty batch caches.")
        y_proba = np.concatenate(prob_chunks, axis=0)
        y_true = np.concatenate(target_chunks, axis=0)
        return self._compute_metrics(y_true, y_proba, classes)

    def _train_with_early_stopping_loader(
        self,
        model,
        *,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_classes: int,
        classes: np.ndarray,
        fold_idx=None,
    ):
        """Loader-based epoch training with early stopping."""
        criterion = nn.CrossEntropyLoss()
        optimizer, scheduler, _ = self._create_optimizer_and_scheduler(model)

        max_iter = self.cfg.model.get("max_iter", 100)
        patience = self.cfg.model.get("patience", 10)
        tol = self.cfg.model.get("tol", 1e-4)

        best_val_auroc = 0.0
        best_model_state = None
        patience_counter = 0
        wandb_prefix = self._get_wandb_prefix(fold_idx)

        for epoch in range(max_iter):
            model.model.train()
            train_loss = 0.0
            train_total = 0
            train_prob_chunks: list[np.ndarray] = []
            train_target_chunks: list[np.ndarray] = []
            for raw_batch in train_loader:
                batch = self._prepare_model_batch(model, raw_batch)
                batch_x, batch_y, batch_coords, batch_seq_id, model_kwargs = (
                    self._extract_batch_tensors(batch)
                )
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self._forward_model(
                    model.model,
                    batch_x,
                    batch_coords,
                    batch_seq_id,
                    accepts_coords=getattr(model, "accepts_coords", False),
                    model_kwargs=model_kwargs,
                )
                loss = criterion(outputs, batch_y)
                loss.backward()

                grad_clip = self.cfg.model.get("grad_clip", None)
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.model.parameters(), grad_clip)

                optimizer.step()
                if scheduler is not None:
                    scheduler.step(loss.item())

                train_loss += float(loss.item()) * batch_y.size(0)
                train_total += batch_y.size(0)
                # Keep online train metrics from the same batches used for updates
                # to avoid a second full train-loader pass each epoch.
                train_prob_chunks.append(
                    torch.nn.functional.softmax(outputs.detach(), dim=1).cpu().numpy()
                )
                train_target_chunks.append(batch_y.detach().cpu().numpy())

            avg_train_loss = train_loss / train_total if train_total > 0 else 0.0
            val_accuracy, val_auroc, val_loss = self._evaluate_loader(
                model,
                val_loader,
                n_classes=n_classes,
                classes=classes,
                criterion=criterion,
            )
            train_accuracy, train_auroc = self._compute_metrics_from_train_batches(
                prob_chunks=train_prob_chunks,
                target_chunks=train_target_chunks,
                classes=classes,
            )

            self._log_metrics_to_wandb(
                wandb_prefix,
                val_auroc,
                val_accuracy,
                train_loss=avg_train_loss,
                val_loss=val_loss,
                train_auroc=train_auroc,
                train_accuracy=train_accuracy,
                epoch=epoch,
            )
            val_loss_text = f"{val_loss:.4f}" if val_loss is not None else "n/a"
            fold_label = f"Fold {fold_idx}" if fold_idx is not None else "Fold"
            log(
                f"{fold_label}: train_epoch={epoch + 1}/{max_iter} "
                f"train_loss={avg_train_loss:.4f} "
                f"train_acc={train_accuracy:.3f} train_roc_auc={train_auroc:.3f} "
                f"val_loss={val_loss_text} "
                f"val_acc={val_accuracy:.3f} val_roc_auc={val_auroc:.3f}",
                priority=0,
            )

            if val_auroc > best_val_auroc + tol:
                best_val_auroc = val_auroc
                best_model_state = {
                    k: v.cpu().clone() for k, v in model.model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_model_state is not None:
            model.model.load_state_dict(best_model_state)

    def _train_steps_based_loader(
        self,
        model,
        *,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_classes: int,
        classes: np.ndarray,
        fold_idx=None,
    ):
        """Loader-based fixed-step training loop."""
        criterion = nn.CrossEntropyLoss()
        optimizer, scheduler, learning_rate = self._create_optimizer_and_scheduler(
            model
        )

        total_steps = self.cfg.model.get("total_steps", 2000)
        validation_interval = self.cfg.model.get("validation_interval", 100)

        best_val_auroc = 0.0
        best_model_state = None
        wandb_prefix = self._get_wandb_prefix(fold_idx)
        train_prob_chunks: list[np.ndarray] = []
        train_target_chunks: list[np.ndarray] = []

        step = 0
        train_iter = iter(train_loader)
        while step < total_steps:
            model.model.train()
            try:
                raw_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                raw_batch = next(train_iter)

            batch = self._prepare_model_batch(model, raw_batch)
            batch_x, batch_y, batch_coords, batch_seq_id, model_kwargs = (
                self._extract_batch_tensors(batch)
            )
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            optimizer.zero_grad()
            outputs = self._forward_model(
                model.model,
                batch_x,
                batch_coords,
                batch_seq_id,
                accepts_coords=getattr(model, "accepts_coords", False),
                model_kwargs=model_kwargs,
            )
            loss = criterion(outputs, batch_y)
            loss.backward()

            grad_clip = self.cfg.model.get("grad_clip", None)
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), grad_clip)

            optimizer.step()
            if scheduler is not None:
                scheduler.step(loss.item())

            # Cache train-window predictions/targets so validation checkpoints can
            # report train metrics without re-iterating the full train loader.
            train_prob_chunks.append(
                torch.nn.functional.softmax(outputs.detach(), dim=1).cpu().numpy()
            )
            train_target_chunks.append(batch_y.detach().cpu().numpy())

            step += 1
            if step % validation_interval != 0 and step != total_steps:
                continue

            val_accuracy, val_auroc, val_loss = self._evaluate_loader(
                model,
                val_loader,
                n_classes=n_classes,
                classes=classes,
                criterion=criterion,
            )
            train_accuracy, train_auroc = self._compute_metrics_from_train_batches(
                prob_chunks=train_prob_chunks,
                target_chunks=train_target_chunks,
                classes=classes,
            )
            train_prob_chunks = []
            train_target_chunks = []

            current_lr = scheduler.get_lr() if scheduler is not None else learning_rate
            self._log_metrics_to_wandb(
                wandb_prefix,
                val_auroc,
                val_accuracy,
                train_loss=float(loss.item()),
                val_loss=val_loss,
                train_auroc=train_auroc,
                train_accuracy=train_accuracy,
                step=step,
                learning_rate=current_lr,
            )
            val_loss_text = f"{val_loss:.4f}" if val_loss is not None else "n/a"
            fold_label = f"Fold {fold_idx}" if fold_idx is not None else "Fold"
            log(
                f"{fold_label}: train_step={step}/{total_steps} "
                f"train_loss={float(loss.item()):.4f} "
                f"train_acc={train_accuracy:.3f} train_roc_auc={train_auroc:.3f} "
                f"val_loss={val_loss_text} "
                f"val_acc={val_accuracy:.3f} val_roc_auc={val_auroc:.3f}",
                priority=0,
            )

            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_model_state = {
                    k: v.cpu().clone() for k, v in model.model.state_dict().items()
                }

        if best_model_state is not None:
            model.model.load_state_dict(best_model_state)

    def _create_optimizer_and_scheduler(self, model):
        """
        Create optimizer and scheduler from config.

        Returns:
            tuple: (optimizer, scheduler, learning_rate)
        """
        learning_rate = self.cfg.model.get("learning_rate", 0.001)
        optimizer_name = self.cfg.model.get("optimizer", "Adam")
        weight_decay = self.cfg.model.get("weight_decay", 0.0)
        optimizer_cls = getattr(torch.optim, optimizer_name, torch.optim.Adam)

        param_groups = None
        if hasattr(model, "get_parameter_groups"):
            param_groups = model.get_parameter_groups()

        if param_groups:
            optimizer = optimizer_cls(param_groups, weight_decay=weight_decay)
            log(
                f"  Using separate learning rates: {[pg.get('lr', learning_rate) for pg in param_groups]}",
                priority=0,
                indent=2,
            )
        else:
            optimizer = optimizer_cls(
                model.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )

        # Initialize scheduler if configured
        scheduler = None
        if "scheduler" in self.cfg.model:
            from neuroprobe_eval.schedulers import build_scheduler

            scheduler = build_scheduler(self.cfg.model.scheduler, optimizer)
            if scheduler is not None:
                log(
                    f"  Scheduler: {self.cfg.model.scheduler.get('name', 'unknown')}",
                    priority=0,
                    indent=2,
                )

        return optimizer, scheduler, learning_rate

    def _get_wandb_prefix(self, fold_idx):
        """
        Get wandb logging prefix for a fold.

        Returns:
            str: Prefix string (e.g., "fold_0/") or empty string
        """
        if fold_idx is not None and self.wandb_run is not None:
            return f"fold_{fold_idx}/"
        return ""

    def _log_metrics_to_wandb(
        self,
        prefix,
        val_auroc,
        val_accuracy,
        train_loss=None,
        val_loss=None,
        train_auroc=None,
        train_accuracy=None,
        epoch=None,
        step=None,
        learning_rate=None,
    ):
        """
        Log metrics to wandb.

        Args:
            prefix: Wandb prefix (from _get_wandb_prefix)
            val_auroc: Validation AUROC
            val_accuracy: Validation accuracy
            train_loss: Training loss (optional)
            val_loss: Validation loss (optional)
            train_auroc: Training ROC-AUC (optional)
            train_accuracy: Training accuracy (optional)
            epoch: Epoch number (for epoch-based training)
            step: Step number (for steps-based training)
            learning_rate: Current learning rate (optional)
        """
        if self.wandb_run is None:
            return

        log_dict = {
            f"{prefix}val_roc_auc": val_auroc,
            f"{prefix}val_accuracy": val_accuracy,
        }

        if train_loss is not None:
            log_dict[f"{prefix}train_loss"] = train_loss

        if val_loss is not None:
            log_dict[f"{prefix}val_loss"] = val_loss

        if train_auroc is not None:
            log_dict[f"{prefix}train_roc_auc"] = train_auroc

        if train_accuracy is not None:
            log_dict[f"{prefix}train_accuracy"] = train_accuracy

        if epoch is not None:
            log_dict[f"{prefix}epoch"] = epoch

        if step is not None:
            log_dict[f"{prefix}step"] = step

        if learning_rate is not None:
            log_dict[f"{prefix}learning_rate"] = learning_rate

        self.wandb_run.log(log_dict)

    def _forward_model(
        self,
        torch_model,
        inputs,
        coords=None,
        seq_id=None,
        accepts_coords=False,
        batch_start_idx=0,
        batch_size=None,
        model_kwargs=None,
    ):
        """
        Forward helper that optionally supplies coordinates and seq_id to the model.

        Args:
            torch_model: The PyTorch model
            inputs: Input tensor (batch_size, n_electrodes+1, hidden_dim)
            coords: Optional coordinate tensor - can be:
                - (n_electrodes, 3) shared across all samples (broadcasted)
                - (n_samples, n_electrodes, 3) per-sample coordinates
            seq_id: Optional seq_id tensor - can be:
                - (n_electrodes,) shared across all samples (broadcasted)
                - (n_samples, n_electrodes) per-sample seq_id
            batch_start_idx: Starting index of current batch (for per-sample coords/seq_id)
            batch_size: Size of current batch (for per-sample coords/seq_id)
            model_kwargs: Optional kwargs to forward to torch_model (e.g. pad_mask)
        """
        model_kwargs = dict(model_kwargs or {})

        # Check if model accepts positions parameter
        sig = inspect.signature(torch_model.forward)
        accepts_positions = "positions" in sig.parameters
        accepts_pad_mask = "pad_mask" in sig.parameters
        if "pad_mask" in model_kwargs and not accepts_pad_mask:
            model_kwargs.pop("pad_mask")

        if accepts_positions and coords is not None and seq_id is not None:
            batch_size_actual = inputs.shape[0]
            batch_start = 0 if batch_start_idx is None else int(batch_start_idx)

            coords_t = torch.as_tensor(coords)
            if coords_t.ndim == 2:
                # Shared coords: (n_electrodes, 3) -> broadcast to (batch_size, n_electrodes, 3)
                batch_coords = coords_t.unsqueeze(0).expand(batch_size_actual, -1, -1)
            elif coords_t.ndim == 3:
                # Per-sample coords: slice for this batch
                batch_end = min(batch_start + batch_size_actual, coords_t.shape[0])
                batch_coords = coords_t[batch_start:batch_end]
            else:
                raise ValueError(f"Unexpected coords shape: {tuple(coords_t.shape)}")

            seq_id_t = torch.as_tensor(seq_id)
            if seq_id_t.ndim == 1:
                # Shared seq_id: (n_electrodes,) -> broadcast to (batch_size, n_electrodes)
                batch_seq_id = seq_id_t.unsqueeze(0).expand(batch_size_actual, -1)
            elif seq_id_t.ndim == 2:
                # Per-sample seq_id: slice for this batch
                batch_end = min(batch_start + batch_size_actual, seq_id_t.shape[0])
                batch_seq_id = seq_id_t[batch_start:batch_end]
            else:
                raise ValueError(f"Unexpected seq_id shape: {tuple(seq_id_t.shape)}")

            # Convert to tensors on device
            # Coords are used as indices in MultiSubjBrainPositionalEncoding, so use int64
            # Note: LPI coordinates are cast to integers (matching PopT behavior)
            batch_coords = batch_coords.to(device=self.device, dtype=torch.int64)
            batch_seq_id = batch_seq_id.to(device=self.device, dtype=torch.int64)

            positions = (batch_coords, batch_seq_id)
            return torch_model(inputs, positions=positions, **model_kwargs)
        elif coords is not None and accepts_coords:
            # Explicitly-declared coords-only fallback for legacy models.
            return torch_model(inputs, coords, **model_kwargs)
        else:
            return torch_model(inputs, **model_kwargs)


def _log_model_summary(model, input_shape, n_classes):
    """
    Log model architecture summary including parameter count.
    Uses PyTorch's model representation and parameter counting.

    Args:
        model: Model instance with a .model attribute (the PyTorch module)
        input_shape: Input shape tuple
        n_classes: Number of output classes
    """
    if model.model is None:
        return

    log("=" * 80, priority=0, indent=1)
    log("Model Architecture Summary:", priority=0, indent=1)
    log("=" * 80, priority=0, indent=1)

    # Input/output info
    log(f"Input shape: {input_shape}", priority=0, indent=2)
    log(f"Output classes: {n_classes}", priority=0, indent=2)

    # Count parameters
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(
        p.numel() for p in model.model.parameters() if p.requires_grad
    )
    non_trainable_params = total_params - trainable_params

    log(f"Total parameters: {total_params:,}", priority=0, indent=2)
    log(f"Trainable parameters: {trainable_params:,}", priority=0, indent=2)
    if non_trainable_params > 0:
        log(f"Non-trainable parameters: {non_trainable_params:,}", priority=0, indent=2)

    # Model-specific details
    if hasattr(model, "hidden_dims"):
        hidden_dims = model.hidden_dims
        if isinstance(hidden_dims, (list, tuple)) and len(hidden_dims) > 0:
            log(f"Hidden layers: {hidden_dims}", priority=0, indent=2)
        else:
            log(f"Architecture: Linear (no hidden layers)", priority=0, indent=2)

    # Basic model structure (safe, no hooks that can interfere with training)
    log("Model structure:", priority=0, indent=2)
    model_str = str(model.model)
    for line in model_str.split("\n")[:50]:
        if line.strip():
            log(line, priority=0, indent=3)
    if len(model_str.split("\n")) > 50:
        log("... (output truncated)", priority=0, indent=3)

    log("=" * 80, priority=0, indent=1)
