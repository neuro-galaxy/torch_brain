import logging
from tqdm import tqdm
import torch
import torch.nn.functional as F

from torch_brain.models.base_class import TorchBrainModel


def move_to_device(data, device):
    """
    Recursively move data to the specified device.

    Parameters
    ----------
    data : torch.Tensor, dict, list, or other
        The data to move. Can be a tensor, a dictionary of tensors, a list of tensors, or nested structures thereof.
    device : torch.device or str
        The device to move the data to (e.g., 'cpu', 'cuda', or a torch.device object).

    Returns
    -------
    Moved data of the same structure as input, with all tensors moved to the specified device.
    Non-tensor objects are returned unchanged.
    """
    if isinstance(data, torch.Tensor):
        # Specify dtype on the move for float tensors.
        if data.is_floating_point():
            return data.to(device=device, dtype=torch.float32)
        else:
            return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    else:
        return data


def r2_score(y_pred, y_true):
    """
    Computes the coefficient of determination (R² score) between predictions and true values.

    R² is calculated as: R² = 1 - (SS_res / SS_tot)
    where SS_res is the sum of squared residuals and SS_tot is the total sum of squares.

    Args:
        y_pred (torch.Tensor): Predicted values. Shape should match y_true.
        y_true (torch.Tensor): Ground truth (target) values.

    Returns:
        torch.Tensor: The R² score as a scalar tensor.
    """
    # Compute total sum of squares (variance of the true values)
    y_true_mean = torch.mean(y_true, dim=0, keepdim=True)
    ss_total = torch.sum((y_true - y_true_mean) ** 2)

    # Compute residual sum of squares
    ss_res = torch.sum((y_true - y_pred) ** 2)

    # Compute R^2
    r2 = 1 - ss_res / ss_total

    return r2


def compute_r2(dataloader, model, device):
    """
    Compute the R^2 score for a model over a given dataloader.

    Parameters:
        dataloader: DataLoader
            An iterable over batches of data, each containing model inputs and target values.
        model: torch.nn.Module
            The model to evaluate.
        device: torch.device or str
            The device on which computation is performed.

    Returns:
        tuple:
            r2 (float): The R^2 score computed over all batches.
            total_target (torch.Tensor): Concatenated ground truth target values.
            total_pred (torch.Tensor): Concatenated model predictions.
    """
    model.eval()  # turn off dropout, etc.
    total_target = []
    total_pred = []
    with torch.no_grad():  # <-- crucial: no graph, no huge memory
        for batch in dataloader:
            batch = move_to_device(batch, device)
            pred = model(**batch["model_inputs"])
            target = batch["target_values"]

            # If your model returns [B, T, 1], squeeze to [B, T]
            if pred.dim() == 3 and pred.size(-1) == 1:
                pred = pred.squeeze(-1)

            mask = torch.ones_like(target, dtype=torch.bool)
            if "output_mask" in batch["model_inputs"]:
                mask = batch["model_inputs"]["output_mask"]
                if mask.dim() == 3 and mask.size(-1) == 1:
                    mask = mask.squeeze(-1)

            total_target.append(target[mask])
            total_pred.append(pred[mask])

    total_target = torch.cat(total_target)
    total_pred = torch.cat(total_pred)

    r2 = r2_score(total_pred.flatten(), total_target.flatten())
    return r2.item(), total_target, total_pred


def regression_training_step(
    batch: dict,
    model: TorchBrainModel,
    optimizer: torch.optim.Optimizer,
):
    """
    Performs a single training step: forward pass, loss computation, backward pass, and optimizer step.

    Args:
        batch (dict): A batch of data containing 'model_inputs' and 'target_values'.
        model (TorchBrainModel): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used to update model parameters.

    Returns:
        torch.Tensor: The computed loss for the batch.
    """
    # Clear old gradients
    optimizer.zero_grad()

    # Get input and target values from the batch
    if "model_inputs" not in batch:
        raise ValueError("Batch must contain 'model_inputs' key.")
    if "target_values" not in batch:
        raise ValueError("Batch must contain 'target_values' key.")
    inputs = batch["model_inputs"]
    target = batch["target_values"]

    # Do forward pass
    pred = model(**inputs)

    # Squeeze if singleton last dimension
    # shapes: [B, T, 1] -> [B, T]
    if pred.dim() == 3 and pred.size(-1) == 1:
        pred = pred.squeeze(-1)

    # Compute loss
    loss = F.mse_loss(pred, target)

    # Backward pass
    loss.backward()

    # Update model params
    optimizer.step()
    return loss


def train_model(
    device: torch.device,
    model: TorchBrainModel,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs=50,
    store_params: list[str] | None = None,
):
    """Train the model using the provided training and validation data loaders."""
    # Store intermediate outputs for visualization
    train_outputs = {
        "n_epochs": num_epochs,
        "output_pred": [],
        "output_gt": [],
    }
    if store_params is not None and len(store_params) > 0:
        for param in store_params:
            train_outputs[param] = []

    # Training loop
    r2_log = []
    loss_log = []
    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress", leave=True)
    for epoch in epoch_pbar:
        # Validation before training step
        with torch.no_grad():
            model.eval()  # make sure we're in eval mode during validation
            r2, target, pred = compute_r2(val_loader, model, device)
            r2_log.append(r2)

        # Switch back to training mode
        model.train()

        # Training steps
        # Inner progress bar for training batches
        running_loss = 0.0
        batch_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            leave=False,
        )
        for batch in batch_pbar:
            batch = move_to_device(data=batch, device=device)
            loss = regression_training_step(
                batch=batch,
                model=model,
                optimizer=optimizer,
            )
            loss_log.append(loss.item())
            running_loss += loss.item()

            # Update inner bar postfix
            batch_pbar.set_postfix(
                {"Loss": f"{loss.item():.4f}", "Val R2": f"{r2:.3f}"}
            )

        avg_loss = running_loss / len(train_loader)
        epoch_pbar.set_postfix({"Avg Loss": f"{avg_loss:.4f}", "Val R2": f"{r2:.3f}"})

        # Store intermediate outputs
        if store_params is not None and len(store_params) > 0:
            for param in store_params:
                if not hasattr(model, param):
                    raise ValueError(f"Model has no parameter named '{param}'")
                train_outputs[param].append(
                    getattr(model, param).weight[1:].detach().cpu().numpy()
                )

        train_outputs["output_gt"].append(target.detach().cpu().numpy())
        train_outputs["output_pred"].append(pred.detach().cpu().numpy())

    # Compute final R² score
    r2, _, _ = compute_r2(val_loader, model, device)
    r2_log.append(r2)
    print(f"\nDone! Final validation R2 = {r2:.3f}")

    return r2_log, loss_log, train_outputs
