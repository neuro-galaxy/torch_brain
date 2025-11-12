import torch
import torch.nn.functional as F


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


def training_step(batch, model, optimizer):
    optimizer.zero_grad()  # Step 0. Clear old gradients

    inputs = batch["model_inputs"]
    target = batch["target_values"]

    pred = model(**inputs)  # Step 1. Do forward pass

    # shapes: [B, T, 1] -> [B, T]
    if pred.dim() == 3 and pred.size(-1) == 1:
        pred = pred.squeeze(-1)

    loss = F.mse_loss(pred, target)  # Step 2. Compute loss
    loss.backward()  # Step 3. Backward pass
    optimizer.step()  # Step 4. Update model params
    return loss
