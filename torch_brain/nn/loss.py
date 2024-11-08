import torch
import torch.nn.functional as F
from torchmetrics import R2Score

from torch_brain.registry import DataType


def compute_loss_or_metric(
    loss_or_metric: str,
    output_type: DataType,
    output: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    r"""Helper function to compute various losses or metrics for a given output type.

    It supports both continuous and discrete output types, and a variety of losses
    and metrics, including mse loss, binary cross entropy loss, and R2 score.

    Args:
        loss_or_metric: The name of the metric to compute.
        output_type: The nature of the output. One of the values from DataType.
        output: The output tensor.
        target: The target tensor.
        weights: The sample-wise weights for the loss computation.
    """
    if output_type == DataType.CONTINUOUS:
        if loss_or_metric == "mse":
            # TODO mse could be used as a loss or as a metric. Currently it fails when
            # called as a metric
            # MSE loss
            loss_noreduce = F.mse_loss(output, target, reduction="none").mean(dim=1)
            return (weights * loss_noreduce).sum() / weights.sum()
        elif loss_or_metric == "r2":
            r2score = R2Score(num_outputs=target.shape[1])
            return r2score(output, target)
        elif loss_or_metric == "frame_diff_acc":
            normalized_window = 30 / 450
            differences = torch.abs(output - target)
            correct_predictions = differences <= normalized_window
            accuracy = (
                correct_predictions.float().mean()
            )  # Convert boolean tensor to float and calculate mean
            return accuracy
        else:
            raise NotImplementedError(
                f"Loss/Metric {loss_or_metric} not implemented for continuous output"
            )

    if output_type in [
        DataType.BINARY,
        DataType.MULTINOMIAL,
        DataType.MULTILABEL,
    ]:
        if loss_or_metric == "bce":
            target = target.to(torch.long).squeeze()
            # target = target.squeeze(dim=1)
            loss_noreduce = F.cross_entropy(output, target, reduction="none")
            if loss_noreduce.ndim > 1:
                loss_noreduce = loss_noreduce.mean(dim=1)
            return (weights * loss_noreduce).sum() / weights.sum()
        elif loss_or_metric == "mallows_distance":
            num_classes = output.size(-1)
            output = torch.softmax(output, dim=-1).view(-1, num_classes)
            target = target.view(-1, 1)
            weights = weights.view(-1)
            # Mallow distance
            target = torch.zeros_like(output).scatter_(1, target, 1.0)
            # we compute the mallow distance as the sum of the squared differences
            loss = torch.mean(
                torch.square(
                    torch.cumsum(target, dim=-1) - torch.cumsum(output, dim=-1)
                ),
                dim=-1,
            )
            loss = (weights * loss).sum() / weights.sum()
            return loss
        elif loss_or_metric == "accuracy":
            pred_class = torch.argmax(output, dim=1)
            return (pred_class == target.squeeze()).sum() / len(target)
        elif loss_or_metric == "frame_diff_acc":
            pred_class = torch.argmax(output, dim=1)
            difference = torch.abs(pred_class - target.squeeze())
            correct_predictions = difference <= 30
            return correct_predictions.float().mean()
        else:
            raise NotImplementedError(
                f"Loss/Metric {loss_or_metric} not implemented for binary/multilabel "
                "output"
            )

    raise NotImplementedError(
        "I don't know how to handle this task type. Implement plis"
    )
