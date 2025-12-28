import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torchmetrics import MeanSquaredError
from torchmetrics.classification import MulticlassAccuracy, BinaryAccuracy
from torch_brain.metrics import MetricWrapper  # adjust if needed


def _dist_test_fn(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Setup MetricWrapper with multiple syncable metrics
    metric = MetricWrapper(
        {
            "recording_001": MeanSquaredError(),
            "recording_002": MulticlassAccuracy(num_classes=3),
            "recording_003": BinaryAccuracy(),
        }
    )

    # Only rank 0 updates
    if rank == 0:
        # MSE data
        mse_preds = torch.tensor([1.0, 2.0, 3.0])
        mse_targets = torch.tensor([1.1, 2.1, 3.1])
        mse_timestamps = torch.tensor([10.0, 11.0, 12.0])
        metric.update(
            mse_preds, mse_targets, mse_timestamps, recording_id="recording_001"
        )

        # Multiclass accuracy data
        mc_preds = torch.tensor([[0.1, 0.8, 0.1], [0.2, 0.1, 0.7], [0.9, 0.05, 0.05]])
        mc_targets = torch.tensor([1, 2, 0])
        mc_timestamps = torch.tensor([10.0, 11.0, 12.0])
        metric.update(mc_preds, mc_targets, mc_timestamps, recording_id="recording_002")

        # Binary accuracy data
        bin_preds = torch.tensor([0.1, 0.8, 0.9, 0.2])
        bin_targets = torch.tensor([0, 1, 1, 0])
        bin_timestamps = torch.tensor([10.0, 11.0, 12.0, 13.0])
        metric.update(
            bin_preds, bin_targets, bin_timestamps, recording_id="recording_003"
        )

    # All ranks call compute
    result = metric.compute()

    # Both ranks should receive the same result due to distributed synchronization
    expected_mse = torch.tensor(0.01)
    expected_mc_acc = torch.tensor(1.0)  # All predictions are correct
    expected_bin_acc = torch.tensor(1.0)  # All binary predictions are correct

    assert torch.allclose(
        result["recording_001"], expected_mse, atol=1e-6
    ), f"Rank {rank}: Expected {expected_mse}, got {result['recording_001']}"
    assert torch.allclose(
        result["recording_002"], expected_mc_acc, atol=1e-6
    ), f"Rank {rank}: Expected {expected_mc_acc}, got {result['recording_002']}"
    assert torch.allclose(
        result["recording_003"], expected_bin_acc, atol=1e-6
    ), f"Rank {rank}: Expected {expected_bin_acc}, got {result['recording_003']}"

    dist.destroy_process_group()


def test_metric_wrapper_no_update_on_some_ranks():
    world_size = 4
    mp.spawn(_dist_test_fn, args=(world_size,), nprocs=world_size, join=True)
