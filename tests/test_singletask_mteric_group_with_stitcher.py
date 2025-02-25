import pytest
import torch
import torchmetrics

from torch_brain.metrics import MetricGroupWithStitcher


@pytest.fixture
def mock_session_ids():
    return [
        "session1",
        "session2",
        "session3",
    ]


def test_initialization(mock_session_ids):
    # Test for R2Score
    metrics = {session_id: torchmetrics.R2Score() for session_id in mock_session_ids}
    evaluator = MetricGroupWithStitcher(metrics=metrics)
    assert list(evaluator.metrics.keys()) == mock_session_ids
    for metric in evaluator.metrics.values():
        assert isinstance(metric, torchmetrics.R2Score)

    # Test for Accuracy
    metrics = {
        session_id: torchmetrics.classification.MulticlassAccuracy(num_classes=8)
        for session_id in mock_session_ids
    }
    evaluator = MetricGroupWithStitcher(metrics=metrics)
    assert list(evaluator.metrics.keys()) == mock_session_ids
    for metric in evaluator.metrics.values():
        assert isinstance(metric, torchmetrics.classification.MulticlassAccuracy)

    # Test custom metric factory
    metric_cls = torchmetrics.classification.BinaryAccuracy
    metrics = {session_id: metric_cls() for session_id in mock_session_ids}
    evaluator = MetricGroupWithStitcher(metrics=metrics)
    assert list(evaluator.metrics.keys()) == mock_session_ids
    for metric in evaluator.metrics.values():
        assert isinstance(metric, metric_cls)


def test_update(mock_session_ids):
    metrics = {session_id: torchmetrics.R2Score() for session_id in mock_session_ids}
    evaluator = MetricGroupWithStitcher(metrics=metrics)

    B = 3  # batch size
    N = 17  # tokens per sample
    D = 2  # output dim (for cursor velocity 2d)

    def step_with_one_session(session_id):
        timestamps = torch.rand(B, N)
        absolute_starts = torch.rand(B)
        timestamps = timestamps + absolute_starts[:, None]
        preds = torch.rand(B, N, D)
        targets = torch.rand(B, N, D)

        for i in range(B):
            evaluator.update(
                timestamps=timestamps[i],
                preds=preds[i],
                targets=targets[i],
                recording_id=session_id,
            )

        return timestamps, preds, targets

    # Test with first session
    sess_id1 = mock_session_ids[0]
    exp_times1, exp_preds1, exp_targets1 = step_with_one_session(sess_id1)

    assert len(evaluator._cache) == len(mock_session_ids)
    assert len(evaluator._cache[sess_id1]["timestamps"]) == B
    assert torch.allclose(
        torch.cat(evaluator._cache[sess_id1]["timestamps"]), exp_times1.flatten(0, 1)
    )
    assert torch.allclose(
        torch.cat(evaluator._cache[sess_id1]["pred"]), exp_preds1.flatten(0, 1)
    )
    assert torch.allclose(
        torch.cat(evaluator._cache[sess_id1]["target"]), exp_targets1.flatten(0, 1)
    )

    # Step again with same session
    exp_times2, exp_preds2, exp_targets2 = step_with_one_session(sess_id1)
    exp_times2 = torch.cat((exp_times1, exp_times2))
    exp_preds2 = torch.cat((exp_preds1, exp_preds2))
    exp_targets2 = torch.cat((exp_targets1, exp_targets2))

    assert len(evaluator._cache[sess_id1]["timestamps"]) == 2 * B
    assert torch.allclose(
        torch.cat(evaluator._cache[sess_id1]["timestamps"]), exp_times2.flatten(0, 1)
    )
    assert torch.allclose(
        torch.cat(evaluator._cache[sess_id1]["pred"]), exp_preds2.flatten(0, 1)
    )
    assert torch.allclose(
        torch.cat(evaluator._cache[sess_id1]["target"]), exp_targets2.flatten(0, 1)
    )

    # Step with 2nd session
    sess_id2 = mock_session_ids[1]
    exp_times3, exp_preds3, exp_targets3 = step_with_one_session(sess_id2)

    assert len(evaluator._cache) == len(mock_session_ids)
    assert len(evaluator._cache[sess_id1]["timestamps"]) == 2 * B
    assert len(evaluator._cache[sess_id2]["timestamps"]) == B
    # First session cache should be unchanged
    assert torch.allclose(
        torch.cat(evaluator._cache[sess_id1]["timestamps"]), exp_times2.flatten(0, 1)
    )
    assert torch.allclose(
        torch.cat(evaluator._cache[sess_id1]["pred"]), exp_preds2.flatten(0, 1)
    )
    assert torch.allclose(
        torch.cat(evaluator._cache[sess_id1]["target"]), exp_targets2.flatten(0, 1)
    )
    # Second session cache should match new data
    assert torch.allclose(
        torch.cat(evaluator._cache[sess_id2]["timestamps"]), exp_times3.flatten(0, 1)
    )
    assert torch.allclose(
        torch.cat(evaluator._cache[sess_id2]["pred"]), exp_preds3.flatten(0, 1)
    )
    assert torch.allclose(
        torch.cat(evaluator._cache[sess_id2]["target"]), exp_targets3.flatten(0, 1)
    )


def test_end_to_end_r2(mock_session_ids):
    B = 16  # batch size
    N = 32  # tokens per sample
    D = 2  # prediction dimension (for cursor velocity 2d)
    num_sessions = len(mock_session_ids)

    metrics = {session_id: torchmetrics.R2Score() for session_id in mock_session_ids}
    evaluator = MetricGroupWithStitcher(metrics=metrics)

    for epoch in range(3):
        assert len(evaluator._cache) == len(mock_session_ids)

        for batch_step in range(10):
            batch_session_ids = [
                mock_session_ids[idx] for idx in torch.arange(B) % num_sessions
            ]
            timestamps = torch.linspace(0, 1, N).repeat(B, 1)
            preds = torch.rand(B, N, D)
            targets = torch.rand(B, N, D)
            masks = torch.rand(B, N) > 0.5
            absolute_starts = torch.rand(B)

            for i in range(B):
                evaluator.update(
                    timestamps=timestamps[i][masks[i]] + absolute_starts[i],
                    preds=preds[i][masks[i]],
                    targets=targets[i][masks[i]],
                    recording_id=batch_session_ids[i],
                )

        metric_dict = evaluator.compute()
        assert len(metric_dict) == num_sessions

        evaluator.reset()


def test_end_to_end_accuracy(mock_session_ids):
    B = 9  # batch size
    N = 50  # tokens per sample
    D = 8  # prediction dimension (for drifting gratings orientation)
    num_sessions = len(mock_session_ids)

    metrics = {
        session_id: torchmetrics.classification.MulticlassAccuracy(num_classes=D)
        for session_id in mock_session_ids
    }
    evaluator = MetricGroupWithStitcher(metrics=metrics)

    for epoch in range(3):
        assert len(evaluator._cache) == len(mock_session_ids)

        for batch_step in range(10):
            batch_session_ids = [
                mock_session_ids[idx] for idx in torch.arange(B) % num_sessions
            ]
            timestamps = torch.linspace(0, 1, N).repeat(B, 1)
            preds = torch.rand(B, N, D)
            targets = torch.randint(D, (B, N))
            masks = torch.rand(B, N) > 0.5
            absolute_starts = torch.rand(B)

            for i in range(B):
                evaluator.update(
                    timestamps=timestamps[i][masks[i]] + absolute_starts[i],
                    preds=preds[i][masks[i]],
                    targets=targets[i][masks[i]],
                    recording_id=batch_session_ids[i],
                )

        metric_dict = evaluator.compute()
        assert len(metric_dict) == num_sessions

        evaluator.reset()
