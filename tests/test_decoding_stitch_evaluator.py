import pytest
import torch
import torchmetrics

from torch_brain.registry import MODALITY_REGISTRY
from torch_brain.utils.stitcher import DecodingStitchEvaluator


@pytest.fixture
def mock_session_ids():
    return [
        "session1",
        "session2",
        "session3",
    ]


def test_initialization(mock_session_ids):

    # Test for R2Score
    evaluator = DecodingStitchEvaluator(
        session_ids=mock_session_ids,
        modality_spec=MODALITY_REGISTRY["cursor_velocity_2d"],
    )
    assert list(evaluator.metrics.keys()) == mock_session_ids
    for metric in evaluator.metrics.values():
        assert isinstance(metric, torchmetrics.R2Score)

    # Test for Accuracy
    evaluator = DecodingStitchEvaluator(
        session_ids=mock_session_ids,
        modality_spec=MODALITY_REGISTRY["drifting_gratings_orientation"],
    )
    assert list(evaluator.metrics.keys()) == mock_session_ids
    for metric in evaluator.metrics.values():
        assert isinstance(metric, torchmetrics.classification.MulticlassAccuracy)

    # Test custom metric factory
    metric_cls = torchmetrics.classification.BinaryAccuracy
    evaluator = DecodingStitchEvaluator(
        session_ids=mock_session_ids,
        metric_factory=metric_cls,
    )
    assert list(evaluator.metrics.keys()) == mock_session_ids
    for metric in evaluator.metrics.values():
        assert isinstance(metric, metric_cls)


def test_update(mock_session_ids):
    evaluator = DecodingStitchEvaluator(
        session_ids=mock_session_ids,
        modality_spec=MODALITY_REGISTRY["cursor_velocity_2d"],
    )

    B = 3  # batch size
    N = 17  # tokens per sample
    D = 2  # output dim (for cursor velocity 2d)

    def step_with_one_session(session_id):
        timestamps = torch.rand(B, N)
        absolute_starts = torch.rand(B)
        preds = torch.rand(B, N, D)
        targets = torch.rand(B, N, D)
        mask = torch.rand(B, N) > 0.5

        evaluator.update(
            timestamps=timestamps,
            preds=preds,
            targets=targets,
            eval_masks=mask,
            session_ids=[session_id] * B,
            absolute_starts=absolute_starts,
        )

        expected_timestamps = (timestamps + absolute_starts[:, None])[mask]
        return expected_timestamps, preds[mask], targets[mask]

    # Test with first session
    sess_id1 = mock_session_ids[0]
    exp_times1, exp_preds1, exp_targets1 = step_with_one_session(sess_id1)

    assert len(evaluator._cache) == 1
    assert torch.equal(torch.cat(evaluator._cache[sess_id1]["timestamps"]), exp_times1)
    assert torch.equal(torch.cat(evaluator._cache[sess_id1]["pred"]), exp_preds1)
    assert torch.equal(torch.cat(evaluator._cache[sess_id1]["target"]), exp_targets1)

    # Step again with same session
    exp_times2, exp_preds2, exp_targets2 = step_with_one_session(sess_id1)
    exp_times2 = torch.cat((exp_times1, exp_times2))
    exp_preds2 = torch.cat((exp_preds1, exp_preds2))
    exp_targets2 = torch.cat((exp_targets1, exp_targets2))

    assert len(evaluator._cache) == 1
    assert torch.equal(torch.cat(evaluator._cache[sess_id1]["timestamps"]), exp_times2)
    assert torch.equal(torch.cat(evaluator._cache[sess_id1]["pred"]), exp_preds2)
    assert torch.equal(torch.cat(evaluator._cache[sess_id1]["target"]), exp_targets2)

    # Step with 2nd session
    sess_id2 = mock_session_ids[1]
    exp_times3, exp_preds3, exp_targets3 = step_with_one_session(sess_id2)

    assert len(evaluator._cache) == 2
    # First session cache should be unchanged
    assert torch.equal(torch.cat(evaluator._cache[sess_id1]["timestamps"]), exp_times2)
    assert torch.equal(torch.cat(evaluator._cache[sess_id1]["pred"]), exp_preds2)
    assert torch.equal(torch.cat(evaluator._cache[sess_id1]["target"]), exp_targets2)
    # Second session cache should match new data
    assert torch.equal(torch.cat(evaluator._cache[sess_id2]["timestamps"]), exp_times3)
    assert torch.equal(torch.cat(evaluator._cache[sess_id2]["pred"]), exp_preds3)
    assert torch.equal(torch.cat(evaluator._cache[sess_id2]["target"]), exp_targets3)


def test_end_to_end_r2(mock_session_ids):
    B = 16  # batch size
    N = 32  # tokens per sample
    D = 2  # prediction dimension (for cursor velocity 2d)
    num_sessions = len(mock_session_ids)

    evaluator = DecodingStitchEvaluator(
        session_ids=mock_session_ids,
        modality_spec=MODALITY_REGISTRY["cursor_velocity_2d"],
    )

    for epoch in range(3):
        assert len(evaluator._cache) == 0

        for batch_step in range(10):
            evaluator.update(
                timestamps=torch.linspace(0, 1, N).repeat(B, 1),
                preds=torch.rand(B, N, D),
                targets=torch.rand(B, N, D),
                eval_masks=torch.rand(B, N) > 0.5,
                session_ids=[
                    mock_session_ids[idx] for idx in torch.arange(B) % num_sessions
                ],
                absolute_starts=torch.rand(B),
            )

        metric_dict = evaluator.compute()
        assert len(metric_dict) == num_sessions

        evaluator.reset()


def test_end_to_end_accuracy(mock_session_ids):
    B = 9  # batch size
    N = 50  # tokens per sample
    D = 8  # prediction dimension (for drifting gratings orientation)
    num_sessions = len(mock_session_ids)

    evaluator = DecodingStitchEvaluator(
        session_ids=mock_session_ids,
        modality_spec=MODALITY_REGISTRY["drifting_gratings_orientation"],
    )

    for epoch in range(3):
        assert len(evaluator._cache) == 0

        for batch_step in range(10):
            evaluator.update(
                timestamps=torch.linspace(0, 1, N).repeat(B, 1),
                preds=torch.rand(B, N, D),
                targets=torch.randint(D, (B, N, D)),
                eval_masks=torch.rand(B, N) > 0.5,
                session_ids=[
                    mock_session_ids[idx] for idx in torch.arange(B) % num_sessions
                ],
                absolute_starts=torch.rand(B),
            )

        metric_dict = evaluator.compute()
        assert len(metric_dict) == num_sessions

        evaluator.reset()
