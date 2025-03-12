import torch
from torch_brain.utils.stitcher import stitch


def test_stitch_float():
    in_timestamps = torch.tensor([0.1, 0.3, 0.42, 0.5, 0.3, 0.3, 0.5])
    expected_out_timestamsp = torch.tensor([0.1, 0.3, 0.42, 0.5])

    # Test on a (N, D) shaped input
    D = 3
    in_values = torch.rand((len(in_timestamps), D))
    expected_out_values = torch.stack(
        [
            in_values[0],
            (in_values[1] + in_values[4] + in_values[5]) / 3,
            in_values[2],
            (in_values[3] + in_values[6]) / 2,
        ]
    )

    out_timestamps, out_values = stitch(in_timestamps, in_values)
    assert torch.equal(out_values, expected_out_values)
    assert torch.equal(out_timestamps, expected_out_timestamsp)

    # Test on (N,) shaped input
    in_values = torch.rand(len(in_timestamps))
    expected_out_values = torch.tensor(
        [
            in_values[0],
            (in_values[1] + in_values[4] + in_values[5]) / 3,
            in_values[2],
            (in_values[3] + in_values[6]) / 2,
        ]
    )

    out_timestamps, out_values = stitch(in_timestamps, in_values)
    assert torch.equal(out_values, expected_out_values)
    assert torch.equal(out_timestamps, expected_out_timestamsp)


def test_stitch_long():
    in_timestamps = torch.tensor([0.1, 0.3, 0.42, 0.5, 0.3, 0.3, 0.5, 0.15, 0.15])
    expected_out_timestamsp = torch.tensor([0.1, 0.15, 0.3, 0.42, 0.5])

    in_values = torch.LongTensor([0, 1, 2, 3, 1, 2, 3, 6, 5])
    expected_outputs = torch.LongTensor([0, 5, 1, 2, 3])

    out_timestamps, out_values = stitch(in_timestamps, in_values)
    assert torch.equal(out_values, expected_outputs)
    assert torch.equal(out_timestamps, expected_out_timestamsp)
