import torch


class RandomOutputSampler:
    r"""Randomly samples a subset of output tokens from the data.

    This transform randomly selects ``num_output_tokens`` from the behavior
    timestamps. If the number of available timestamps is less than or equal
    to ``num_output_tokens``, all timestamps are kept.

    Note: This transform assumes that the data object has a ``behavior`` attribute.

    Args:
        num_output_tokens (int): Maximum number of output tokens to sample.
    """

    def __init__(self, num_output_tokens):
        self.num_output_tokens = num_output_tokens

    def __call__(self, data):
        out = data.behavior
        timestamps = out.timestamps

        if len(timestamps) <= self.num_output_tokens:
            return data

        # sample from timestamps
        mask = torch.zeros(len(timestamps), dtype=bool)
        mask[torch.randperm(len(timestamps))[: self.num_output_tokens]] = True

        for key, value in out.__dict__.items():
            out.__dict__[key] = value[mask].clone()

        data.behavior = out
        return data
