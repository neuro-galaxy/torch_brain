import torch


class RandomOutputSampler:
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
