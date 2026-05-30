import torch


class RandomCrop:
    def __init__(self, crop_len):
        self.crop_len = crop_len

    def __call__(self, data):
        sequence_len = data.end - data.start

        if sequence_len <= self.crop_len:
            return data

        start = torch.rand(1).item() * (sequence_len - self.crop_len)
        end = start + self.crop_len

        return data.slice(start, end)
