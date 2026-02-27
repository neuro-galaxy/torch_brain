import torch


class RandomCrop:
    r"""Randomly crop a fixed-length window from the data.

    If the data is shorter than the crop length, the original data is returned.

    Args:
        crop_len (float): Length of the crop window.
    """

    def __init__(self, crop_len):
        self.crop_len = crop_len

    def __call__(self, data):
        sequence_len = data.end - data.start

        if sequence_len <= self.crop_len:
            return data

        start = torch.rand(1).item() * (sequence_len - self.crop_len)
        end = start + self.crop_len

        return data.slice(start, end)
