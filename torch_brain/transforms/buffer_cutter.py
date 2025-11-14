import torch

class Buffer_Cutter:
    def __init__(self, buffer_len):
        self.buffer_len = buffer_len

    def __call__(self, data):
        sequence_len = data.domain.end[-1] - data.domain.start[0]

        if sequence_len <= self.buffer_len:
            return data

        start = data.domain.start[0]+self.buffer_len
        end = data.domain.end[-1]-self.buffer_len

        return data.slice(start, end)
