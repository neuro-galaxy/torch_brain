# need to make it as a proper test, it is not now...
# test with random one

import torch
import random
from types import SimpleNamespace
from torch_brain.data.sampler import SequentialFixedWindowSampler
from torch_brain.data.sampler import RandomFixedWindowSampler
from torch_brain.data.sampler import TrialSampler
from torch_brain.data.buffer_sampler import BufferedSampler

def make_fake_sampling_intervals(num_records=5, max_time=100, n_interval=5):
    """
    Create a fake sampling_intervals dict:
    {recording_id: domain} where domain has start and end lists.
    """
    sampling_intervals = {}
    for i in range(num_records):
        recording_id = f"recording_{i+1}"
        n_interval = n_interval #random.randint(1, max_intervals)
        starts = sorted(random.sample(range(max_time), n_interval))
        ends = [s + random.randint(10, 20) for s in starts]

        domain = SimpleNamespace(start=starts, end=ends)
        sampling_intervals[recording_id] = domain
    return sampling_intervals

# Example usage
fake_sampling_intervals = make_fake_sampling_intervals(num_records=1)
for rec_id, domain in fake_sampling_intervals.items():
    print(rec_id, domain.start, domain.end)

# sampler = SequentialFixedWindowSampler(
#    sampling_intervals=fake_sampling_intervals,
#    window_length=5.0,
# )

sampler = TrialSampler(
   sampling_intervals=fake_sampling_intervals,
)

# sampler = RandomFixedWindowSampler(
#     sampling_intervals=fake_sampling_intervals,
#     window_length=5.0,
#     generator = torch.Generator().manual_seed(1115), 
# )

#base_windows = list(sampler)
for didx in sampler:
    print(f'start: {didx.start:.2f}, end: {didx.end:.2f}')

print("=======")

buffered_sampler = BufferedSampler(
    base_sampler = sampler,
    sampling_intervals=fake_sampling_intervals,
    buffer_len = 3.0,
    IsTrialSampler = True,
)

for didx in buffered_sampler:
    print(f'start: {didx.start:.2f}, end: {didx.end:.2f}')