import os

import numpy as np
from bokeh.plotting import figure, show

from torch_brain.data import Dataset

from _utils import download_file_from_s3

root_dir = os.path.dirname(__file__)
download_file_from_s3(
    "_ressources/c_20131003_center_out_reaching.h5",
    os.path.join(
        root_dir, "perich_miller_population_2018/c_20131003_center_out_reaching.h5"
    ),
)

dataset = Dataset(
    root_dir,
    recording_id="perich_miller_population_2018/c_20131003_center_out_reaching",
    split="train",
)

sampling_intervals = dataset.get_sampling_intervals()[
    "perich_miller_population_2018/c_20131003_center_out_reaching"
]

# Create figure
p = figure(
    width=800, height=200, x_axis_label="Time (s)", y_range=(-0.2, 1.2), tools="hover"
)
# Add hover tooltip
p.hover.tooltips = [("start", "@left{0.2f} s"), ("end", "@right{0.2f} s")]

# Plot each interval as a block
intervals_data = {
    "left": sampling_intervals.start,
    "right": sampling_intervals.end,
    "top": np.ones_like(sampling_intervals.start),
    "bottom": np.zeros_like(sampling_intervals.start),
}

p.quad(
    top="top",
    bottom="bottom",
    left="left",
    right="right",
    source=intervals_data,
    alpha=0.3,
    color="gray",
)

# Set axis limits with padding
p.x_range.start = sampling_intervals.start[0] - 10
p.x_range.end = sampling_intervals.end[-1] + 10

# Remove y-axis ticks since they're not meaningful
p.yaxis.visible = False

show(p)
