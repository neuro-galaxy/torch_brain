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
    split=None,
)

data = dataset.get_recording_data(
    "perich_miller_population_2018/c_20131003_center_out_reaching"
)

train_sampling_intervals = data.train_domain
valid_sampling_intervals = data.valid_domain
test_sampling_intervals = data.test_domain
# Create figure
p = figure(
    width=800, height=250, x_axis_label="Time (s)", y_range=(-0.2, 1.2), tools="hover"
)
# Add hover tooltip
p.hover.tooltips = [
    ("start", "@left{0.2f} s"),
    ("end", "@right{0.2f} s"),
    ("split", "@type"),
]
# Remove horizontal grid lines while keeping vertical ones
p.xgrid.grid_line_color = "#E5E5E5"
p.ygrid.grid_line_color = None

# Plot train intervals
train_data = {
    "left": train_sampling_intervals.start,
    "right": train_sampling_intervals.end,
    "top": 0.8 * np.ones_like(train_sampling_intervals.start),
    "bottom": np.zeros_like(train_sampling_intervals.start),
    "type": ["train"] * len(train_sampling_intervals.start),
}
p.quad(
    top="top",
    bottom="bottom",
    left="left",
    right="right",
    source=train_data,
    alpha=0.3,
    color="blue",
    legend_label="Train",
)

# Plot validation intervals
valid_data = {
    "left": valid_sampling_intervals.start,
    "right": valid_sampling_intervals.end,
    "top": 0.8 * np.ones_like(valid_sampling_intervals.start),
    "bottom": np.zeros_like(valid_sampling_intervals.start),
    "type": ["validation"] * len(valid_sampling_intervals.start),
}
p.quad(
    top="top",
    bottom="bottom",
    left="left",
    right="right",
    source=valid_data,
    alpha=0.3,
    color="green",
    legend_label="Validation",
)

# Plot test intervals
test_data = {
    "left": test_sampling_intervals.start,
    "right": test_sampling_intervals.end,
    "top": 0.8 * np.ones_like(test_sampling_intervals.start),
    "bottom": np.zeros_like(test_sampling_intervals.start),
    "type": ["test"] * len(test_sampling_intervals.start),
}
p.quad(
    top="top",
    bottom="bottom",
    left="left",
    right="right",
    source=test_data,
    alpha=0.3,
    color="red",
    legend_label="Test",
)

# Set axis limits with padding
all_starts = np.concatenate(
    [
        train_sampling_intervals.start,
        valid_sampling_intervals.start,
        test_sampling_intervals.start,
    ]
)
all_ends = np.concatenate(
    [
        train_sampling_intervals.end,
        valid_sampling_intervals.end,
        test_sampling_intervals.end,
    ]
)
p.x_range.start = np.min(all_starts) - 10
p.x_range.end = np.max(all_ends) + 10

# Remove y-axis ticks since they're not meaningful
p.yaxis.visible = False

# Configure legend
p.legend.location = "top_right"
p.legend.click_policy = "hide"
p.legend.orientation = "horizontal"

# Show the plot
show(p)
