import os

import numpy as np
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Button, Div
from bokeh.models.callbacks import CustomJS
from bokeh.layouts import layout
from omegaconf import OmegaConf

from torch_brain.data import Dataset
from torch_brain.data.sampler import RandomFixedWindowSampler

from _utils import download_file_from_s3

config_str = """
- selection:
  - brainset: perich_miller_population_2018
    sessions:
      - c_20131003_center_out_reaching
      - c_20131022_center_out_reaching
      - c_20131023_center_out_reaching
"""

config = OmegaConf.create(config_str)

root_dir = os.path.dirname(__file__)
download_file_from_s3(
    "_ressources/c_20131003_center_out_reaching.h5",
    os.path.join(
        root_dir, "perich_miller_population_2018/c_20131003_center_out_reaching.h5"
    ),
)
download_file_from_s3(
    "_ressources/c_20131022_center_out_reaching.h5",
    os.path.join(
        root_dir, "perich_miller_population_2018/c_20131022_center_out_reaching.h5"
    ),
)
download_file_from_s3(
    "_ressources/c_20131023_center_out_reaching.h5",
    os.path.join(
        root_dir, "perich_miller_population_2018/c_20131023_center_out_reaching.h5"
    ),
)

dataset = Dataset(
    root_dir,
    recording_id="perich_miller_population_2018/c_20131003_center_out_reaching",
    split=None,
)
dataset = Dataset(
    root_dir,
    config=config,
    split="train",
)

sampler = RandomFixedWindowSampler(
    sampling_intervals=dataset.get_sampling_intervals(),
    window_length=1.0,
    generator=None,
)


sampling_intervals = dataset.get_sampling_intervals()

sampling_intervals_1 = sampling_intervals[
    "perich_miller_population_2018/c_20131003_center_out_reaching"
]
sampling_intervals_2 = sampling_intervals[
    "perich_miller_population_2018/c_20131022_center_out_reaching"
]
sampling_intervals_3 = sampling_intervals[
    "perich_miller_population_2018/c_20131023_center_out_reaching"
]

# Create single figure
p = figure(
    width=800,
    height=300,
    x_axis_label="Time (s)",
    y_range=(-0.1, 0.9),
    tools=["hover", "xwheel_zoom", "xpan", "reset"],
    active_scroll="xwheel_zoom",
)
p.x_range.bounds = (
    min(
        sampling_intervals_1.start[0],
        sampling_intervals_2.start[0],
        sampling_intervals_3.start[0],
    )
    - 10,
    max(
        sampling_intervals_1.end[-1],
        sampling_intervals_2.end[-1],
        sampling_intervals_3.end[-1],
    )
    + 10,
)

# Remove horizontal grid lines while keeping vertical ones
p.xgrid.grid_line_color = "#E5E5E5"
p.ygrid.grid_line_color = None

# Add hover tooltip only for sampled windows
p.hover.tooltips = [("start", "@left{0.2f} s"), ("end", "@right{0.2f} s")]

# Set axis limits with padding
p.x_range.start = (
    min(
        sampling_intervals_1.start[0],
        sampling_intervals_2.start[0],
        sampling_intervals_3.start[0],
    )
    - 10
)
p.x_range.end = (
    max(
        sampling_intervals_1.end[-1],
        sampling_intervals_2.end[-1],
        sampling_intervals_3.end[-1],
    )
    + 10
)
# Remove y-axis ticks
p.yaxis.visible = False

# Plot original intervals
# Create data for all three interval sets
intervals_data_1 = {
    "left": sampling_intervals_1.start,
    "right": sampling_intervals_1.end,
    "top": 0.8 * np.ones_like(sampling_intervals_1.start),
    "bottom": 0.6 * np.ones_like(sampling_intervals_1.start),
}
intervals_data_2 = {
    "left": sampling_intervals_2.start,
    "right": sampling_intervals_2.end,
    "top": 0.5 * np.ones_like(sampling_intervals_2.start),
    "bottom": 0.3 * np.ones_like(sampling_intervals_2.start),
}
intervals_data_3 = {
    "left": sampling_intervals_3.start,
    "right": sampling_intervals_3.end,
    "top": 0.2 * np.ones_like(sampling_intervals_3.start),
    "bottom": 0.0 * np.ones_like(sampling_intervals_3.start),
}

# Create ColumnDataSources for each interval set
intervals_source_1 = ColumnDataSource(intervals_data_1)
intervals_source_2 = ColumnDataSource(intervals_data_2)
intervals_source_3 = ColumnDataSource(intervals_data_3)

# Plot each interval set
intervals_1 = p.quad(
    top="top",
    bottom="bottom",
    left="left",
    right="right",
    color="gray",
    source=intervals_source_1,
    alpha=0.3,
)
intervals_2 = p.quad(
    top="top",
    bottom="bottom",
    left="left",
    right="right",
    color="gray",
    source=intervals_source_2,
    alpha=0.3,
)
intervals_3 = p.quad(
    top="top",
    bottom="bottom",
    left="left",
    right="right",
    color="gray",
    source=intervals_source_3,
    alpha=0.3,
)

# Add y-axis labels
p.yaxis.visible = True
p.yaxis.ticker = [0.1, 0.4, 0.7]  # Center of each interval band
p.yaxis.major_label_overrides = {
    0.1: "c_20131023_center_out_reaching",
    0.4: "c_20131022_center_out_reaching",
    0.7: "c_20131003_center_out_reaching",
}


# Create data source for sampled windows
sampled_windows_data = {"left": [], "right": [], "top": [], "bottom": []}
sampled_source = ColumnDataSource(sampled_windows_data)
windows = p.quad(
    top="top",
    bottom="bottom",
    left="left",
    right="right",
    source=sampled_source,
    alpha=0.5,
    color="red",
    line_color="black",
    line_width=0.3,
)
# p.legend.click_policy = "hide"

# Move legend outside of plot area and stack horizontally
# p.legend.location = "top_right"
# p.legend.click_policy = "hide"
# p.legend.border_line_color = None
# p.legend.background_fill_alpha = 0
# p.legend.orientation = "horizontal"
# p.add_layout(p.legend[0], 'above')


def sample_windows(sampler):
    sampled_windows = []
    sampler_iter = iter(sampler)
    for _ in range(len(sampler)):
        try:
            idx = next(sampler_iter)
            sampled_windows.append(
                {"recording_id": idx.recording_id, "start": idx.start, "end": idx.end}
            )
        except StopIteration:
            break
    return sampled_windows


num_samples = len(sampler)

# Sample windows upfront
sampled_windows = sample_windows(sampler)

# Add controls
div = Div(
    text="""<p>Click the Generate samples button to start sampling windows of 1s.</p>""",
    width=400,
    # height=30,
)

# Create auto-play callback
auto_callback = CustomJS(
    args={
        "source": sampled_source,
        "sampled_windows": sampled_windows,
        "button": None,  # Add button reference that will be updated later
    },
    code="""
    // Disable button while playing
    button.disabled = true;
    
    // Clear existing windows
    source.data['left'] = [];
    source.data['right'] = [];
    source.data['top'] = [];
    source.data['bottom'] = [];
    source.change.emit();
    
    let current_window_idx = 0;
    
    function playNextWindow() {
        if (current_window_idx < sampled_windows.length) {
            var window = sampled_windows[current_window_idx];
            
            // Get the y position based on recording_id
            var top, bottom;
            if (window.recording_id.includes('20131003')) {
                top = 0.8;
                bottom = 0.6;
            } else if (window.recording_id.includes('20131022')) {
                top = 0.5;
                bottom = 0.3;
            } else {
                top = 0.2;
                bottom = 0.0;
            }
            
            var data = source.data;
            data['left'].push(window.start);
            data['right'].push(window.end);
            data['top'].push(top);
            data['bottom'].push(bottom);
            
            source.change.emit();
            current_window_idx++;
            
            setTimeout(playNextWindow, 10);  
        } else {
            // All windows played
            button.disabled = false;
            button.label = 'Generate samples for epoch 1';
        }
    }
    
    // Start playing windows
    playNextWindow();
""",
)

auto_button = Button(
    label="Generate samples for epoch 1", button_type="primary", width=200
)
auto_callback.args["button"] = auto_button
auto_button.js_on_click(auto_callback)  # Add this line to connect the callback

# Create layout and show
show(layout([[div], [auto_button], [p]]))
