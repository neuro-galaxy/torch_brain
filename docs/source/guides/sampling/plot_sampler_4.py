import os

import numpy as np
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Button, Div
from bokeh.models.callbacks import CustomJS
from bokeh.layouts import layout

from torch_brain.data import Dataset
from torch_brain.data.sampler import RandomFixedWindowSampler

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

sampler = RandomFixedWindowSampler(
    sampling_intervals=dataset.get_sampling_intervals(),
    window_length=5.0,
    generator=None,
)

sampling_intervals = dataset.get_sampling_intervals()[
    "perich_miller_population_2018/c_20131003_center_out_reaching"
]
# Create single figure
p = figure(
    width=800,
    height=300,
    x_axis_label="Time (s)",
    y_range=(0.0, 1.0),
    tools=["hover", "xwheel_zoom", "xpan", "reset"],
    active_scroll="xwheel_zoom",
)
p.x_range.bounds = (sampling_intervals.start[0] - 10, sampling_intervals.end[-1] + 10)


# Remove horizontal grid lines while keeping vertical ones
p.xgrid.grid_line_color = "#E5E5E5"
p.ygrid.grid_line_color = None

# Add hover tooltip only for sampled windows
p.hover.tooltips = [("start", "@left{0.2f} s"), ("end", "@right{0.2f} s")]

# Set axis limits with padding
p.x_range.start = sampling_intervals.start[0] - 10
p.x_range.end = sampling_intervals.end[-1] + 10
# Remove y-axis ticks
p.yaxis.visible = False

# Plot original intervals
intervals_data = {
    "left": sampling_intervals.start,
    "right": sampling_intervals.end,
    "top": 0.4 * np.ones_like(sampling_intervals.start),
    "bottom": 0.1 * np.ones_like(sampling_intervals.start),
}
intervals_source = ColumnDataSource(intervals_data)
intervals = p.quad(
    top="top",
    bottom="bottom",
    left="left",
    right="right",
    color="gray",
    source=intervals_source,
    alpha=0.3,
    legend_label="Sampling Intervals",
)

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
    legend_label="Sampled Windows",
)
p.legend.click_policy = "hide"

# Move legend outside of plot area and stack horizontally
p.legend.location = "top_right"
p.legend.click_policy = "hide"
p.legend.border_line_color = None
p.legend.background_fill_alpha = 0
p.legend.orientation = "horizontal"
# p.add_layout(p.legend[0], 'above')


def sample_windows(sampler):
    sampled_windows = []
    sampler_iter = iter(sampler)
    for _ in range(len(sampler)):
        try:
            idx = next(sampler_iter)
            sampled_windows.append({"start": idx.start, "end": idx.end})
        except StopIteration:
            break
    return sampled_windows


num_samples = len(sampler)


# Sample windows upfront
sampled_windows_1 = sample_windows(sampler)
sampled_windows_2 = sample_windows(sampler)
sampled_windows_3 = sample_windows(sampler)

# Add controls
div = Div(
    text="""<p>Click the Generate samples button to start sampling windows of 5s.</p>""",
    width=400,
    # height=30,
)

# Create auto-play callback
auto_callback = CustomJS(
    args={
        "source": sampled_source,
        "windows_1": sampled_windows_1,
        "windows_2": sampled_windows_2,
        "windows_3": sampled_windows_3,
        "current_windows": sampled_windows_1.copy(),  # Start with first set
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
    
    // Initialize window set if not already set
    if (!button.window_set) {
        button.window_set = 1;
    }
    
    function playNextWindow() {
        if (current_windows.length > 0) {
            var window = current_windows.shift();  // Remove and get first window
            
            var data = source.data;
            data['left'].push(window.start);
            data['right'].push(window.end);
            data['top'].push(0.8);
            data['bottom'].push(0.5);
            
            source.change.emit();
            
            setTimeout(playNextWindow, 50);
        } else {
            // All windows played - prepare for next seed
            button.disabled = false;
            
            if (button.window_set === 1) {
                current_windows.push.apply(current_windows, windows_2);
                button.window_set = 2;
                button.label = 'Generate samples for epoch 2';
            } else if (button.window_set === 2) {
                current_windows.push.apply(current_windows, windows_3);
                button.window_set = 3;
                button.label = 'Generate samples for epoch 3';
            } else {
                current_windows.push.apply(current_windows, windows_1);
                button.window_set = 1;
                button.label = 'Generate samples for epoch 1';
            }
        }
    }
    
    // Start playing windows
    playNextWindow();
""",
)

auto_button = Button(
    label="Generate samples for epoch 1", button_type="primary", width=100
)
auto_callback.args["button"] = auto_button

# Enable HTML rendering for the label
auto_button.css_classes = ["fa-button"]
auto_button.js_on_click(auto_callback)

# Create layout and show
show(layout([[div], [auto_button], [p]]))
