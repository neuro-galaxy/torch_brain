import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torch_brain.data import Interval

cmap = mpl.colormaps["Accent"]
height = 0.8
half_height = height / 2
grid_alpha = 0.4


def draw_interval(ax, interval, y, color):
    for s, e in zip(interval.start, interval.end):
        rect = Rectangle(
            (s, y - half_height),
            e - s,
            height,
            facecolor=color,
            edgecolor="white",
        )
        ax.add_patch(rect)


def plot_intersection():
    interval1 = Interval(start=[1.0, 12.0], end=[8.0, 18.0])
    interval2 = Interval(start=[2.0, 7.0, 14.0], end=[5.0, 10.0, 17.0])
    intersection = interval1 & interval2

    fig, ax = plt.subplots(figsize=(7, 2))

    draw_interval(ax, interval1, 3.0, cmap(0))
    draw_interval(ax, interval2, 2.0, cmap(1))
    draw_interval(ax, intersection, 1.0, cmap(2))

    max_t = int(19.0)
    ax.set_xlim(0.0, max_t)
    ax.set_ylim(half_height, 3.0 + half_height)
    ax.set_yticks([1.0, 2.0, 3.0], ["I1 & I2", "I2", "I1"])
    ax.set_xticks([i for i in range(0, max_t, 2)], [str(i) for i in range(0, max_t, 2)])

    ax.grid(True, axis="x", alpha=grid_alpha)
    ax.set_axisbelow(True)  # force grid below objects
    for spine in ("top", "left", "right"):
        ax.spines[spine].set_visible(False)


def plot_union():
    interval1 = Interval(start=[1.0, 12.0], end=[8.0, 18.0])
    interval2 = Interval(start=[2.0, 7.0, 14.0], end=[5.0, 10.0, 17.0])
    union = interval1 | interval2

    fig, ax = plt.subplots(figsize=(7, 2))

    draw_interval(ax, interval1, 3.0, cmap(0))
    draw_interval(ax, interval2, 2.0, cmap(1))
    draw_interval(ax, union, 1.0, cmap(2))

    max_t = int(19.0)
    ax.set_xlim(0.0, max_t)
    ax.set_ylim(half_height, 3.0 + half_height)
    ax.set_yticks([1.0, 2.0, 3.0], ["I1 | I2", "I2", "I1"])
    ax.set_xticks([i for i in range(0, max_t, 2)], [str(i) for i in range(0, max_t, 2)])

    ax.grid(True, axis="x", alpha=grid_alpha)
    ax.set_axisbelow(True)  # force grid below objects
    for spine in ("top", "left", "right"):
        ax.spines[spine].set_visible(False)


def plot_difference():
    interval1 = Interval(start=[1.0, 12.0], end=[8.0, 18.0])
    interval2 = Interval(start=[2.0, 7.0, 14.0], end=[5.0, 10.0, 17.0])
    diff = interval1.difference(interval2)

    fig, ax = plt.subplots(figsize=(7, 2))

    draw_interval(ax, interval1, 3.0, cmap(0))
    draw_interval(ax, interval2, 2.0, cmap(1))
    draw_interval(ax, diff, 1.0, cmap(2))

    max_t = int(19.0)
    ax.set_xlim(0.0, max_t)
    ax.set_ylim(half_height, 3.0 + half_height)
    ax.set_yticks([1.0, 2.0, 3.0], ["I1 - I2", "I2", "I1"])
    ax.set_xticks([i for i in range(0, max_t, 2)], [str(i) for i in range(0, max_t, 2)])

    ax.grid(True, axis="x", alpha=grid_alpha)
    ax.set_axisbelow(True)  # force grid below objects
    for spine in ("top", "left", "right"):
        ax.spines[spine].set_visible(False)


def plot_dilation():
    interval = Interval(start=[1.0, 10.0, 14.0], end=[5.0, 13.5, 18.0])
    dilated = interval.dilate(0.5)

    fig, ax = plt.subplots(figsize=(7.5, 1.5))

    draw_interval(ax, interval, 2.0, cmap(0))
    draw_interval(ax, dilated, 1.0, cmap(1))

    max_t = int(19.0)
    ax.set_xlim(0.0, max_t)
    ax.set_ylim(half_height, 2.0 + half_height)
    ax.set_yticks([1.0, 2.0], ["dilated", "original"])
    ax.set_xticks([i for i in range(0, max_t, 2)], [str(i) for i in range(0, max_t, 2)])

    ax.grid(True, axis="x", alpha=grid_alpha)
    ax.set_axisbelow(True)  # force grid below objects
    for spine in ("top", "left", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()


def plot_coalesce():
    interval = Interval(
        start=[1.0, 6.1, 11.3, 14.5],
        end=[6.0, 11.0, 14.5, 17.8],
    )
    coalesced = interval.coalesce(grid_alpha)

    fig, ax = plt.subplots(figsize=(7.5, 1.5))

    draw_interval(ax, interval, 2.0, cmap(0))
    draw_interval(ax, coalesced, 1.0, cmap(1))

    max_t = int(19.0)
    ax.set_xlim(0.0, max_t)
    ax.set_ylim(half_height, 2.0 + half_height)
    ax.set_yticks([1.0, 2.0], ["coalesced", "original"])
    ax.set_xticks([i for i in range(0, max_t, 2)], [str(i) for i in range(0, max_t, 2)])

    ax.grid(True, axis="x", alpha=grid_alpha)
    ax.set_axisbelow(True)  # force grid below objects
    for spine in ("top", "left", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
