"""torch_brain.utils benchmarks.

bin_spikes over spike windows at production-typical sizes. bin_spikes is
imported lazily inside the driver (not at module scope) so importing this
module never requires torch_brain.utils to exist -- the rest of the suite then
still runs against commits/sources predating the module, where only the
bin_spikes benchmarks error out in isolation.

The sys.path shim that resolves ``torch_brain`` lives in benchmark.py and runs
before this module is imported (see TORCH_BRAIN_SOURCE).
"""

from __future__ import annotations

import numpy as np
from harness import bench

from torch_brain.data import Interval, IrregularTimeSeries


def _run_bin_spikes(label, n_units, n_bins, n_spikes, duration, number):
    """Shared driver: build one spike window and time bin_spikes over it.

    bin_spikes is imported lazily so the rest of the suite still runs against
    commits/sources where torch_brain.utils is unavailable.
    """
    from torch_brain.utils.binning import bin_spikes

    rng = np.random.RandomState(42)
    bin_size = duration / n_bins
    ts = np.sort(rng.uniform(0.0, duration, n_spikes))
    spikes = IrregularTimeSeries(
        timestamps=ts,
        unit_index=rng.randint(0, n_units, n_spikes),
        domain=Interval(0.0, duration),
    )

    def go():
        bin_spikes(spikes, num_units=n_units, bin_size=bin_size)

    return bench(label, go, number=number)


def bench_bin_spikes_realistic():
    """Real per-__getitem__ session window: 1s, 358 units, 20 bins (~3.5k)."""
    return _run_bin_spikes(
        "bin_spikes (1s, 358u, 20 bins)",
        n_units=358,
        n_bins=20,
        n_spikes=3_538,
        duration=1.0,
        number=2_000,
    )


def bench_bin_spikes_many_units():
    """Large population, coarse bins: 1s, 1024 units, 20 bins (~10k)."""
    return _run_bin_spikes(
        "bin_spikes (1s, 1024u, 20 bins)",
        n_units=1024,
        n_bins=20,
        n_spikes=10_000,
        duration=1.0,
        number=2_000,
    )


def bench_bin_spikes_fine_bins():
    """Fine time resolution: 1s, 358 units, 200 bins (5ms bins, ~3.5k)."""
    return _run_bin_spikes(
        "bin_spikes (1s, 358u, 200 bins)",
        n_units=358,
        n_bins=200,
        n_spikes=3_538,
        duration=1.0,
        number=2_000,
    )


def bench_bin_spikes_large():
    """Long window + large population: 10s, 1024 units, 500 bins (~50k)."""
    return _run_bin_spikes(
        "bin_spikes (10s, 1024u, 500 bins)",
        n_units=1024,
        n_bins=500,
        n_spikes=50_000,
        duration=10.0,
        number=500,
    )


BENCHMARKS = [
    bench_bin_spikes_realistic,
    bench_bin_spikes_many_units,
    bench_bin_spikes_fine_bins,
    bench_bin_spikes_large,
]
