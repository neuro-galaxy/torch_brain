.. _sampling_guide:

.. currentmodule:: torch_brain.samplers

Data Sampling
=============

Neural recordings are rarely tidy. A single session typically contains
periods of task engagement separated by long inter-trial intervals,
short artifacts that need to be excluded, and stretches where only a
subset of channels were recording cleanly. Different downstream tasks
also want different views of the same data — a decoder might train on
1-second windows while a self-supervised model trains on 10-second
ones — and real experiments often span multiple sessions or subjects
that have to be sampled from jointly.

**torch_brain** is designed around these realities. One of our major
design goals is to make it easy to sample *arbitrary* time windows
from multi-recording datasets, without having to reprocess the
underlying data every time the training recipe changes.

In this guide you will learn:

- what *sampling intervals* are, and how they tell a sampler where
  it is allowed to sample from,
- how to get these sampling intervals from a :obj:`~torch_brain.datasets.Dataset`,
- how to use the built-in samplers to generate fixed or variable length
  windows for training and evaluation,
- and how the same setup scales seamlessly to datasets that contain
  multiple recordings.

Concept: *Sampling Intervals*
-----------------------------

Sampling intervals are *Intervals* from which a data sampler is allowed
to sample data.

In **torch_brain**, the primary way of sampling is using *continuous
time* intervals. This is achieved with the following design:

- :obj:`~torch_brain.datasets.Dataset` tells the sampler where it is
  allowed to sample from. This is achieved with the
  :obj:`~torch_brain.datasets.Dataset.get_sampling_intervals` method.
- The sampler uses this information to create samples, which are
  effectively :math:`(t_{start}, t_{end})` pairs emitted as a
  :obj:`~torch_brain.datasets.DatasetIndex` object.
- :obj:`~torch_brain.datasets.Dataset` uses this sample information to load and slice
  the data samples.

.. figure:: static/overview.png
    :width: 80%
    :align: center

    Interplay between Dataset and Sampler.

.. tip::

    Typically, all this is orchestrated by a PyTorch :obj:`~torch.utils.data.DataLoader`.
    In this guide, we will go through the logic of sampling *manually*,
    to give you a more fundamental understanding of these concepts.
    See
    :doc:`this </generated/notebooks/nlb_maze_minimal_example>` minimal
    training example for how it all fits with a DataLoader.

Datasets in **torch_brain** typically contain multiple recordings, and
so we define sampling intervals as *dictionaries* keyed by the recording
IDs. Each entry is a :obj:`~torch_brain.data.Interval` object that specifies
the valid start and end sampling times for that recording:

.. code:: python

    sampling_intervals = {
        "recording_id1": Interval(...),
        "recording_id2": Interval(...),
        ...
    }

These intervals do not have to be contiguous, and can be of any length.

Many **brainsets** provide default train/validation/test sampling
intervals. For example, let’s load a single recording in the
:obj:`~torch_brain.datasets.PerichMillerPopulation2018` dataset.

If the dataset has not been downloaded and processed yet, we can first download 
(~40 MB) and process the specific recording by running, from the command line:

.. code-block:: console

    $ brainsets prepare perich_miller_population_2018 -s c_20131003_center_out_reaching

We can then load the recording and get its sampling intervals for the train split. 

.. note:: 
    We can specify which directory to load the data from, but if we set :obj:`~root=None`, 
    the data is loaded by default from the directory configured for storing processed 
    brainsets (see :ref:`brainsets config <cli_ref-brainsets-config>`).

.. code-block:: pycon

    >>> from torch_brain.datasets import PerichMillerPopulation2018

    >>> dataset = PerichMillerPopulation2018(
    ...     root=None, 
    ...     recording_ids=["c_20131003_center_out_reaching"],
    ... )
    >>> sampling_intervals = dataset.get_sampling_intervals("train")
    >>> print(sampling_intervals)
    {'c_20131003_center_out_reaching': LazyInterval(
      end: lazy array, shape=(38,), dtype=float64,
      start: lazy array, shape=(38,), dtype=float64,
    )}


We note that there are a total of 38 sampling intervals for the train
part of the recording. Let’s print the first 5:

.. code-block:: pycon

    >>> for recording_id in sampling_intervals:
    ...     for start, end in zip(sampling_intervals[recording_id].start[:5], sampling_intervals[recording_id].end[:5]):
    ...         print(f"start: {start:.2f}, end: {end:.2f}")
    start: 0.00, end: 38.51
    start: 44.02, end: 49.32
    start: 55.78, end: 60.20
    start: 65.15, end: 71.30
    start: 77.15, end: 83.56


Notice that the intervals are of different lengths. Visually, they look like:

.. figure:: static/sampler_10_0.png

    Train sampling intervals for one recording. Each gray block marks
    an interval that a sampler is allowed to draw from.

The same dataset also exposes validation and test splits via
``get_sampling_intervals("valid")`` and ``get_sampling_intervals("test")``.
Overlaying all three shows how the brainset partitions the recording:

.. figure:: static/sampler_12_0.png

    Train, validation, and test sampling intervals on the same recording.


Samplers in action
------------------

**torch_brain** provides a number of samplers that can be used to
generate samples for training, or evaluation. For example,


.. list-table::
    :widths: 25 125

    * - :obj:`~torch_brain.samplers.SequentialFixedWindowSampler`
      - A Sequential sampler, that samples fixed-length windows.
    * - :obj:`~torch_brain.samplers.RandomFixedWindowSampler`
      - A Random sampler, that samples fixed-length windows.
    * - :obj:`~torch_brain.samplers.TrialSampler`
      - A sampler that randomly samples a full contiguous interval without slicing it into windows.


RandomFixedWindowSampler
~~~~~~~~~~~~~~~~~~~~~~~~

The most common sampler used in practice is the
:obj:`~torch_brain.samplers.RandomFixedWindowSampler`,
which randomly samples windows of a fixed length from the data.

.. code-block:: pycon

    >>> import torch
    >>> from torch_brain.samplers import RandomFixedWindowSampler

    >>> torch_rng = torch.Generator()
    >>> torch_rng.manual_seed(123)  # seeded for reproducibility

    >>> sampler = RandomFixedWindowSampler(
    ...     sampling_intervals=dataset.get_sampling_intervals("train"),
    ...     window_length=1.0,
    ...     generator=torch_rng,
    ... )

    >>> print("Number of sampled windows in one epoch: ", len(sampler))
    WARNING:root:Skipping 1.8665333333332796 seconds of data due to short intervals. Remaining: 417.0 seconds.
    Number of sampled windows in one epoch:  417


This sampler will generate exactly 417 samples, and some small isolated
intervals of total length 1.86 seconds are skipped because they are too
short to sample 1 second windows from.

We can visualize what the sampler is doing as we are iterating over it.

.. code-block:: pycon

    >>> for i, sample_index in enumerate(sampler):
    ...     print(f"Sample between {sample_index.start:.2f} and {sample_index.end:.2f}")
    Sample between 94.44 and 95.44
    Sample between 362.45 and 363.45
    Sample between 46.54 and 47.54
    Sample between 31.30 and 32.30
    Sample between 146.32 and 147.32
    ...

.. figure:: static/sampler_1s.gif

    Windows of length 1 s drawn by the random fixed-window sampler.
    Each red block is one sample placed over the gray sampling
    intervals.

Note that the order of the samples is shuffled, and that temporal jitter
is used, so that the windows are not sampled at the same time from epoch
to epoch.

We can also easily change the window length of the sampler to get
different sized windows. This flexibility is achieved without having to
reprocess the underlying data.

.. code-block:: pycon

    >>> torch_rng = torch.Generator()
    >>> torch_rng.manual_seed(123)  # seeded for reproducibility

    >>> sampler = RandomFixedWindowSampler(
    ...     sampling_intervals=dataset.get_sampling_intervals("train"),
    ...     window_length=10.0,
    ...     generator=torch_rng,
    ... )
    >>> print("Number of sampled windows in one epoch: ", len(sampler))
    WARNING:root:Skipping 113.86756666666673 seconds of data due to short intervals. Remaining: 270.0 seconds.
    Number of sampled windows in one epoch:  27

    >>> for i, sample_index in enumerate(sampler):
    ...     print(f"Sample between {sample_index.start:.2f} and {sample_index.end:.2f}")
    Sample between 298.95 and 308.95
    Sample between 485.97 and 495.97
    Sample between 114.44 and 124.44
    Sample between 141.66 and 151.66
    Sample between 171.66 and 181.66
    ...

.. figure:: static/sampler_10s.gif

    Same sampler with a window length of 10 seconds. Larger windows
    mean fewer samples per epoch, and short intervals get skipped
    because no 10 s window fits inside them.

While we're at it, let's also look at the other two samplers:

SequentialFixedWindowSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: pycon

    >>> from torch_brain.samplers import SequentialFixedWindowSampler

    >>> sampler = SequentialFixedWindowSampler(
    ...     sampling_intervals=dataset.get_sampling_intervals("train"),
    ...     window_length=10.0,
    ...     drop_short=True,
    ... )
    >>> print("Number of sampled windows in one epoch: ", len(sampler))
    WARNING:root:Skipping 113.86756666666673 seconds of data due to short intervals. Remaining: 420.0 seconds.
    Number of sampled windows in one epoch:  42

    >>> for i, sample_index in enumerate(sampler):
    ...     print(f"Sample between {sample_index.start:.2f} and {sample_index.end:.2f}")
    Sample between 0.00 and 10.00
    Sample between 10.00 and 20.00
    Sample between 20.00 and 30.00
    Sample between 28.51 and 38.51
    Sample between 109.40 and 119.40
    ...

.. figure:: static/sampler_sequential_10s.gif

    Sequential fixed-window sampler with a window length of 10
    seconds. Windows tile each interval from left to right without
    overlap, and in deterministic order.

TrialSampler
~~~~~~~~~~~~

A :obj:`~torch_brain.samplers.TrialSampler` treats each interval start-end pair as
a *"trial"*, and samples them as whole.

.. code-block:: pycon

    >>> from torch_brain.samplers import TrialSampler

    >>> torch_rng = torch.Generator()
    >>> torch_rng.manual_seed(123)  # seeded for reproducibility

    >>> sampler = TrialSampler(
    ...     sampling_intervals=dataset.get_sampling_intervals("train"),
    ...     shuffle=True,
    ... )
    >>> print("Number of sampled windows in one epoch: ", len(sampler))
    Number of sampled windows in one epoch:  38

    >>> for i, sample_index in enumerate(sampler):
    ...     print(f"Sample between {sample_index.start:.2f} and {sample_index.end:.2f}")
    Sample between 617.42 and 633.21
    Sample between 237.87 and 239.33
    Sample between 597.53 and 598.73
    Sample between 129.91 and 130.70
    Sample between 44.02 and 49.32
    ...

.. figure:: static/sampler_trial_10s.gif

    The trial sampler returns each interval as-is, so samples are
    *variable-length* and follow the trial structure of the data.

Notice that the samples are *variable-length* in this case!


From a sample to a data slice
-----------------------------

All three samplers above emit the same kind of object: a
:obj:`~torch_brain.datasets.DatasetIndex`. It is a small dataclass that records *where* a
sample comes from — which recording, and which time window:

.. code-block:: python

    from dataclasses import dataclass

    @dataclass
    class DatasetIndex:
        recording_id: str
        start: float
        end: float

These three fields are everything a :obj:`~torch_brain.datasets.Dataset` needs to load a
sample. Indexing the dataset with a :obj:`~torch_brain.datasets.DatasetIndex` returns a
:obj:`~torch_brain.data.Data` object containing just that time slice of
the underlying recording:

.. code-block:: pycon

    >>> sample_index = next(iter(sampler))
    >>> sample_index
    DatasetIndex(recording_id='c_20131003_center_out_reaching', start=402.986, end=406.388)

    >>> data = dataset[sample_index]
    >>> data
    Data(
      brainset=Data(derived_version: ..., description: ...),
      cursor=LazyIrregularTimeSeries(timestamps: ..., acc: ..., pos: ..., vel: ...),
      ...
      spikes=LazyIrregularTimeSeries(timestamps: ..., unit_index: ...),
      trials=LazyInterval(...),
      ...
    )

    >>> data.start, data.end, data.absolute_start
    0.0, 3.402, 402.986

This is the bridge between sampling and data loading: the sampler
emits a stream of :obj:`~torch_brain.datasets.DatasetIndex` objects, and the
:obj:`~torch_brain.datasets.Dataset` turns each one into a sliced
:obj:`~torch_brain.data.Data` sample.


Sampling from multiple recordings
---------------------------------

The sampler seamlessly works with datasets containing multiple
recordings.

.. code-block:: console

    $ brainsets prepare perich_miller_population_2018 -s c_20131022_center_out_reaching
    $ brainsets prepare perich_miller_population_2018 -s c_20131023_center_out_reaching

For example, we can create a dataset with multiple recordings:

.. code-block:: pycon

    >>> dataset = PerichMillerPopulation2018(
    ...     root=None,
    ...     recording_ids=[
    ...         "c_20131003_center_out_reaching",
    ...         "c_20131022_center_out_reaching",
    ...         "c_20131023_center_out_reaching",
    ...     ],
    ... )
    >>> dataset.get_sampling_intervals("train")
    {'c_20131003_center_out_reaching': LazyInterval(
      end: lazy array, shape=(38,), dtype=float64,
      start: lazy array, shape=(38,), dtype=float64,
    ),
    'c_20131022_center_out_reaching': LazyInterval(
      end: lazy array, shape=(33,), dtype=float64,
      start: lazy array, shape=(33,), dtype=float64,
    ),
    'c_20131023_center_out_reaching': LazyInterval(
      end: lazy array, shape=(40,), dtype=float64,
      start: lazy array, shape=(40,), dtype=float64,
    )}


The same :obj:`~torch_brain.datasets.Dataset.get_sampling_intervals` method is used as before, and the
sampling intervals dictionary has three elements, corresponding to the
three recordings.

The sampler can be initialized in the same way as before.

.. code-block:: pycon

    >>> torch_rng = torch.Generator()
    >>> torch_rng.manual_seed(123) # seeded for reproducibility

    >>> sampler = RandomFixedWindowSampler(
    ...     sampling_intervals=dataset.get_sampling_intervals("train"),
    ...     window_length=1.0,
    ...     generator=torch_rng,
    ... )
    >>> print("Number of sampled windows in one epoch: ", len(sampler))
    WARNING:root:Skipping 8.784433333333283 seconds of data due to short intervals. Remaining: 1233.0 seconds.
    Number of sampled windows in one epoch:  1233

    >>> for i, sample_index in enumerate(sampler):
    ...     print(
    ...         f"Sample between {sample_index.start:.2f} and {sample_index.end:.2f} "
    ...         f"from recording {sample_index.recording_id}"
    ...     )
    Sample between 414.22 and 415.22 from recording c_20131022_center_out_reaching
    Sample between 677.32 and 678.32 from recording c_20131023_center_out_reaching
    Sample between 557.35 and 558.35 from recording c_20131022_center_out_reaching
    Sample between 565.35 and 566.35 from recording c_20131022_center_out_reaching
    Sample between 522.82 and 523.82 from recording c_20131023_center_out_reaching
    ...

We visualize this below:

.. figure:: static/sampler_multi.gif

    Sampling across three recordings simultaneously. Each row shows
    one recording's intervals (gray) and the windows drawn from them
    (red); samples interleave freely across recordings.


Further Reading
---------------

The pieces covered above — sampling intervals, samplers, and
multi-recording datasets — work with a normal PyTorch training loop.
For a worked end-to-end example that brings these together, see the
:doc:`NLB Maze minimal example </generated/notebooks/nlb_maze_minimal_example>`.
It walks through:

- defining a custom :obj:`~torch_brain.datasets.Dataset` with sampling intervals tailored
  to your use case,
- wiring a sampler into a :class:`~torch.utils.data.DataLoader`, and
- iterating batches into a model for training and evaluation.
