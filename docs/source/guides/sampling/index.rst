.. _sampling:

Sampling
========

The advanced sampling capabilities in ``torch_brain`` enable flexible and customizable data
loading by allowing users to define arbitrary sampling intervals and window lengths for
their neural data. This design makes it easy to handle complex experimental protocols
with non-contiguous recording periods, while providing a simple interface that
automatically handles the complexities of sampling from multiple intervals or sessions.


Sampling intervals
------------------

*Sampling intervals* are the intervals from which a data sampler is allowed to sample data.

Datasets in ``torch_brain`` typically contain multiple recordings, and so the sampling intervals
are dictionaries keyed by the recording IDs and contain :obj:`temporaldata.Interval`
values that specify the valid start and end sampling times to the samplers.
These intervals do not have to be contiguous, and can be of any length.

The typical code-pattern for creating custom sampling intervals and using them
with a sampler is shown below:

.. code-block:: python

   from typing import Literal
   from torch_brain.dataset import Dataset
   from torch_brain.data.sampler import SequentialFixedWindowSampler

   class MyDataset(Dataset):
       ...

       def get_sampling_intervals(self, split: Literal["train", "val", "test"]):
           samp_intervals = {}
           for rid in self.recording_ids:
               recording = self.get_recording(rid)
               samp_intervals[rid] = ... # create or load an Interval that makes sense
           return samp_intervals


   dataset = MyDataset()

   sampler = SequentialFixedWindowSampler(
       sampling_intervals=dataset.get_sampling_interval(),
       window_length=1.0,
   )


Many **brainsets** provide default train/validation/test intervals which
are stored in :obj:`data.train_domain`, :obj:`data.valid_domain`, :obj:`data.test_domain`
respectively.

For example, let's load a recording from the :obj:`perich_miller_population_2018` dataset.

.. note ::

    To follow this tutorial, you can run the following brainset pipeline:

    .. code-block:: shell

        brainsets prepare perich_miller_population_2018 --raw-dir ./data/raw --processed-dir ./data/processed

.. code-block:: python

    >>> from brainsets.datasets import PerichMillerPopulation2018
    >>> dataset = PerichMillerPopulation2018(root="./data/processed")
    >>> sampling_intervals = dataset.get_sampling_intervals("train")
    >>> print(sampling_intervals)
    {'c_20131003_center_out_reaching': LazyInterval(
      end=<HDF5 dataset "end": shape (38,), type "<f8">,
      start=<HDF5 dataset "start": shape (38,), type "<f8">
    ), 'c_20131009_random_target_reaching': LazyInterval(
      end=<HDF5 dataset "end": shape (30,), type "<f8">,
      start=<HDF5 dataset "start": shape (30,), type "<f8">
    ), 'c_20131010_random_target_reaching': LazyInterval(
    ...


We note that there are a total of 38 sampling intervals for the train part of the
``'c_20131003_center_out_reaching'`` recording. We can print the first 5 sampling intervals as follows:

.. code-block:: python

    >>> for recording_id in sampling_intervals:
    >>>     for start, end in zip(sampling_intervals[recording_id].start[:5], sampling_intervals[recording_id].end[:5]):
    >>>         print(f"start: {start:.2f}, end: {end:.2f}")
    start: 0.00, end: 38.51
    start: 44.02, end: 49.32
    start: 55.78, end: 60.20
    start: 65.15, end: 71.30
    start: 77.15, end: 83.56


The intervals are of different lengths. We visualize the intervals below.

.. bokeh-plot:: guides/sampling/plot_sampler_1.py
   :source-position: none

We can visualize the validation and testing intervals as well.

.. code-block:: python

    >>> train_dataset = Dataset(
    >>>     "./processed",
    >>>     recording_id="perich_miller_population_2018/c_20131003_center_out_reaching",
    >>>     split="train"
    >>> )
    >>> train_sampling_intervals = train_dataset.get_sampling_intervals()

    >>> valid_dataset = Dataset(
    >>>     "./processed",
    >>>     recording_id="perich_miller_population_2018/c_20131003_center_out_reaching",
    >>>     split="valid"
    >>> )
    >>> valid_sampling_intervals = valid_dataset.get_sampling_intervals()


    >>> test_dataset = Dataset(
    >>>     "./processed",
    >>>     recording_id="perich_miller_population_2018/c_20131003_center_out_reaching",
    >>>     split="test"
    >>> )
    >>> test_sampling_intervals = test_dataset.get_sampling_intervals()


.. bokeh-plot:: guides/sampling/plot_sampler_2.py
   :source-position: none


Samplers in action
------------------

**torch_brain** provides a number of samplers that can be used to generate samples for training, or evaluation.


.. currentmodule:: torch_brain.data.sampler

.. list-table::
   :widths: 25 125

   * - :py:class:`SequentialFixedWindowSampler`
     - A Sequential sampler, that samples fixed-length windows.
   * - :py:class:`RandomFixedWindowSampler`
     - A Random sampler, that samples fixed-length windows.
   * - :py:class:`TrialSampler`
     - A sampler that randomly samples a full contiguous interval without slicing it into windows.


The most common sampler used in practice is the :py:class:`RandomFixedWindowSampler`, which randomly samples
windows of a fixed length from the data. We provide the sampling intervals in order to
restrict where the sampler can sample from.

.. code-block:: python

    >>> from torch_brain.data.sampler import RandomFixedWindowSampler

    >>> sampler = RandomFixedWindowSampler(
    >>>     sampling_intervals=dataset.get_sampling_intervals(),
    >>>     window_length=1.0,
    >>>     generator=None,
    >>> )

    >>> print("Number of sampled windows in one epoch: ", len(sampler))
    WARNING:root:Skipping 0.481 seconds of data due to short intervals. Remaining: 349.0 seconds.
    Number of sampled windows in one epoch:  349


This sampler will generate exactly 349 samples, and that a small isolated interval of length 0.481 seconds is skipped because it is too short to sample 1s windows from.
This is the default behavior of the sampler, which will raise a warning if any intervals are skipped. To raise an error instead, set the ``drop_short`` parameter to ``False``.

We can visualize what the sampler is doing as we are iterating over it.

.. code-block:: python

    >>> for sample_index in sampler:
    >>>     print(f"Sample between {sample_index.start:.2f} and {sample_index.end:.2f} from recording {sample_index.recording_id}")
    Sample between 500.96s and 501.96s from recording perich_miller_population_2018/c_20131003_center_out_reaching
    Sample between 617.12s and 618.12s from recording perich_miller_population_2018/c_20131003_center_out_reaching
    Sample between 326.50s and 327.50s from recording perich_miller_population_2018/c_20131003_center_out_reaching
    ...



.. bokeh-plot:: guides/sampling/plot_sampler_3.py
   :source-position: none


Note that the order of the samples is shuffled, and that temporal jitter is used, so that
the windows are not sampled at the same time from epoch to epoch.

We can also easily change the window length of the sampler to get different sized windows.
This flexibility is achieved without having to reprocess the underlying data.


.. code-block:: python

    >>> sampler = RandomFixedWindowSampler(
    >>>     sampling_intervals=dataset.get_sampling_intervals(),
    >>>     window_length=10.0,
    >>>     generator=None,
    >>> )

    >>> print("Number of sampled windows in one epoch: ", len(sampler))
    WARNING:root:Skipping 12.062000000000026 seconds of data due to short intervals. Remaining: 300.0 seconds.
    Number of sampled windows in one epoch:  60


.. bokeh-plot:: guides/sampling/plot_sampler_4.py
   :source-position: none


Sampling from multiple recordings
---------------------------------

The sampler seamlessly works with datasets containing multiple recordings.

For example, we can create a dataset with multiple recordings using a configuration file:

.. code-block:: yaml
    :caption: config.yaml

    - selection:
      - brainset: perich_miller_population_2018
        sessions:
          - c_20131003_center_out_reaching
          - c_20131022_center_out_reaching
          - c_20131023_center_out_reaching


.. code-block:: python

    >>> dataset = Dataset("./processed", config="config.yaml", split="train")
    >>> print(dataset.get_sampling_intervals())
    {'perich_miller_population_2018/c_20131003_center_out_reaching': LazyInterval(
    end=<HDF5 dataset "end": shape (23,), type "<f8">,
    start=<HDF5 dataset "start": shape (23,), type "<f8">
    ), 'perich_miller_population_2018/c_20131022_center_out_reaching': LazyInterval(
    end=<HDF5 dataset "end": shape (22,), type "<f8">,
    start=<HDF5 dataset "start": shape (22,), type "<f8">
    ), 'perich_miller_population_2018/c_20131023_center_out_reaching': LazyInterval(
    end=<HDF5 dataset "end": shape (31,), type "<f8">,
    start=<HDF5 dataset "start": shape (31,), type "<f8">
    )}

The same `get_sampling_intervals` method is used as before, and the sampling intervals
dictionary has three elements, corresponding to the three recordings.

The sampler can be initialized in the same way as before.

.. code-block:: python

    >>> sampler = RandomFixedWindowSampler(
    >>>     sampling_intervals=dataset.get_sampling_intervals(),
    >>>     window_length=1.0,
    >>>     generator=None,
    >>> )

    >>> print("Number of sampled windows in one epoch: ", len(sampler))
    WARNING:root:Skipping 3.225999999999999 seconds of data due to short intervals. Remaining: 959.0 seconds.
    Number of sampled windows in one epoch:  959

    >>> for sample_index in sampler:
    >>>     print(f"Sample between {sample_index.start:.2f} and {sample_index.end:.2f} from recording {sample_index.recording_id}")
    Sample between 487.99 and 488.99 from recording perich_miller_population_2018/c_20131003_center_out_reaching
    Sample between 445.04 and 446.04 from recording perich_miller_population_2018/c_20131023_center_out_reaching
    Sample between 617.21 and 618.21 from recording perich_miller_population_2018/c_20131003_center_out_reaching
    Sample between 470.31 and 471.31 from recording perich_miller_population_2018/c_20131023_center_out_reaching
    Sample between 333.30 and 334.30 from recording perich_miller_population_2018/c_20131003_center_out_reaching
    ...

Below, we visualize how the sampler will sample from all three recordings.

.. bokeh-plot:: guides/sampling/plot_sampler_5.py
   :source-position: none


Sampling intervals modifier
---------------------------

For certain models, you may want to use only a subset of the data. For this, we make it easy
to modify the sampling intervals through the configuration file.

This is done by adding a ``sampling_intervals_modifier`` key to the dataset configuration file.

.. code-block:: yaml
    :caption: config.yaml

    - selection:
        - brainset: [YOUR_BRAINSET]
      config:
        sampling_intervals_modifier: |
          [YOUR_PYTHON_CODE_GOES_HERE]
           # sampling_intervals = ...


The sampling_intervals_modifier allows you to modify the sampling intervals for each
recording by executing custom Python code. You have access the following variables:

- ``data``: The Data object for the current recording
- ``sampling_intervals``: The current sampling intervals for the recording
- ``split``: The current split (e.g. "train", "val", "test")

The modifier code should update the ``sampling_intervals`` variable with the modified intervals.

**Example 1**: Modify the sampling intervals to only include times during reaching periods.

.. code-block:: yaml
    :caption: config.yaml

    - selection:
      - brainset: perich_miller_population_2018
        sessions:
          - c_20131003_center_out_reaching
          - c_20131022_center_out_reaching
          - c_20131023_center_out_reaching
      config:
        sampling_intervals_modifier: |
          sampling_intervals = sampling_intervals & data.movement_phases.reach_period

.. note::

    The ``&`` operator performs an intersection between intervals. :obj:`temporaldata` allows for
    powerful interval operations, such as union, intersection, difference, and more. Refer to
    the  :obj:`temporaldata` documentation for more information.

**Example 2**: Modify the sampling intervals to only include the first 10 intervals for the training split.

.. code-block:: yaml
    :caption: config.yaml

    - selection:
      - brainset: perich_miller_population_2018
        sessions:
          - c_20131003_center_out_reaching
          - c_20131022_center_out_reaching
          - c_20131023_center_out_reaching
      config:
        sampling_intervals_modifier: |
            import numpy as np
            sampling_intervals = sampling_intervals & data.movement_phases.reach_period
            if split == "train":
                mask = np.zeros(len(sampling_intervals), dtype=bool)
                mask[:10] = True
                sampling_intervals = sampling_intervals.select_by_mask(mask)
