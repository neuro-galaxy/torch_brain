Build the Session Object
========================

In :meth:`~torch_brain.pipeline.BrainsetPipeline.process`, convert downloaded
raw data into a standardized **Session** :class:`~torch_brain.data.Data`
object.

For ``ArrayDict``, ``Interval``, and related types, see
:doc:`../../data/creating_objects`.

Add brainset metadata
---------------------

Create a :class:`~torch_brain.data.BrainsetDescription`:

.. code-block:: python

   from torch_brain.data import BrainsetDescription

   brainset_description = BrainsetDescription(
       id="my_brainset_2024",
       origin_version="1.0.0",
       derived_version="1.0.0",
       source="https://example.com/dataset",
       description="Description of your brainset...",
   )

* ``origin_version`` — version of the original data (use ``"0.0.0"`` if unversioned)
* ``derived_version`` — version of your processing logic
* ``source`` — URL or access instructions for the raw data
* ``description`` — short human-readable summary

Load raw data
-------------

.. tab:: NWB File

   .. code-block:: python

      from pynwb import NWBHDF5IO

      io = NWBHDF5IO(fpath, "r")
      nwbfile = io.read()
      # remember to close: io.close()

.. tab:: MATLAB File

   .. code-block:: python

      from scipy.io import loadmat

      mat_data = loadmat("path/to/file.mat")

.. tab:: NumPy File

   .. code-block:: python

      import numpy as np

      neural_data = np.load("path/to/spikes.npy")
      behavior = np.load("path/to/behavior.npy")

Extract subject metadata
------------------------

.. code-block:: python

   from torch_brain.data import SubjectDescription

   subject = SubjectDescription(
       id="subject_1",
       species="MACACA_MULATTA",
       sex="MALE",
   )

For NWB files from DANDI, use the helper:

.. code-block:: python

   from torch_brain.utils.dandi import extract_subject_from_nwb

   subject = extract_subject_from_nwb(nwbfile)

Extract session metadata
------------------------

.. code-block:: python

   import datetime
   from torch_brain.data import SessionDescription

   session = SessionDescription(
       id="session_1",
       recording_date=datetime.datetime(2024, 1, 1),
   )

Extract device metadata
-----------------------

.. code-block:: python

   from torch_brain.data import DeviceDescription

   device = DeviceDescription(
       id="device_1",
       recording_tech="UTAH_ARRAY_SPIKES",
   )

Extract neural data
-------------------

For spiking data, the typical outputs are ``spikes`` and ``units``.

.. tab:: NumPy

   .. code-block:: python

      import numpy as np
      from torch_brain.data import IrregularTimeSeries, ArrayDict

      spike_times = ...       # (n_spikes,)
      spike_clusters = ...    # (n_spikes,)

      spikes = IrregularTimeSeries(
          timestamps=spike_times,
          unit_index=spike_clusters,
          domain="auto",
      )

      units = ArrayDict(
          id=...,
      )

.. tab:: NWB

   .. code-block:: python

      from torch_brain.utils.dandi import extract_spikes_from_nwbfile

      spikes, units = extract_spikes_from_nwbfile(
          nwbfile,
          recording_tech="UTAH_ARRAY_SPIKES",
      )

Extract behavioral data
-----------------------

.. code-block:: python

   from torch_brain.data import IrregularTimeSeries

   cursor = IrregularTimeSeries(
       timestamps=...,
       pos=...,
       vel=...,
       acc=...,
       domain="auto",
   )

Extract trials
--------------

.. code-block:: python

   from torch_brain.data import Interval

   trials = Interval(
       start=...,
       end=...,
       go_cue=...,
       reach_direction=...,
   )

Assemble the Session
--------------------

.. code-block:: python

   from torch_brain.data import Data

   data = Data(
       brainset=brainset_description,
       subject=subject,
       session=session,
       device=device,
       spikes=spikes,
       units=units,
       trials=trials,
       cursor=cursor,
       domain="auto",
   )

Train/validation/test **Splits** and **Sampling Intervals** are typically
defined on the **Dataset** side rather than baked into Session files. See
:doc:`../../training/sampling/index`.

Notes
-----

- Validate after processing: missing values, timestamp alignment, trial bounds.
- Use :class:`~torch_brain.data.RegularTimeSeries` for fixed-rate signals and
  :class:`~torch_brain.data.IrregularTimeSeries` for event-based data.
- See ``brainsets_pipelines/`` in the repository for complete examples.

Next, see :doc:`save_to_disk` to write the ``Data`` object to H5.
