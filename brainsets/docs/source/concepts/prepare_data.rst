Preparing a new Dataset
=======================

This tutorial walks through how to prepare a new neural dataset using **brainsets**. 


**1. Pick a `brainset_id`**

Pick a unique `brainset_id` for your dataset, you may format it as:
``{first_author_last_name}_{last_author_last_name}_{first_word_in_title}_{publication_year}``.
For example, the dataset from `Perich et al. 2018 <https://www.sciencedirect.com/science/article/pii/S0896627318308328>`_
would be ``perich_miller_population_2018``.

**2. Download the raw data**

Download the raw data from the source and place it in a directory, e.g. ``./data/raw/{brainset_id}``.

**3. Start with the prepare_data.py template**

Write a ``prepare_data.py`` script to load the raw data from a single recording, extract all relevant data and save it as a :obj:`temporaldata.Data` object.

The following is an example ``prepare_data.py`` script that can serve as a template:

.. code-block:: python
    :emphasize-lines: 15, 16, 18, 19, 21, 22, 23, 24
    
    import argparse
    import os
    import h5py

    from temporaldata import Data
    from brainsets import serialize_fn_map

    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_file", type=str)
        parser.add_argument("--output_dir", type=str, default="./processed")

        args = parser.parse_args()
        
        # load data, extract metadata and process any relevant variables
        ...
        
        # create data object
        data = Data(...)

        # split data into train, validation and test
        data.set_train_domain(...)
        data.set_valid_domain(...)
        data.set_test_domain(...)

        # save data to disk
        path = os.path.join(args.output_dir, f"{session_id}.h5")
        with h5py.File(path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)


    if __name__ == "__main__":
        main()


The highlighted sections are where you will need to add code to load the raw data, extract metadata and process any relevant variables.
The sections below describe how to fill in the highlighted sections.


**4. Add the experiment metadata**
First, create a :obj:`BrainsetDescription <brainsets.descriptions.BrainsetDescription>` object to specify dataset-level metadata:

.. code-block:: python

    from brainsets.descriptions import BrainsetDescription

    brainset_description = BrainsetDescription(
        id="my_dataset_2024",
        origin_version="1.0.0", 
        derived_version="1.0.0",
        source="https://example.com/dataset",
        description="Description of your dataset..."
    )

* ``origin_version``: The version of the original data. Used for data repositories that version their data (e.g. DANDI archive). If the data is not versioned, use ``"0.0.0"``.

* ``derived_version``: The version of the processed data. Used to track any changes made to the processed data.

* ``source``: The URL of the original data repository. If the data is not available online, add information about how to gain access (e.g. requests to original dataset authors).

* ``description``: A short description of the dataset.

**5. Load your data**

Based on the file format for you raw data, use the necessary imports and code to load the data.

.. tab:: NWB File

    .. code-block:: python

        from pynwb import NWBHDF5IO

        # Open NWB file
        io = NWBHDF5IO(args.input_file, "r")
        nwbfile = io.read()

        # Note: make sure you close the file when you are done
        # with io.close()


.. tab:: MATLAB File

    .. code-block:: python

        from scipy.io import loadmat
        
        # Load .mat file
        mat_data = loadmat("path/to/file.mat")

.. tab:: NumPy File

    .. code-block:: python

        import numpy as np
        
        # Load .npy files
        neural_data = np.load("path/to/spikes.npy")
        behavior = np.load("path/to/behavior.npy")
        metadata = np.load("path/to/metadata.npy", allow_pickle=True).item()

**6. Extract Subject metadata**

Create a :class:`~brainsets.descriptions.SubjectDescription` object to store metadata:

.. code-block:: python

    from brainsets.descriptions import SubjectDescription
    from brainsets.taxonomy import Species, Sex

    subject = SubjectDescription(
        id="subject_1",
        species=Species.MACACA_MULATTA,  # or other species from taxonomy
        sex=Sex.MALE,  # or Sex.FEMALE, Sex.OTHER, Sex.UNKNOWN
    )

This metadata can be extracted from your raw data file, or you can create it manually.

Note that if you are using NWB files from the DANDI archive, you can use the helper function ``extract_subject_from_nwb()`` to extract the subject metadata:

.. code-block:: python

    from brainsets.utils.dandi_utils import extract_subject_from_nwb
    
    subject = extract_subject_from_nwb(nwbfile)

**7. Extract Session metadata**

Create a :class:`~brainsets.descriptions.SessionDescription` object to store metadata:

.. code-block:: python

    from brainsets.descriptions import SessionDescription

    session = SessionDescription(
        id="session_1",
        recording_date=datetime.datetime(2024, 1, 1),
    )

**8. Extract Device metadata**

Create a :class:`~brainsets.descriptions.DeviceDescription` object to store metadata:

.. code-block:: python

    from brainsets.descriptions import DeviceDescription
    from brainsets.taxonomy import RecordingTech

    device = DeviceDescription(
        id="device_1",
        recording_tech=RecordingTech.UTAH_ARRAY_SPIKES
    )

**9. Extract Neural Data**

Extract and process the neural data. If you are working with spiking data, 
the expected outputs are ``spikes``and ``units``.

.. tab:: Numpy File

    .. code-block:: python

        import numpy as np
        from temporaldata import IrregularTimeSeries, ArrayDict

        spike_times = ... # np.ndarray of spike times of shape (n_spikes,)
        spike_clusters = ... # np.ndarray of cluster IDs of shape (n_spikes,)

        spikes = IrregularTimeSeries(
            timestamps=spike_times,
            unit_index=spike_clusters,
            domain="auto"
        )

        units = ... # np.ndarray of unit IDs of shape (n_units,)

        units = ArrayDict(
            id=units,
            ... # any additional metadata
        )

.. tab:: NWB File

    .. code-block:: python

        from brainsets.dandi_utils import extract_spikes_from_nwbfile
        from brainsets.taxonomy import RecordingTech
        
        spikes, units = extract_spikes_from_nwbfile(
            nwbfile, 
            recording_tech=RecordingTech.UTAH_ARRAY_SPIKES
        )

**10. Extract Behavioral Data**

Extract and process the behavioral data. If you are working with reaching task data, 
the expected outputs are ``cursor``.

.. code-block:: python
    
    from temporaldata import IrregularTimeSeries

    cursor = IrregularTimeSeries(
        timestamps=...,
        pos=...,
        vel=...,
        acc=...,
        domain="auto"
    )

**11. Extract Trial Information**

If you are working with data that is structured into trials, extract the trial information using a :class:`~temporaldata.Interval`:

.. code-block:: python

    from temporaldata import Interval

    trials = Interval(
        start=...,
        end=...,
        go_cue=...,
        reach_direction=...,
        ... # any additional trial-level attributes
    )

**12. Put it all together**

Add all the metadata and data objects to a :obj:`temporaldata.Data` object:

.. code-block:: python

    from temporaldata import Data

    data = Data(
        # metadata
        brainset=brainset_description,
        subject=subject,
        session=session,
        device=device,
        # neural activity
        spikes=spikes,
        units=units,
        # behavior
        trials=trials,
        cursor=cursor,
        # domain
        domain="auto",
    )

**13. Split the data**

Split the data into train, validation and test sets, you can do this in any way that makes sense for your dataset.

.. tab:: Split by Trials

    .. code-block:: python

        # Split trials into train/valid/test sets
        successful_trials = trials.select_by_mask(trials.is_valid)
        train_trials, valid_trials, test_trials = successful_trials.split(
            [0.7, 0.1, 0.2],  # proportions for train/valid/test 
            shuffle=True,      # randomly shuffle trials
            random_seed=42     # for reproducibility
        )

        # Set domains based on trial splits
        data.set_train_domain(train_trials)
        data.set_valid_domain(valid_trials) 
        data.set_test_domain(test_trials)

.. tab:: Split by Time

    .. code-block:: python

        # Create time intervals for train/valid/test splits
        total_time = data.domain.end - data.domain.start
        train_end = data.domain.start + 0.7 * total_time    # 70% for training
        valid_end = train_end + 0.1 * total_time            # 10% for validation
        
        train_interval = Interval(data.domain.start, train_end)
        valid_interval = Interval(train_end, valid_end)
        test_interval = Interval(valid_end, data.domain.end)

        # Set domains based on time intervals
        data.set_train_domain(train_interval)
        data.set_valid_domain(valid_interval)
        data.set_test_domain(test_interval)

Tips
----

- Always validate your data after processing by checking for:
    - Missing values
    - Proper alignment of timestamps
    - Reasonable ranges for behavioral measures
    - Valid trial segmentation

- Use appropriate data types:
    - RegularTimeSeries for fixed sampling rate data
    - IrregularTimeSeries for event-based data or data with many missing values
    - Interval for trials or other data segments

For more examples, check out the example pipelines in the brainsets repository.
