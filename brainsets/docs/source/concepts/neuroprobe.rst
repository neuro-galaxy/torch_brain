Working with the Neuroprobe Benchmark
=====================================

`Neuroprobe <https://neuroprobe.dev>`_ is a standardized benchmark for evaluating
neural decoding models on human intracranial EEG (iEEG) data. It defines 15
tasks spanning audio, language, and vision domains, and supports both binary
and multiclass classification label modes. This benchmark is derived from the
`BrainTreebank <https://braintreebank.dev>`_ dataset — 40 hours
of sEEG recordings from 10 human subjects watching naturalistic movies.

For full details, see the `Neuroprobe paper <https://arxiv.org/abs/2509.21671>`_
and the updated `OpenReview submission <https://openreview.net/forum?id=n0WDVWqgzC>`_.

Preparing the data
------------------

Download and process the Neuroprobe data using the brainsets CLI::

    brainsets prepare neuroprobe_2025

.. note::

   Processing includes downloading raw BrainTreebank data and computing all
   benchmark splits. This may take several hours depending on your hardware and
   network connection. Use ``--cores <N>`` to parallelize.


Key concepts
------------

**Tasks.** Each task uses a 1-second window of neural data aligned to a word
onset and can be evaluated in either binary or multiclass label mode.
Available tasks:

Auditory tasks
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 55 35 55
   :class: center-first-col

   * - Task
     - Binary labels
     - Multiclass labels
   * - | ``volume`` 
     - | 0 - low 
       | 1 - high 
     - | 0 - low (<25th percentile)
       | 1 - medium (37.5th-62.5th)
       | 2 - high (>=75th percentile)
   * - | ``pitch``
     - | 0 - low 
       | 1 - high 
     - | 0 - low (<25th percentile)
       | 1 - medium (37.5th-62.5th)
       | 2 - high (>=75th percentile)
   * - | ``delta_volume`` 
     - | 0 - low 
       | 1 - high 
     - | 0 - low (<25th percentile)
       | 1 - medium (37.5th-62.5th)
       | 2 - high (>=75th percentile)

Language tasks
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 55 35 55
   :class: center-first-col

   * - Task
     - Binary labels
     - Multiclass labels
   * - ``speech`` 
     - | 1 - speech
       | 0 - nonverbal 
     - Not used (task remains binary)
   * - ``onset`` 
     - | 1 - sentence-start
       | 0 - nonverbal
     - Not used (task remains binary)
   * - | ``gpt2_surprisal``
     - | 0 - low 
       | 1 - high 
     - | 0 - low (<25th percentile)
       | 1 - medium (37.5th-62.5th)
       | 2 - high (>=75th percentile)
   * - | ``word_length`` 
     - | 0 - short 
       | 1 - long
     - | 0 - low (<25th percentile)
       | 1 - medium (37.5th-62.5th)
       | 2 - high (>=75th percentile)
   * - | ``word_gap``
     - | 0 - short
       | 1 - long
     - | 0 - short gap (<25th percentile)
       | 1 - medium gap (37.5th-62.5th)
       | 2 - long gap (>=75th percentile)
   * - | ``word_index``
     - | 0 - first word 
       | 1 - other word
     - | 0 - first word
       | 1 - second word
       | 2 - any later word
   * - | ``word_head_pos``
     - | 1 - ``bin_head == 0``
       | 0 - ``bin_head == 1``
     - Not used (task remains binary)
   * - ``word_part_speech``
     - | 1 - verb
       | 0 - non-verb
     - | 0 - noun
       | 1 - verb
       | 2 - pronoun
       | 3 - determiner
       | 4 - adjective
       | 5 - adverb

Visual tasks
^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 55 35 55
   :class: center-first-col

   * - Task
     - Binary labels
     - Multiclass labels
   * - | ``frame_brightness``
     - | 0 - low 
       | 1 - high
     - | 0 - low (<25th percentile)
       | 1 - medium (37.5th-62.5th)
       | 2 - high (>=75th percentile)
   * - | ``global_flow``
     - | 0 - low
       | 1 - high
     - | 0 - low (<25th percentile)
       | 1 - medium (37.5th-62.5th)
       | 2 - high (>=75th percentile)
   * - | ``local_flow``
     - | 0 - low
       | 1 - high
     - | 0 - low (<25th percentile)
       | 1 - medium (37.5th-62.5th)
       | 2 - high (>=75th percentile)
   * - | ``face_num``
     - | 0 - no faces
       | 1 - one or more 
     - | 0 - no faces
       | 1 - exactly one face
       | 2 - more than one face

**Label mode.** Neuroprobe supports both binary and multiclass classification.
Use ``label_mode="multiclass"`` when constructing
:class:`~brainsets.datasets.Neuroprobe2025` to load multiclass splits.

**Regimes (split types).** Neuroprobe defines three evaluation regimes that test
different levels of generalization:

- **SS-SM** (*within-session*): Train and test on data from the same subject
  watching the same movie. Uses 2-fold cross-validation on contiguous blocks to
  prevent temporal autocorrelation leakage.
- **SS-DM** (*cross-session*): Train on one movie session and test on a
  different movie from the same subject.
- **DS-DM** (*cross-subject*): Train on a fixed anchor recording (Subject 2,
  Trial 4) and test on a different subject and movie. This is the most
  challenging regime.

The default leaderboard ranking uses the **cross-session (SS-DM)** split.

**Subset tiers.** Three subset sizes control the number of subject/trial pairs
and electrodes included:

- ``"full"``: All eligible subject/trial pairs.
- ``"lite"``: A curated subset of 6 subjects with 2 trials each (12 sessions),
  capped at 120 electrodes per subject. This is the standard benchmark
  configuration.
- ``"nano"``: A single trial per subject for rapid prototyping. Only supports
  the within-session regime.

**Metric.** AUROC (Area Under the ROC Curve) remains the primary metric in
binary mode. For multiclass evaluations, follow the current Neuroprobe
evaluation protocol and leaderboard reporting guidelines.


Loading benchmark splits
------------------------

The :class:`~brainsets.datasets.Neuroprobe2025` class handles split resolution
automatically. Specify the benchmark parameters to get the correct train/test
partition:

.. code-block:: python

    from brainsets.datasets import Neuroprobe2025

    train_ds = Neuroprobe2025(
        subset_tier="lite",
        test_subject=1,
        test_session=1,
        split="train",
        label_mode="multiclass",
        task="word_part_speech",
        regime="SS-DM",
    )

    test_ds = Neuroprobe2025(
        subset_tier="lite",
        test_subject=1,
        test_session=1,
        split="test",
        label_mode="multiclass",
        task="word_part_speech",
        regime="SS-DM",
    )

The constructor resolves which recordings to load and which channel subset
and time intervals to use based on the requested split.

**Within-session (SS-SM)** uses 2-fold cross-validation. You can iterate over
folds:

.. code-block:: python

    from brainsets.datasets import Neuroprobe2025

    for fold in range(Neuroprobe2025.num_folds_for_regime("SS-SM")):
        train_ds = Neuroprobe2025(
            subset_tier="lite",
            test_subject=1,
            test_session=1,
            split="train",
            task="speech",
            regime="SS-SM",
            fold=fold,
        )
        test_ds = Neuroprobe2025(
            subset_tier="lite",
            test_subject=1,
            test_session=1,
            split="test",
            task="speech",
            regime="SS-SM",
            fold=fold,
        )


Accessing neural data and labels
---------------------------------

Each recording exposes sEEG data as a :obj:`~temporaldata.RegularTimeSeries`
sampled at 2048 Hz, along with split-specific sampling intervals and
channel inclusion masks:

.. code-block:: python

    intervals = train_ds.get_sampling_intervals()
    for recording_id, interval in intervals.items():
        rec = train_ds.get_recording(recording_id)
        print(rec.seeg_data.data.shape)
        print(interval.start[:5], interval.end[:5])
        print(interval.label[:5])

The ``interval.label`` array contains class labels for each trial window
(binary or multiclass, depending on ``label_mode``).

Channel metadata (electrode names, coordinates, inclusion masks) is available
via :meth:`~brainsets.datasets.Neuroprobe2025.get_channel_metadata`:

.. code-block:: python

    meta = train_ds.get_channel_metadata(recording_id)
    print(meta["names"])
    print(meta["coords"])           # LIP coordinates
    print(meta["included_mask"])    # benchmark electrode subset


Loading raw recordings
----------------------

If you want access to full continuous recordings without benchmark splits
(e.g. for pre-training), pass explicit ``recording_ids``:

.. code-block:: python

    from brainsets.datasets import Neuroprobe2025

    ds = Neuroprobe2025(recording_ids=["sub_1_trial001", "sub_2_trial004"])

In this mode, no split/task/regime resolution is performed; you get the
complete neural data for the requested sessions.


Running a complete benchmark evaluation
---------------------------------------

A typical benchmark loop iterates over all tasks, regimes, and subject/trial
pairs. Here is a minimal skeleton:

.. code-block:: python

    from brainsets.datasets import Neuroprobe2025
    from brainsets.datasets.Neuroprobe2025 import (
        VALID_TASKS,
        NEUROPROBE_LITE_SUBJECT_TRIALS,
    )

    regime = "SS-DM"
    results = {}

    for task in VALID_TASKS:
        for subject, session in sorted(NEUROPROBE_LITE_SUBJECT_TRIALS):
            for fold in range(Neuroprobe2025.num_folds_for_regime(regime)):
                train_ds = Neuroprobe2025(
                    subset_tier="lite",
                    test_subject=subject,
                    test_session=session,
                    split="train",
                    task=task,
                    regime=regime,
                    fold=fold,
                )
                test_ds = Neuroprobe2025(
                    subset_tier="lite",
                    test_subject=subject,
                    test_session=session,
                    split="test",
                    task=task,
                    regime=regime,
                    fold=fold,
                )

                # Train your model on train_ds, evaluate on test_ds
                # auroc = evaluate(model, test_ds)
                # results[(task, subject, session, fold)] = auroc

Report the mean AUROC across all subject/session pairs for each task, along
with the overall mean. Submit results to the
`Neuroprobe leaderboard <https://neuroprobe.dev>`_ following the instructions
in the `Neuroprobe code repository <https://github.com/azaho/neuroprobe>`_.


References
----------

.. code-block:: bibtex

    @article{zahorodnii2025neuroprobe,
        title={Neuroprobe: Evaluating Intracranial Brain Responses to Naturalistic Stimuli},
        author={Zahorodnii, Andrii and Wang, Christopher and Stankovits, Bennett
                and Moraitaki, Charikleia and Chau, Geeling and Barbu, Andrei
                and Katz, Boris and Fiete, Ila R},
        journal={arXiv preprint arXiv:2509.21671},
        year={2025}
    }
