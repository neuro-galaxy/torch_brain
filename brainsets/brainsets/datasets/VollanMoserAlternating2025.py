from typing import Callable, Optional, Literal
from pathlib import Path

from torch_brain.dataset import Dataset, SpikingDatasetMixin


class _RecordingGroup(list):
    """A list of recording IDs that also supports attribute access for sub-groups.

    Iterating or indexing works like a normal list.  Named sub-groups are
    accessible as attributes and are themselves ``_RecordingGroup`` instances::

        >>> RECORDING_IDS.navigation          # all 42 nav sessions
        >>> RECORDING_IDS.navigation.of       # 31 open-field sessions
        >>> RECORDING_IDS.sleep               # 9 sleep sessions
        >>> list(RECORDING_IDS)               # all 51 sessions
    """

    def __init__(self, ids=None, **subgroups):
        flat = list(ids or [])
        for sub_ids in subgroups.values():
            flat.extend(sub_ids)
        super().__init__(flat)
        for name, sub in subgroups.items():
            if isinstance(sub, _RecordingGroup):
                setattr(self, name, sub)
            else:
                setattr(self, name, _RecordingGroup(sub))

    def __repr__(self):
        return f"_RecordingGroup({len(self)} recordings)"


RECORDING_IDS = _RecordingGroup(
    sleep=_RecordingGroup(
        [
            "sleep_25691_1",
            "sleep_25843_2",
            "sleep_25953_5",
            "sleep_26034_3",
            "sleep_26648_2",
            "sleep_27765_3",
            "sleep_28063_5",
            "sleep_28229_2",
            "sleep_28304_2",
        ]
    ),
    navigation=_RecordingGroup(
        of=_RecordingGroup(
            [
                "of_24365_2",
                "of_24666_1",
                "of_25127_1",
                "of_25691_1",
                "of_25691_2",
                "of_25843_1",
                "of_25843_2",
                "of_25843_5",
                "of_25953_4",
                "of_25953_5",
                "of_25954_1",
                "of_26018_2",
                "of_26034_3",
                "of_26035_1",
                "of_26648_1",
                "of_26648_2",
                "of_26820_2",
                "of_27764_1",
                "of_27765_1",
                "of_27765_2",
                "of_27765_3",
                "of_28063_1",
                "of_28063_4",
                "of_28063_5",
                "of_28229_2",
                "of_28229_3",
                "of_28258_4",
                "of_28304_1",
                "of_28304_2",
                "of_29502_1",
                "of_29502_3",
            ]
        ),
        lt=_RecordingGroup(
            [
                "lt_26648_1",
                "lt_27764_1",
                "lt_27765_2",
                "lt_28063_1",
                "lt_28229_3",
                "lt_28304_1",
                "lt_29502_3",
            ]
        ),
        mmaze=_RecordingGroup(
            [
                "mmaze_29502_1",
            ]
        ),
        ww=_RecordingGroup(
            [
                "ww_25691_1",
                "ww_25843_1",
            ]
        ),
        of_novel=_RecordingGroup(
            [
                "of_novel_25843_5",
            ]
        ),
    ),
)


class VollanMoserAlternating2025(SpikingDatasetMixin, Dataset):
    """Neuropixels recordings from MEC and hippocampus in rats during spatial navigation
    and sleep.

    Rats performed various navigation tasks (open field, linear track, M-maze, wagon
    wheel) while neural activity was recorded from medial entorhinal cortex (MEC) and/or
    hippocampus (HC) using Neuropixels probes. Sleep sessions with identified SWS and REM
    epochs are also included. The dataset contains grid cells, head direction cells, and
    other spatially tuned neurons.

    .. admonition:: Preprocessing

        To download and prepare this dataset, run
        ``brainsets prepare vollan_moser_alternating_2025``.

    **Tasks:** Open Field, Linear Track, M-Maze, Wagon Wheel, Sleep

    **Brain Regions:** MEC, Hippocampus

    **Dataset Statistics**

    - **Subjects:** 19
    - **Total Sessions:** 51 (31 Open Field, 7 Linear Track, 2 Wagon Wheel, 1 M-Maze, 1 Novel Open Field, 9 Sleep)
    - **Recording Tech:** Neuropixels

    **Navigation sessions** (42 sessions) contain:

    ``rec.spikes``
        Spike timestamps and unit indices.

    ``rec.units``
        Per-unit metadata: ``id``, ``location`` (mec/hc), ``probe_id``,
        ``shank_id``, ``shank_pos``, ``mean_rate``, ``ks2_label``, ``is_grid``.

    ``rec.samples``
        All variables on the shared 10 ms timebase (speed-filtered at 5 cm/s)
        in a single flat ``IrregularTimeSeries``:

        *Observed tracking variables:*

        - ``x`` -- head x-position relative to arena centre (m)
        - ``y`` -- head y-position relative to arena centre (m)
        - ``z`` -- head z-position relative to floor (m)
        - ``hd`` -- 2D head direction / azimuth (rad)
        - ``speed`` -- horizontal head speed (m/s)
        - ``theta`` -- instantaneous theta phase (rad)
        - ``id`` -- decoded internal direction (rad, from the LMT model)

        *LMT decoded variables* (for each population ``{pop}`` in
        ``mec``, ``hc``, ``mec_hc``):

        - ``lmt_{pop}_theta`` -- theta phase (fixed; largely duplicates ``theta``)
        - ``lmt_{pop}_hd`` -- head direction (fixed; largely duplicates ``hd``)
        - ``lmt_{pop}_id`` -- internal direction (latent)
        - ``lmt_{pop}_pos_x`` -- decoded x-position (latent)
        - ``lmt_{pop}_pos_y`` -- decoded y-position (latent)

        Populations not present for a given animal are NaN-padded so all
        sessions share the same field schema.

    ``rec.theta_chunks``
        Theta-cycle-binned timeseries (separate timebase, one sample per
        theta cycle): ``id``, ``L`` (log-likelihood, n_cycles x 30),
        ``P`` (probability, n_cycles x 30).

    ``rec.probe_channels``
        Probe channel geometry: ``probe_id``, ``channel_index``,
        ``x_um``, ``y_um``, ``shank_id``, ``connected``.

    **Sleep sessions** (9 sessions) have a fundamentally different structure
    and should be treated independently.  They contain only:

    ``rec.spikes``
        Spike timestamps and unit indices within SWS/REM epochs.

    ``rec.units``
        Minimal unit metadata (``id`` only).

    ``rec.domain``
        Union of SWS and REM epoch intervals.

    None of the navigation fields (``samples``, ``theta_chunks``,
    ``probe_channels``) are present on sleep sessions.

    **References**

    Vollan, A. Z., Gardner, R. J., Moser, M.-B. & Moser, E. I.
    *Left-right-alternating theta sweeps in the entorhinal-hippocampal spatial map.*
    Dataset: `EBRAINS <https://search.kg.ebrains.eu/instances/4080b78d-edc5-4ae4-8144-7f6de79930ea>`_.

    Args:
        root (str): Root directory for the dataset.
        recording_ids (list[str] or str, optional): Recording IDs to load.
            Defaults to all sessions.  Can be:

            - A **list** of individual recording IDs.
            - A **string shorthand**: ``"all"`` (default), ``"sleep"``,
              ``"navigation"``, ``"of"``, ``"lt"``, ``"mmaze"``, ``"ww"``,
              ``"of_novel"``.
            - A ``RECORDING_IDS`` sub-group for finer control::

                  from brainsets.datasets.VollanMoserAlternating2025 import RECORDING_IDS
                  ds = VollanMoserAlternating2025(root, recording_ids=RECORDING_IDS.navigation.of)

        transform (Callable, optional): Data transformation to apply.
        dirname (str, optional): Subdirectory for the dataset. Defaults to "vollan_moser_alternating_2025".
    """

    # Map string shorthands to RECORDING_IDS sub-groups.
    _SHORTHAND = {
        "all": None,
        "sleep": lambda: list(RECORDING_IDS.sleep),
        "navigation": lambda: list(RECORDING_IDS.navigation),
        "of": lambda: list(RECORDING_IDS.navigation.of),
        "lt": lambda: list(RECORDING_IDS.navigation.lt),
        "mmaze": lambda: list(RECORDING_IDS.navigation.mmaze),
        "ww": lambda: list(RECORDING_IDS.navigation.ww),
        "of_novel": lambda: list(RECORDING_IDS.navigation.of_novel),
    }

    def __init__(
        self,
        root: str,
        recording_ids: Optional[list[str] or str] = None,
        transform: Optional[Callable] = None,
        dirname: str = "vollan_moser_alternating_2025",
        **kwargs,
    ):
        if isinstance(recording_ids, str):
            if recording_ids not in self._SHORTHAND:
                raise ValueError(
                    f"Unknown recording_ids shorthand: {recording_ids!r}. "
                    f"Valid options: {', '.join(repr(k) for k in self._SHORTHAND)}."
                )
            resolver = self._SHORTHAND[recording_ids]
            recording_ids = resolver() if resolver is not None else None

        super().__init__(
            dataset_dir=Path(root) / dirname,
            recording_ids=recording_ids,
            transform=transform,
            namespace_attributes=["session.id", "subject.id", "units.id"],
            **kwargs,
        )

        self.spiking_dataset_mixin_uniquify_unit_ids = True

    def get_sampling_intervals(
        self,
        split: Optional[Literal["train", "valid", "test"]] = None,
    ):
        domain_key = "domain" if split is None else f"{split}_domain"
        return {
            rid: getattr(self.get_recording(rid), domain_key)
            for rid in self.recording_ids
        }
