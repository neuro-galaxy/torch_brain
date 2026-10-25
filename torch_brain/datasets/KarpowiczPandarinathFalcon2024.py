from collections.abc import Callable
from pathlib import Path
from typing import Literal

from ._utils import get_processed_dir
from .dataset import Dataset
from .mixins import SpikingDatasetMixin


class KarpowiczPandarinathFalcon2024(SpikingDatasetMixin, Dataset):
    """
    FALCON: Few-shot Algorithms for COnsistent Neural decoding. Intracortical
    threshold crossings with a behavioral decoding target, recorded over many days, for
    measuring how decoders hold up on sessions recorded after training.

    .. admonition:: Preprocessing

        To download and prepare this dataset, run one of

        .. code:: shell

            brainsets prepare karpowicz_pandarinath_falcon_h1_2024
            brainsets prepare karpowicz_pandarinath_falcon_m1_2024
            brainsets prepare karpowicz_pandarinath_falcon_m2_2024

    **Tasks**

    - **H1:** human participant with tetraplegia, open-loop 7-DoF reach and grasp,
      176 channels, target ``kinematics.vel`` (7-D). Dates are shifted for
      de-identification.
    - **M1:** monkey reach-to-grasp, 64 channels, target ``emg.data`` (16 muscles).
    - **M2:** monkey two-finger-group task, 96 channels, target ``finger.vel`` (2-D).

    **Splits.** Each recording has ``falcon_split``, one of ``held_in_calib`` (the
    ``train`` split), ``held_in_minival`` (the ``valid`` split) and ``held_out_calib``
    (small calibration sets from later sessions, for few-shot adaptation; FALCON's
    evaluation data for these sessions is private). ``falcon_eval_intervals`` holds the
    time points the official evaluator scores.

    **References**

    Karpowicz, B. M., Ye, J., Fan, C., Tostado-Marcos, P., Rizzoglio, F., Washington, C.,
    et al. *Few-shot Algorithms for Consistent Neural Decoding (FALCON) Benchmark.*
    NeurIPS 2024 Datasets and Benchmarks Track.
    Dataset: `Dandisets 000954, 000941, 000953 <https://dandiarchive.org/dandiset/000954>`_.

    Args:
        task: Which FALCON dataset to load: ``"h1"``, ``"m1"`` or ``"m2"``.
        root: Root directory for the dataset. Defaults to ``processed_dir`` from brainsets config.
        recording_ids: List of recording IDs to load.
        transform: Data transformation to apply.
        dirname: Subdirectory for the dataset. Defaults to ``karpowicz_pandarinath_falcon_{task}_2024``.
    """

    def __init__(
        self,
        task: Literal["h1", "m1", "m2"],
        root: str | None = None,
        recording_ids: list[str] | None = None,
        transform: Callable | None = None,
        dirname: str | None = None,
        **kwargs,
    ):
        if task not in ("h1", "m1", "m2"):
            raise ValueError(f"task must be 'h1', 'm1' or 'm2', got {task!r}")
        if root is None:
            root = get_processed_dir()
        if dirname is None:
            dirname = f"karpowicz_pandarinath_falcon_{task}_2024"
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
        split: Literal["train", "valid", "test", "held_out_calib"] | None = None,
    ):
        r"""``held_out_calib`` returns the full domain of the held-out calibration
        recordings and nothing for the others."""
        if split == "held_out_calib":
            return {
                rid: rec.domain
                for rid in self.recording_ids
                if (rec := self.get_recording(rid)).falcon_split == "held_out_calib"
            }
        domain_key = "domain" if split is None else f"{split}_domain"
        return {
            rid: getattr(self.get_recording(rid), domain_key)
            for rid in self.recording_ids
        }
