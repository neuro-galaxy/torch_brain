# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "mne==1.11.0",
#   "mne-bids==0.18",
#   "boto3>=1.42.32",
#   "requests==2.32.5",
# ]
# ///

from argparse import ArgumentParser
from pathlib import Path

import h5py
import pandas as pd

from torch_brain.data import serialize_fn_map
from torch_brain.pipeline.openneuro import (
    OpenNeuroPipeline,
    base_openneuro_parser,
    fetch_participants_tsv,
)

RELEASES = {
    1: "ds005505",
    2: "ds005506",
    3: "ds005507",
    4: "ds005508",
    5: "ds005509",
    6: "ds005510",
    7: "ds005511",
    8: "ds005512",
    9: "ds005514",
    10: "ds005515",
    11: "ds005516",
}

# Channels remapping are done according to the original dataset channel
# description files associated with each recording (*_channels.tsv)
TYPE_CHANNELS_REMAPPING = {"EEG": [f"E{i}" for i in range(1, 129)] + ["Cz"]}

parser = ArgumentParser(parents=[base_openneuro_parser], add_help=False)
parser.add_argument(
    "--release",
    type=int,
    default=None,
    help="Prepare only this release (1-11). Omit to prepare all releases.",
)


class Pipeline(OpenNeuroPipeline):
    modality = "eeg"
    brainset_id = "shirazi_hbn"
    dataset_id = "ds005505"
    description = (
        "Healthy Brain Network (HBN) EEG dataset combining all 11 data releases. "
        "Contains recordings from participants performing various passive "
        "and active tasks including resting state, movie watching, "
        "and cognitive tasks."
    )
    origin_version = "1.0.1"
    derived_version = "1.0.0"
    TYPE_CHANNELS_REMAPPING = TYPE_CHANNELS_REMAPPING
    parser = parser

    @classmethod
    def get_manifest(cls, raw_dir, args):
        if args and args.release is not None:
            if args.release not in RELEASES:
                raise ValueError(
                    f"Invalid choice: release must be one of 1-11, got {args.release}."
                )
            releases = {args.release: RELEASES[args.release]}
        else:
            releases = RELEASES

        all_manifests = []
        original_dataset_id = cls.dataset_id
        try:
            for release_id, dataset_id in releases.items():
                cls.dataset_id = dataset_id
                manifest = super().get_manifest(raw_dir, args)
                manifest["release_id"] = release_id
                manifest["release_dataset_id"] = dataset_id
                all_manifests.append(manifest)
        finally:
            cls.dataset_id = original_dataset_id
        return pd.concat(all_manifests)

    def _release_raw_dir(self, release_id) -> Path:
        # Capture the runner-assigned raw dir once, before we start repointing it,
        # then nest per-release subdirs inside it: <raw>/shirazi_hbn/r{n}.
        if not hasattr(self, "_base_raw_dir"):
            self._base_raw_dir = self.raw_dir
        return self._base_raw_dir / f"R{int(release_id)}"

    def download(self, manifest_item):
        self.dataset_id = manifest_item.release_dataset_id
        self.raw_dir = self._release_raw_dir(manifest_item.release_id)
        return super().download(manifest_item)

    def process(self, download_output: pd.Series) -> None:

        self.dataset_id = download_output.release_dataset_id
        self.raw_dir = self._release_raw_dir(download_output.release_id)
        release_id = int(download_output.release_id)
        recording_id = download_output.Index

        # All releases share a single processed dir; the release is encoded in the
        # output filename prefix (HBN_R{n}_) and in the Data object metadata.
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        store_path = self.processed_dir / f"HBN_R{release_id}_{recording_id}.h5"
        if store_path.exists() and not getattr(self.args, "reprocess", False):
            self.update_status("Already Processed")
            return

        result = super().process_common(download_output)

        if result is None:
            return

        data, _ = result

        participants_data = fetch_participants_tsv(self.dataset_id)

        if participants_data is not None:
            row = participants_data.loc[data.subject.id]
            if row is not None:
                data.subject.species = "HOMO_SAPIENS"
                data.subject.ehq_total = row.get("ehq_total", None)
                data.subject.commercial_use = row.get("commercial_use", None)
                data.subject.full_pheno = row.get("full_pheno", None)
                data.subject.p_factor = row.get("p_factor", None)
                data.subject.attention = row.get("attention", None)
                data.subject.internalizing = row.get("internalizing", None)
                data.subject.externalizing = row.get("externalizing", None)

        data.release_id = release_id
        data.release_dataset_id = str(download_output.release_dataset_id)

        self.update_status("Storing")
        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)
