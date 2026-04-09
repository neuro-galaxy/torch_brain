_functions = [
    "extract_subject_from_nwb",
    "extract_spikes_from_nwbfile",
    "download_file",
    "get_nwb_asset_list",
]

__all__ = _functions


from pathlib import Path
import numpy as np
import pandas as pd
from pynwb import NWBFile

from temporaldata import ArrayDict, IrregularTimeSeries

from brainsets.descriptions import SubjectDescription
from brainsets.taxonomy import (
    RecordingTech,
    Sex,
    Species,
)

try:
    import dandi

    DANDI_AVAILABLE = True
except ImportError:
    DANDI_AVAILABLE = False


def _check_dandi_available(func_name: str) -> None:
    """Raise ImportError if DANDI is not available."""
    if not DANDI_AVAILABLE:
        raise ImportError(
            f"{func_name} requires the dandi library which is not installed. "
            "Install it with `pip install dandi`"
        )


def extract_subject_from_nwb(nwbfile: NWBFile):
    r"""Extract a :obj:`SubjectDescription <brainsets.descriptions.SubjectDescription>` from an NWBFile

    The resultant description will include ``id``, ``species``, and ``sex``

    Args:
        nwbfile: An open NWB file handle

    Returns:
        A :obj:`SubjectDescription <brainsets.descriptions.SubjectDescription>`
    """

    # DANDI has requirements for metadata included in `subject`
    # - subject_id: A subject identifier must be provided.
    # - species: either a latin binomial or NCBI taxonomic identifier.
    # - sex: must be "M", "F", "O" (other), or "U" (unknown).
    # - date_of_birth or age: this does not appear to be enforced, so will be skipped.
    species = nwbfile.subject.species

    if "NCBITaxon" in species:
        species = "NCBITaxon_" + species.split("_")[-1]

    return SubjectDescription(
        id=nwbfile.subject.subject_id.lower(),
        species=Species.from_string(species),
        sex=Sex.from_string(nwbfile.subject.sex),
    )


def extract_spikes_from_nwbfile(nwbfile: NWBFile, recording_tech: RecordingTech):
    r"""Extract spikes and unit metadata from an NWBFile

    Args:
        nwbfile: An open NWB file handle
        recording_tech: Only supports
            :obj:`RecordingTech.UTAH_ARRAY_THRESHOLD_CROSSINGS` and
            :obj:`RecordingTech.UTAH_ARRAY_SPIKES`
    """
    # spikes
    timestamps = []
    unit_index = []

    # units
    unit_meta = []

    units = nwbfile.units.spike_times_index[:]
    electrodes = nwbfile.units.electrodes.table

    # all these units are obtained using threshold crossings
    for i in range(len(units)):
        if recording_tech == RecordingTech.UTAH_ARRAY_THRESHOLD_CROSSINGS:
            # label unit
            group_name = electrodes["group_name"][i]
            unit_id = f"group_{group_name}/elec{i}/multiunit_{0}"
        elif recording_tech == RecordingTech.UTAH_ARRAY_SPIKES:
            # label unit
            electrode_id = nwbfile.units[i].electrodes.item().item()
            group_name = electrodes["group_name"][electrode_id]
            unit_id = f"group_{group_name}/elec{electrode_id}/unit_{i}"
        else:
            raise ValueError(f"Recording tech {recording_tech} not supported")

        # extract spikes
        spiketimes = units[i]
        timestamps.append(spiketimes)

        if len(spiketimes) > 0:
            unit_index.append([i] * len(spiketimes))

        # extract unit metadata
        unit_meta.append(
            {
                "id": unit_id,
                "unit_number": i,
                "count": len(spiketimes),
                "type": int(recording_tech),
            }
        )

    # convert unit metadata to a Data object
    unit_meta_df = pd.DataFrame(unit_meta)  # list of dicts to dataframe
    units = ArrayDict.from_dataframe(
        unit_meta_df,
        unsigned_to_long=True,
    )

    # concatenate spikes
    timestamps = np.concatenate(timestamps)
    unit_index = np.concatenate(unit_index)

    # create spikes object
    spikes = IrregularTimeSeries(
        timestamps=timestamps,
        unit_index=unit_index,
        domain="auto",
    )

    # make sure to sort the spikes
    spikes.sort()

    return spikes, units


def download_file(
    path: str | Path,
    url: str,
    raw_dir: str | Path,
    overwrite: bool = False,
) -> Path:
    r"""Download a file from DANDI

    Full path of the downloaded path will be ``raw_dir / path``.

    Args:
        path: path of the downloaded file within :obj:`raw_dir`
        url: URL of the DANDI asset
        raw_dir: root directory where the file will be downloaded
        overwrite: Will overwrite existing file if :obj:`True`
            (default :obj:`False`)

    """
    _check_dandi_available("download_file")
    import dandi.download

    raw_dir = Path(raw_dir)
    asset_path = Path(path)
    download_dir = raw_dir / asset_path.parent
    download_dir.mkdir(exist_ok=True, parents=True)
    dandi.download.download(
        url,
        download_dir,
        existing=(
            dandi.download.DownloadExisting.REFRESH
            if not overwrite
            else dandi.download.DownloadExisting.OVERWRITE
        ),
    )
    return raw_dir / asset_path


def get_nwb_asset_list(dandiset_id: str) -> list:
    r"""Get a list of all remote NWB assets in the given dandiset

    Args:
        dandiset_id: The dandiset ID (e.g. 'DANDI:000688/draft')

    Returns:
        A list of all remote NWB assets (``dandi.dandiapi.RemoteBlobAsset``) within this dandiset
    """
    _check_dandi_available("get_nwb_asset_list")
    from dandi import dandiarchive

    parsed_url = dandiarchive.parse_dandi_url(dandiset_id)
    with parsed_url.navigate() as (client, dandiset, assets):
        asset_list = [x for x in assets if x.path.endswith(".nwb")]
    return asset_list
