# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "mne==1.11.0",
#   "mne-bids==0.18",
#   "boto3>=1.42.32",
#   "requests==2.32.5",
# ]
# ///

from brainsets.utils.openneuro import OpenNeuroPipeline

TYPE_CHANNELS_REMAPPING = {"EEG": [f"E{i}" for i in range(1, 129)] + ["Cz"]}


class Pipeline(OpenNeuroPipeline):
    modality = "eeg"
    brainset_id = "shirazi_hbnr1_ds005505"
    dataset_id = "ds005505"
    description = (
        "Healthy Brain Network (HBN) EEG dataset containing recordings from "
        "participants performing various passive and active tasks including "
        "resting state, movie watching, and cognitive tasks."
    )
    origin_version = "1.0.1"
    derived_version = "1.0.0"

    TYPE_CHANNELS_REMAPPING = TYPE_CHANNELS_REMAPPING
