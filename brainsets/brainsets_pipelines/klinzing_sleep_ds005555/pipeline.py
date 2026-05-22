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

# The mappings below were obtained by following the schema
# described in the dataset's README file.
HEADBAND_ELECTRODE_RENAME = {
    "HB_1": "AF7",
    "HB_2": "AF8",
    "HB_IMU_1": "ACC_X",
    "HB_IMU_2": "ACC_Y",
    "HB_IMU_3": "ACC_Z",
    "HB_IMU_4": "GYRO_X",
    "HB_IMU_5": "GYRO_Y",
    "HB_IMU_6": "GYRO_Z",
    "HB_PULSE": "PULSE",
}

HEADBAND_MODALITY_CHANNELS = {
    "EEG": ["AF7", "AF8"],
    "MOTION": ["ACC_X", "ACC_Y", "ACC_Z", "GYRO_X", "GYRO_Y", "GYRO_Z"],
    "PPG": ["PULSE"],
}

PSG_ELECTRODE_RENAME = {
    "PSG_F3": "F3",
    "PSG_F4": "F4",
    "PSG_C3": "C3",
    "PSG_C4": "C4",
    "PSG_O1": "O1",
    "PSG_O2": "O2",
    "PSG_EOG": "EOG",
    "PSG_EOGL": "EOGL",
    "PSG_EOGR": "EOGR",
    "PSG_EMG": "EMG",
    "PSG_THER": "THER",
    "PSG_THOR": "THOR",
    "PSG_ABD": "ABD",
    "PSG_CAN": "CAN",
    "PSG_PULSE": "PULSE",
    "PSG_BEAT": "BEAT",
    "PSG_SPO2": "SPO2",
}

PSG_MODALITY_CHANNELS = {
    "EEG": ["F3", "F4", "C3", "C4", "O1", "O2"],
    "EOG": ["EOG", "EOGL", "EOGR"],
    "EMG": ["EMG"],
    "RESPIRATORY": ["THER", "THOR", "ABD", "CAN"],
    "PPG": ["PULSE", "BEAT", "SPO2"],
}


class Pipeline(OpenNeuroPipeline):
    modality = "eeg"
    brainset_id = "klinzing_sleep_ds005555"
    dataset_id = "ds005555"
    description = (
        "The Bitbrain Open Access Sleep (BOAS) dataset contains simultaneous recordings "
        "from a clinical PSG system (Micromed) and a wearable EEG headband (Bitbrain) "
        "across 128 nights. It includes expert-consensus sleep stage labels."
    )
    origin_version = "1.1.1"
    derived_version = "1.0.0"

    def get_channel_name_remapping(self, recording_id):
        if "acq-headband" in recording_id:
            return HEADBAND_ELECTRODE_RENAME
        return PSG_ELECTRODE_RENAME

    def get_type_channels_remapping(self, recording_id):
        if "acq-headband" in recording_id:
            return HEADBAND_MODALITY_CHANNELS
        return PSG_MODALITY_CHANNELS
