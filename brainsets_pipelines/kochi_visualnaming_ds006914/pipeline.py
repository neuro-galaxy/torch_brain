# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "mne==1.11.0",
#   "mne-bids==0.18",
#   "boto3>=1.42.32",
#   "requests==2.32.5",
# ]
# ///

from torch_brain.pipeline.openneuro import OpenNeuroPipeline


class Pipeline(OpenNeuroPipeline):
    modality = "ieeg"
    brainset_id = "kochi_visualnaming_ds006914"
    dataset_id = "ds006914"
    description = (
        "Visual Naming EC - A large-scale intracranial EEG (iEEG) dataset with 110 subjects "
        "and 353 recordings from ECoG electrodes during picture naming task. "
        "Includes comprehensive electrode localization (MNI coordinates) and channel metadata "
        "from BIDS sidecar files."
    )
    origin_version = "1.0.3"
    derived_version = "1.0.0"
