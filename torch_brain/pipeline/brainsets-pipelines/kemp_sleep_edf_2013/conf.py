"""Sleep-EDF Database Expanded (Kemp et al., PhysioNet 2013).

197 whole-night polysomnographic sleep recordings with EEG, EOG, EMG, and sleep
stage annotations, from two studies: Sleep Cassette (SC) and Sleep Telemetry (ST).
Source: https://www.physionet.org/content/sleep-edfx/1.0.0/
"""

from argparse import ArgumentParser

python_version = "3.11"
dependencies = [
    "mne~=1.11.0",
    "boto3~=1.41.0",
    "scikit-learn==1.7.2",
]

parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")
parser.add_argument(
    "--study_type",
    type=str,
    choices=["sc", "st", "both"],
    default="both",
    help="Which study to download: 'sc' (Sleep Cassette), 'st' (Sleep Telemetry), or 'both'",
)
