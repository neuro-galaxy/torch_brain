"""FALCON H1 (Karpowicz et al., NeurIPS 2024 Datasets & Benchmarks): a human participant
with tetraplegia (Pitt BCI) attempting 7-DoF reach and grasp in open loop; 176
threshold-crossing channels from motor cortex, 7-D kinematic velocity as decoding target.
Session dates in the files are shifted for de-identification and are kept as given.
"""

from argparse import ArgumentParser

python_version = "3.11"
dependencies = ["dandi==0.82.0"]

parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")
