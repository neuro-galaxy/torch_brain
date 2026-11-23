"""Neural Latents Benchmark '21, maze reaching (Pei et al., 2021).

Sorted spike times and behavior from a macaque performing a delayed center-out
reaching task with obstructing barriers forming a maze, resulting in a variety of
straight and curved reaches.
Source: https://dandiarchive.org/dandiset/000140
"""

from argparse import ArgumentParser

python_version = "3.11"
dependencies = ["dandi==0.82.0"]

parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")
