import sys
from pathlib import Path


def add_pipelines_to_path():
    sys.path.append(str(Path(__file__).parents[2] / "brainsets_pipelines"))
