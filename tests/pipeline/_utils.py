import sys

from torch_brain.pipeline._cli.utils import PIPELINES_PATH


def add_pipelines_to_path():
    sys.path.append(str(PIPELINES_PATH))
