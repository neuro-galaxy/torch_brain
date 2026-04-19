import os
import subprocess
from pathlib import Path

import yaml


def get_data_paths():
    root = Path(__file__).parent.parent
    sh_path = root / "detect_environment.sh"
    environment = subprocess.check_output(["bash", str(sh_path)]).strip().decode()
    yaml_path = root / "configs" / "environments" / f"{environment}.yaml"
    with open(yaml_path, "r") as f:
        data_paths = yaml.load(f, yaml.CLoader)

    for key, value in data_paths.items():
        data_paths[key] = os.path.expandvars(os.path.expanduser(value))

    return data_paths
