import os
from pathlib import Path

raw_dir = "build/data/raw"
processed_dir = "build/data/processed"

rids = [
    "c_20131003_center_out_reaching",
    "c_20131022_center_out_reaching",
    "c_20131023_center_out_reaching",
]

brainset = "perich_miller_population_2018"

for rid in rids:
    if not (Path(processed_dir) / brainset / f"{rid}.h5").exists():
        ret = os.system(
            f"brainsets prepare {brainset}"
            f" --raw-dir {raw_dir} --processed-dir {processed_dir}"
            f" -s {rid}"
        )
        assert ret == 0
