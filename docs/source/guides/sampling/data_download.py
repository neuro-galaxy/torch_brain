import os
from pathlib import Path

raw_dir = "build/data/raw"
processed_dir = "build/data/processed"

rids = [
    "c_20131003_center_out_reaching",
    "c_20131022_center_out_reaching",
    "c_20131023_center_out_reaching",
]

for rid in rids:
    ret = os.system(
        f"brainsets prepare perich_miller_population_2018"
        f" --raw-dir {raw_dir} --processed-dir {processed_dir}"
        f" -s {rid}"
    )
    assert ret == 0
