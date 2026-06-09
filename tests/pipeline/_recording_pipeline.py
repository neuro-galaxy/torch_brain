"""A stub pipeline used by runner tests.

It records every download/process call by appending to files under
``processed_dir`` so that assertions work regardless of which process
(driver or Ray worker) actually executes the work.
"""

import pandas as pd

from torch_brain.pipeline import BrainsetPipeline


class Pipeline(BrainsetPipeline):
    brainset_id = "test_brainset"
    parser = None

    @classmethod
    def get_manifest(cls, raw_dir, args):
        return pd.DataFrame(
            {"value": [10, 20, 30]},
            index=["item_a", "item_b", "item_c"],
        )

    def download(self, manifest_item):
        (self.processed_dir / "downloaded.log").open("a").write(
            f"{manifest_item.Index}\n"
        )
        return manifest_item.Index

    def process(self, download_output):
        (self.processed_dir / "processed.log").open("a").write(f"{download_output}\n")
