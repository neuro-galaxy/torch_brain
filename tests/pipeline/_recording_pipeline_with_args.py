"""Stub pipeline that defines a parser, to exercise the runner's
pipeline-specific argument parsing branch."""

from argparse import ArgumentParser

import pandas as pd

from torch_brain.pipeline import BrainsetPipeline

parser = ArgumentParser()
parser.add_argument("--flavor", default="vanilla")


class Pipeline(BrainsetPipeline):
    brainset_id = "test_brainset_args"
    parser = parser

    @classmethod
    def get_manifest(cls, raw_dir, args):
        return pd.DataFrame({"value": [1]}, index=["only_item"])

    def download(self, manifest_item):
        (self.processed_dir / "flavor.log").open("a").write(f"{self.args.flavor}\n")
        return manifest_item.Index

    def process(self, download_output):
        (self.processed_dir / "processed.log").open("a").write(f"{download_output}\n")
