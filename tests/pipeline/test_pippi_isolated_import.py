import subprocess
import sys
from pathlib import Path

PIPELINE = (
    Path(__file__).resolve().parents[2]
    / "torch_brain"
    / "pipeline"
    / "brainsets-pipelines"
    / "berezutskaya_pippi_2022"
    / "pipeline.py"
)


def test_pippi_pipeline_import_does_not_require_torch_or_datasets_package():
    code = """
import importlib.abc
import importlib.util
import sys

class BlockHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("blocked torch import")
        if fullname == "torch_brain.datasets" or fullname.startswith("torch_brain.datasets."):
            raise ModuleNotFoundError("blocked datasets import")
        return None

sys.meta_path.insert(0, BlockHeavyImports())
spec = importlib.util.spec_from_file_location("pippi_isolated_pipeline", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.PIPPI_SUBSET_TIERS == ("full", "high-cov", "low-cov")
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(PIPELINE)],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
