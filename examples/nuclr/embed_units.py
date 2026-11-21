import collections
import typing
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from urllib.request import urlretrieve

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.nodes import AnyNode
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

from torch_brain.data import Data
from torch_brain.models import NuCLR

BWB_URL = (
    "https://brain-wide-bench.s3.amazonaws.com/brainsets/selected_units/"
    "ibl_brain_wide_bench_2026/eval/{}.h5"
)

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--recording-id", default="19e66dc9-bf9f-430b-9d6a-acfa85de6fb7")
parser.add_argument("--seed", default=42, type=int, choices=range(42, 47))
parser.add_argument("--data-root", default="data")
parser.add_argument(
    "--max-minutes",
    default=None,
    type=float,
    help="Only embed the start of the recording. Useful on CPU.",
)
parser.add_argument("--output", default="nuclr_umap.png")
args = parser.parse_args()


def load_pretrained(seed: int) -> NuCLR:
    path = hf_hub_download("nerdslab/ibl-bwb-nuclr", f"nuclr_{seed}.pt")
    # The checkpoint stores its training config as an omegaconf object.
    safe_globals = [DictConfig, ListConfig, ContainerMetadata, Metadata, AnyNode]
    safe_globals += [typing.Any, typing.Dict, collections.defaultdict]  # noqa: UP006
    safe_globals += [dict, list, int, str]
    with torch.serialization.safe_globals(safe_globals):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)

    cfg = OmegaConf.to_container(ckpt["cfg"]["model"])
    cfg.pop("_target_")
    model = NuCLR(**cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def load_recording(recording_id: str, root: str) -> Data:
    path = Path(root) / f"{recording_id}.h5"
    if not path.exists():
        print(f"Downloading {recording_id}...")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".part")
        urlretrieve(BWB_URL.format(recording_id), tmp_path)
        tmp_path.replace(path)
    with h5py.File(path) as f:
        return Data.from_hdf5(f, lazy=False)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_pretrained(args.seed).to(device).eval()

data = load_recording(args.recording_id, args.data_root)
if args.max_minutes is not None:
    start = data.domain.start[0]
    data = data.slice(start, start + 60 * args.max_minutes)

with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
    emb = model.embed(data).float().cpu().numpy()

# Units are never told where they are: compare their embeddings to brain regions.
region = data.units.region_beryl.astype(str)
keep = ~np.isnan(emb).any(1) & ~np.isin(region, ["root", "void"])
emb, region = normalize(emb[keep]), region[keep]

acc = cross_val_score(KNeighborsClassifier(5), emb, region, cv=5).mean()
majority = np.unique(region, return_counts=True)[1].max() / len(region)
print(f"5-NN brain region accuracy: {acc:.2f} (majority class: {majority:.2f})")

xy = umap.UMAP(random_state=0).fit_transform(emb)
fig, ax = plt.subplots(figsize=(6, 5))
for r in np.unique(region):
    ax.scatter(*xy[region == r].T, s=10, label=f"{r} ({(region == r).sum()})")
ax.set(xticks=[], yticks=[], title=f"NuCLR unit embeddings ({len(region)} units)")
ax.legend(title="Brain region", frameon=False)
fig.tight_layout()
fig.savefig(args.output, dpi=150)
print(f"Saved UMAP to {args.output}")
