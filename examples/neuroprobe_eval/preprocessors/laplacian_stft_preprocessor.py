"""
Laplacian rereferencing building blocks and Laplacian+STFT chain.
"""

import torch
import numpy as np
from .base_preprocessor import BasePreprocessor
from .stft_preprocessor import STFTPreprocessor
from . import register_preprocessor, CompositePreprocessor


def laplacian_rereference_neural_data(
    electrode_data, electrode_labels, remove_non_laplacian=False
):
    """
    Rereference the neural data using the laplacian method.
    Copied from eval_utils.py.

    Args:
        electrode_data: torch tensor of shape (batch_size, n_electrodes, n_samples) or (n_electrodes, n_samples)
        electrode_labels: list of electrode labels
        remove_non_laplacian: boolean, if True, remove the non-laplacian electrodes

    Returns:
        rereferenced_data: torch tensor
        rereferenced_labels: list of electrode labels
        original_electrode_indices: list of indices
    """

    def get_all_laplacian_electrodes(electrode_labels):
        """Get all laplacian electrodes for a given subject."""

        def stem_electrode_name(name):
            found_stem_end = False
            stem, num = [], []
            for c in reversed(name):
                if c.isalpha():
                    found_stem_end = True
                if found_stem_end:
                    stem.append(c)
                else:
                    num.append(c)
            return "".join(reversed(stem)), int("".join(reversed(num)))

        def has_neighbors(stem, stems):
            (x, y) = stem
            return ((x, y + 1) in stems) or ((x, y - 1) in stems)

        def get_neighbors(stem, stems):
            (x, y) = stem
            return [f"{x}{y}" for (x, y) in [(x, y + 1), (x, y - 1)] if (x, y) in stems]

        stems = [stem_electrode_name(e) for e in electrode_labels]
        laplacian_stems = [x for x in stems if has_neighbors(x, stems)]
        electrodes = [f"{x}{y}" for (x, y) in laplacian_stems]
        neighbors = {
            e: get_neighbors(stem_electrode_name(e), stems) for e in electrodes
        }
        return electrodes, neighbors

    # Handle both numpy arrays and torch tensors
    was_tensor = isinstance(electrode_data, torch.Tensor)

    if len(electrode_data.shape) == 2:
        if was_tensor:
            electrode_data = electrode_data.unsqueeze(0)
        else:
            electrode_data = electrode_data[np.newaxis, :, :]

    normalized_labels = [
        str(label).split("/")[-1] if isinstance(label, str) else str(label)
        for label in electrode_labels
    ]
    laplacian_electrodes, laplacian_neighbors = get_all_laplacian_electrodes(
        normalized_labels
    )
    laplacian_neighbor_indices = {
        laplacian_electrode_label: [
            normalized_labels.index(neighbor_label) for neighbor_label in neighbors
        ]
        for laplacian_electrode_label, neighbors in laplacian_neighbors.items()
    }
    laplacian_electrode_set = set(laplacian_electrodes)

    batch_size, n_electrodes, n_samples = electrode_data.shape
    rereferenced_n_electrodes = (
        len(laplacian_electrodes) if remove_non_laplacian else n_electrodes
    )
    if was_tensor:
        rereferenced_data = torch.zeros(
            (batch_size, rereferenced_n_electrodes, n_samples),
            dtype=electrode_data.dtype,
            device=electrode_data.device,
        )
    else:
        rereferenced_data = np.zeros(
            (batch_size, rereferenced_n_electrodes, n_samples),
            dtype=electrode_data.dtype,
        )

    electrode_i = 0
    original_electrode_indices = []
    rereferenced_labels = []
    for original_electrode_index, electrode_label in enumerate(electrode_labels):
        normalized_label = normalized_labels[original_electrode_index]
        if normalized_label in laplacian_electrode_set:
            neighbor_values = electrode_data[
                :, laplacian_neighbor_indices[normalized_label]
            ]
            neighbor_mean = (
                neighbor_values.mean(dim=1)
                if was_tensor
                else neighbor_values.mean(axis=1)
            )
            rereferenced_data[:, electrode_i] = electrode_data[
                :, original_electrode_index
            ] - neighbor_mean
            original_electrode_indices.append(original_electrode_index)
            rereferenced_labels.append(electrode_label)
            electrode_i += 1
        else:
            if remove_non_laplacian:
                continue  # just skip the non-laplacian electrodes
            else:
                rereferenced_data[:, electrode_i] = electrode_data[
                    :, original_electrode_index
                ]
                original_electrode_indices.append(original_electrode_index)
                rereferenced_labels.append(electrode_label)
                electrode_i += 1

    return (
        rereferenced_data,
        rereferenced_labels,
        original_electrode_indices,
    )


@register_preprocessor("laplacian_rereference")
class LaplacianRereferencePreprocessor(BasePreprocessor):
    """Preprocessor that applies Laplacian rereferencing only."""

    def _transform_one(self, sample):
        """Apply Laplacian rereferencing keyed by channel_names."""
        if not isinstance(sample, dict):
            raise TypeError(f"Expected sample dict, got {type(sample).__name__}.")
        if "x" not in sample or "channel_names" not in sample:
            raise KeyError(
                "laplacian_rereference requires sample['x'] and sample['channel_names']."
            )
        if "channel_ids" not in sample:
            raise KeyError("laplacian_rereference requires sample['channel_ids'].")

        channel_names_in = sample["channel_names"]
        if channel_names_in is None:
            raise ValueError(
                "laplacian_rereference requires sample['channel_names'] to be set."
            )
        if not isinstance(channel_names_in, list):
            raise TypeError("sample['channel_names'] must be a list[str].")
        if any(not isinstance(name, str) for name in channel_names_in):
            raise TypeError("sample['channel_names'] must contain only str values.")

        channel_ids_in = list(sample["channel_ids"])
        if len(channel_ids_in) != len(channel_names_in):
            raise ValueError(
                "sample['channel_ids'] and sample['channel_names'] length mismatch: "
                f"{len(channel_ids_in)} vs {len(channel_names_in)}."
            )

        remove_non_laplacian = self.cfg.get("remove_non_laplacian", False)
        x_out, channel_names_out, original_indices = laplacian_rereference_neural_data(
            sample["x"],
            channel_names_in,
            remove_non_laplacian=remove_non_laplacian,
        )
        if isinstance(x_out, torch.Tensor):
            x_out_np = x_out.detach().cpu().numpy()
        else:
            x_out_np = np.asarray(x_out)
        if x_out_np.ndim >= 3 and x_out_np.shape[0] == 1:
            x_out_np = x_out_np[0]

        channel_names_out = list(channel_names_out)
        aligned_old_indices = [int(i) for i in original_indices]
        if x_out_np.ndim < 2:
            raise ValueError(
                "laplacian_rereference output must be at least 2D "
                f"(channels, features...), got shape {x_out_np.shape}."
            )
        if x_out_np.shape[0] != len(channel_names_out):
            raise ValueError(
                "laplacian_rereference output channel axis does not match returned "
                f"labels: {x_out_np.shape[0]} vs {len(channel_names_out)}."
            )
        if len(aligned_old_indices) != len(channel_names_out):
            raise ValueError(
                "laplacian_rereference returned mismatched metadata indices: "
                f"{len(aligned_old_indices)} vs {len(channel_names_out)}."
            )

        out = dict(sample)
        out["x"] = np.asarray(x_out_np, dtype=np.float32)
        out["channel_names"] = channel_names_out
        out["channel_ids"] = [channel_ids_in[i] for i in aligned_old_indices]

        coords = sample.get("channel_coords_lip")
        if coords is not None:
            coords_arr = np.asarray(coords)
            if coords_arr.shape[0] != len(channel_names_in):
                raise ValueError(
                    "channel_coords_lip first dimension mismatch: "
                    f"{coords_arr.shape[0]} vs {len(channel_names_in)}."
                )
            out["channel_coords_lip"] = coords_arr[aligned_old_indices]

        seq_id = sample.get("seq_id")
        if seq_id is not None:
            seq_arr = np.asarray(seq_id)
            if seq_arr.shape[0] != len(channel_names_in):
                raise ValueError(
                    "seq_id first dimension mismatch: "
                    f"{seq_arr.shape[0]} vs {len(channel_names_in)}."
                )
            out["seq_id"] = seq_arr[aligned_old_indices]

        brain_areas = sample.get("brain_areas")
        if brain_areas is not None:
            brain_areas_arr = np.asarray(brain_areas, dtype=object).reshape(-1)
            if brain_areas_arr.shape[0] != len(channel_names_in):
                raise ValueError(
                    "brain_areas first dimension mismatch: "
                    f"{brain_areas_arr.shape[0]} vs {len(channel_names_in)}."
                )
            out["brain_areas"] = brain_areas_arr[aligned_old_indices].tolist()

        return out

    def transform_samples(self, samples):
        return [self._transform_one(sample) for sample in samples]


@register_preprocessor("laplacian_stft")
class LaplacianSTFTPreprocessor(CompositePreprocessor):
    """Backwards-compatible composite preprocessor chaining Laplacian + STFT."""

    def __init__(self, cfg):
        laplacian = LaplacianRereferencePreprocessor(cfg)
        stft = STFTPreprocessor(cfg)
        super().__init__(cfg, [laplacian, stft])
