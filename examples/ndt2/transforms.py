from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from temporaldata import Data, IrregularTimeSeries, RegularTimeSeries
from torch_brain.data import chain
from torch_brain.nn import prepare_for_multitask_readout
from torch_brain.utils.binning import bin_spikes


# TODO rename
class FilterUnit:
    r"""Drop (or keep) units whose ids has ("unsorted" or "sorted"),
    by default drop but can keep is `keep` is set to True"""

    def __init__(self, keyword: str = "unsorted", field="spikes", keep: bool = False):
        self.keyword = keyword
        self.field = field
        self.keep = keep

    def __call__(self, data: Data) -> Data:
        # get units from data
        unit_ids = data.units.id
        num_units = len(unit_ids)

        no_keywork_unit = np.char.find(unit_ids, self.keyword) == -1
        if self.keep == True:
            keep_unit_mask = ~no_keywork_unit
        else:
            keep_unit_mask = no_keywork_unit

        if keep_unit_mask.all():
            # Nothing to drop
            return data

        keep_indices = np.where(keep_unit_mask)[0]
        data.units = data.units.select_by_mask(keep_unit_mask)

        nested_attr = self.field.split(".")
        target_obj = getattr(data, nested_attr[0])
        if isinstance(target_obj, IrregularTimeSeries):
            # make a mask to select spikes that are from the units we want to keep
            spike_mask = np.isin(target_obj.unit_index, keep_indices)

            # using lazy masking, we will apply the mask for all attributes from spikes
            # and units.
            setattr(data, self.field, target_obj.select_by_mask(spike_mask))

            relabel_map = np.zeros(num_units, dtype=int)
            relabel_map[keep_unit_mask] = np.arange(keep_unit_mask.sum())

            target_obj = getattr(data, self.field)
            target_obj.unit_index = relabel_map[target_obj.unit_index]
        elif isinstance(target_obj, RegularTimeSeries):
            assert len(nested_attr) == 2
            setattr(
                target_obj,
                nested_attr[1],
                getattr(target_obj, nested_attr[1])[:, keep_unit_mask],
            )
        else:
            raise ValueError(f"Unsupported type for {self.field}: {type(target_obj)}")

        return data


def float_modulo_test(x, y, eps=1e-6):
    return np.abs(x - y * np.round(x / y)) < eps


class NDT2Tokenizer:
    def __init__(
        self,
        bin_time=20e-3,
        ctx_time=1.0,
        patch_size=(32, 1),
        pad_val=0,
        decoder_registry=None,
        mask_ratio=None,
        session_tokenizer=None,
        subject_tokenizer=None,
        inc_behavior=False,
        inc_mask=False,
    ):
        self.bin_time = bin_time
        self.ctx_time = ctx_time
        self.num_bins = int(np.round(ctx_time / bin_time))
        self.patch_size = patch_size  # (num_neurons, num_time_bins)
        assert float_modulo_test(self.ctx_time, self.bin_time)

        self.pad_val = pad_val

        self.mask_ratio = mask_ratio
        self.decoder_registry = decoder_registry

        self.session_tokenizer = session_tokenizer
        self.subjet_tokenizer = subject_tokenizer

        self.inc_behavior = inc_behavior
        self.inc_mask = inc_mask

    def bin_spikes(self, data):
        t_binned = bin_spikes(data.spikes, len(data.units.id), self.bin_time)
        return torch.tensor(t_binned, dtype=torch.int32)

    def pad_spikes(self, t_binned: torch.Tensor) -> torch.tensor:
        if t_binned.size(0) % self.patch_size[0] != 0:
            assert (t_binned != self.pad_val).all()
            extra_neurons = self.patch_size[0] - (t_binned.size(0) % self.patch_size[0])
            t_binned = F.pad(t_binned, (0, 0, 0, extra_neurons), value=self.pad_val)

        if t_binned.size(1) % self.patch_size[1] != 0:
            assert (t_binned != self.pad_val).all()
            extra_time = self.patch_size[1] - (t_binned.size(1) % self.patch_size[1])
            t_binned = F.pad(t_binned, (0, extra_time, 0, 0), value=self.pad_val)
        return t_binned

    def patchify(self, t_binned: torch.Tensor):
        num_spatial_patches = t_binned.size(0) // self.patch_size[0]
        num_temporal_patches = t_binned.size(1) // self.patch_size[1]
        spike_tokens = rearrange(
            t_binned,
            "(n pn) (t pt) -> (n t) pn pt",
            n=num_spatial_patches,
            t=num_temporal_patches,
            pn=self.patch_size[0],
            pt=self.patch_size[1],
        )

        # time and space indices for flattened patches
        time_idx = torch.arange(num_temporal_patches, dtype=torch.int32)
        time_idx = repeat(time_idx, "t -> (n t)", n=num_spatial_patches)
        space_idx = torch.arange(num_spatial_patches, dtype=torch.int32)
        space_idx = repeat(space_idx, "n -> (n t)", t=num_temporal_patches)

        return spike_tokens, time_idx, space_idx

    def __call__(self, data: Data) -> Dict:
        # -- Spikes
        # bin, pad space-dimension if necessary
        data.spikes.domain.start[0] = 0.0
        data.spikes.domain.end[0] = self.ctx_time  # hack to avoid floating point errors
        t_binned = self.bin_spikes(data)
        t_binned = self.pad_spikes(t_binned)

        # patch neurons
        spikes, time_idx, space_idx = self.patchify(t_binned)

        # -- Session token
        session_idx = self.session_tokenizer(data.session)

        # -- Subject token
        subject_idx = self.subjet_tokenizer(data.subject.id)

        spike_data = {
            "spike_tokens": spikes,
            "time_idx": time_idx,
            "space_idx": space_idx,
            "session_idx": session_idx,
            "subject_idx": subject_idx,
        }

        behavior_data = {}
        if self.inc_behavior:
            # -- Behavior
            (
                output_timestamps,
                output_task_index,
                output_values,
                output_weights,
                output_subtask_index,
            ) = prepare_for_multitask_readout(
                data,
                self.decoder_registry,
            )

            # output_values is a dict to support POYO multitask
            # we only have one task in this case
            assert len(output_values) == 1
            assert len(output_weights) == 1
            assert len(output_subtask_index) == 1

            output_values = list(output_values.values())[0]
            output_weights = list(output_weights.values())[0]
            output_subtask_index = list(output_subtask_index.values())[0]

            behavior_data = {
                "output_values": chain(output_values),
                "output_weights": chain(output_weights),
                "output_time_idx": chain(torch.arange(len(output_values))),
                "output_seqlen": chain(len(output_values)),
                "output_absolute_time": chain(output_timestamps + data._absolute_start),
                "output_subtask_idx": chain(output_subtask_index),
            }

        # Final
        batch = spike_data | behavior_data
        return batch
