from typing import Dict, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from temporaldata import ArrayDict, Data, IrregularTimeSeries, RegularTimeSeries
from torch_brain.nn import InfiniteVocabEmbedding
from torch_brain.utils.binning import bin_spikes

from torch_brain.data import pad, track_mask
from ibl.eval_utils import bin_behaviors


# TODO rename
class FilterUnit:
    r"""
    Drop (or keep) units whose ids has the `keyword` given in the constructor.
    By default drop but can keep if `keep` is set to True
    """

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


class Ndt2Tokenizer:
    def __init__(
        self,
        bin_time: float,
        ctx_time: float,
        patch_size: Tuple[int, int],
        pad_val: int,
        ctx_tokenizer: Dict[str, InfiniteVocabEmbedding],
        unsorted=True,
        is_ssl=True,
        bhvr_key="finger.vel",
        bhvr_dim=2,
        ibl_binning=False,
    ):
        self.bin_time: float = bin_time
        self.ctx_time: float = ctx_time
        self.num_bins: int = int(np.round(ctx_time / bin_time))
        self.patch_size: Tuple[int, int] = patch_size  # (num_neurons, num_time_bins)
        assert float_modulo_test(self.ctx_time, self.bin_time)

        self.pad_val: int = pad_val
        self.unsorted: bool = unsorted
        self.is_ssl: bool = is_ssl
        self.bhvr_key: str = bhvr_key
        self.ibl_binning: bool = ibl_binning
        self.bhvr_dim: int = bhvr_dim
        self.ctx_tokenizer = ctx_tokenizer

    def __call__(self, data: Data) -> Dict:
        # -- Spikes
        spikes = data.spikes
        nb_units = len(data.units.id)
        if self.unsorted:
            chan_nb_mapper = self.extract_chan_nb(data.units)
            spikes.unit_index = chan_nb_mapper.take(spikes.unit_index)
            # TODO do not work need to find an hack
            # nb_units = chan_nb_mapper.max() + 1
            nb_units = 96

        # TODO fix this need to call timestamsp
        spikes.timestamps

        # -- Bin spikes
        try:
            t_binned = bin_spikes(spikes, nb_units, self.bin_time)
        except:
            t_binned = np.zeros((1, self.num_bins))

        t_binned = torch.tensor(t_binned, dtype=torch.int32)

        # -- Patch neurons
        spikes, time_idx, space_idx, channel_counts = self.patchify(t_binned)
        spike_data = {
            "spike_tokens": pad(spikes),
            "time_idx": pad(time_idx),
            "space_idx": pad(space_idx),
            "spike_tokens_mask": track_mask(spikes),
            "channel_counts": pad(channel_counts),
        }

        # -- Ctx tokens
        for key, tokenizer in self.ctx_tokenizer.items():
            # TODO fix this... sesion are not accessible with .id in contrast to subject
            # spike_data[key + "_idx"] = tokenizer(getattr(data, key).id)

            if key == "session":
                spike_data["session_idx"] = tokenizer(data.session)
            else:
                spike_data[key + "_idx"] = tokenizer(getattr(data, key).id)

        behavior_data = {}
        if not self.is_ssl:
            # -- Behavior
            bhvr = getattr(data, self.bhvr_key)

            try:
                bhvr = getattr(bhvr, self.bhvr_key)
                bhvr = np.eye(self.bhvr_dim)[bhvr]
            except:
                pass

            if self.ibl_binning:
                intervals = np.c_[data.trials.start, data.trials.end]
                params = {
                    "interval_len": 2,
                    "binsize": 0.02,
                    "single_region": False,
                    "align_time": "stimOn_times",
                    "time_window": (-0.5, 1.5),
                    "fr_thresh": 0.5,
                }

                # TODO use mask_dict and refactor
                bhvr_data = getattr(data, self.bhvr_key)
                bhvr_value = bhvr_data.values

                behave_dict, mask_dict = bin_behaviors(
                    bhvr_data.timestamps,
                    bhvr_value.squeeze(),
                    intervals=intervals,
                    beh=self.bhvr_key,
                    **params,
                )
                bhvr = behave_dict[self.bhvr_key][:, None]

            behavior_data["bhvr"] = pad(bhvr)
            behavior_data["bhvr_mask"] = track_mask(bhvr)

        return spike_data | behavior_data

    def patchify(self, t_binned: torch.Tensor):
        """
        t_binned: (nb_units, time_length)
        return:
            spike_tokens: (num_temporal_patches, num_spatial_patches, neur_unit_per_patch, time_unit_per_patch)
            time_idx: (num_temporal_patches, num_spatial_patches)
            space_idx: (num_temporal_patches, num_spatial_patches)
        """

        nb_units = t_binned.shape[0]
        nb_units_per_patch = self.patch_size[0]
        if nb_units % nb_units_per_patch != 0:
            assert (t_binned != self.pad_val).all()
            extra_neurons = nb_units_per_patch - (nb_units % nb_units_per_patch)
            t_binned = F.pad(t_binned, (0, 0, 0, extra_neurons), value=0)

        time_bin = t_binned.shape[1]
        if time_bin % self.num_bins != 0:
            assert (t_binned != self.pad_val).all()
            extra_time = self.num_bins - (time_bin % self.num_bins)
            t_binned = F.pad(t_binned, (0, extra_time, 0, 0), value=0)

        num_spatial_patches = t_binned.size(0) // self.patch_size[0]
        num_temporal_patches = t_binned.size(1) // self.patch_size[1]
        # major trick to have time before space, as in o.g. NDT2(nb_units, time_length) ()
        t_binned = t_binned.T
        spike_tokens = rearrange(
            t_binned,
            "(t pt) (n pn) -> (t n) pn pt",
            n=num_spatial_patches,
            t=num_temporal_patches,
            pn=self.patch_size[0],
            pt=self.patch_size[1],
        )

        # time and space indices for flattened patches
        time_idx = torch.arange(num_temporal_patches, dtype=torch.int32)
        time_idx = repeat(time_idx, "t -> (t n)", n=num_spatial_patches)
        space_idx = torch.arange(num_spatial_patches, dtype=torch.int32)
        space_idx = repeat(space_idx, "n -> (t n)", t=num_temporal_patches)

        shape = (num_temporal_patches, num_spatial_patches)
        channel_counts = torch.full(shape, self.patch_size[0], dtype=torch.long)
        if nb_units % nb_units_per_patch != 0:
            channel_counts[:, -1] = self.patch_size[0] - extra_neurons
        channel_counts = rearrange(
            channel_counts,
            "t n -> (t n)",
            n=num_spatial_patches,
            t=num_temporal_patches,
        )

        return spike_tokens, time_idx, space_idx, channel_counts

    def extract_chan_nb(self, units: ArrayDict):
        channel_names = units.channel_name
        res = [int(chan_name.split(b" ")[-1]) for chan_name in channel_names]
        return np.array(res) - 1
