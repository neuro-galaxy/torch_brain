from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import torch
import torch.nn as nn
from torchtyping import TensorType
from temporaldata import Data

from torch_brain.data import chain, pad2d, pad8, track_mask2d, track_mask8
from torch_brain.nn import (
    Embedding,
    FeedForward,
    InfiniteVocabEmbedding,
    MultitaskReadout,
    RotaryCrossAttention,
    RotarySelfAttention,
    RotaryTimeEmbedding,
    prepare_for_multitask_readout,
)
from torch_brain.registry import ModalitySpec, MODALITY_REGISTRY

from torch_brain.utils import (
    create_linspace_latent_tokens,
)

logger = logging.getLogger(__name__)


class GRU(nn.Module):
    """Thin wrapper around nn.GRU that discards hidden state."""

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        bias=True,
        batch_first=True,
        dropout=0,
    ):
        super().__init__()
        self.net = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
        )

    def forward(self, x):
        out, _ = self.net(x)
        return out


class POSSM(nn.Module):
    """POSSM (POYO + State Space Model) combines the POYO perceiver-based encoder
    with a sequential GRU backbone for processing neural spike data in temporal bins.

    The model processes neural spike sequences through the following steps:

    1. Input spikes are binned into temporal intervals.
    2. For each bin, a perceiver encoder compresses the spikes into latent tokens via
       cross-attention, optionally refined by self-attention processing layers.
    3. The sequence of per-bin latent representations is passed through a GRU backbone.
    4. Output queries attend to the GRU outputs via a decoder cross-attention with a
       causal context mask.
    5. Task-specific linear layers produce the final predictions.

    Args:
        sequence_length: Maximum duration of the input spike sequence (in seconds).
        bin_width: Width of each temporal bin (in seconds).
        bin_step: Step between consecutive bins (in seconds).
        num_latents: Number of latent tokens per bin. The latent step is
            auto-calculated as bin_width / num_latents.
        readout_specs: Specifications for each prediction task.
        dim: Dimension of all embeddings.
        depth: Number of processing layers.
        dim_head: Dimension of each attention head.
        cross_heads: Number of attention heads used in cross-attention.
        self_heads: Number of attention heads used in self-attention.
        ffn_dropout: Dropout rate for feed-forward networks.
        lin_dropout: Dropout rate for linear layers.
        atn_dropout: Dropout rate for attention.
        rnn_dim: Hidden size of the GRU backbone.
        num_rnn_layers: Number of GRU layers.
        rnn_dropout: Dropout rate for GRU.
        output_ca_ctx_lim: Number of past bins visible in decoder cross-attention.
        emb_init_scale: Scale for embedding initialization.
        t_min: Minimum timestamp resolution for rotary embeddings.
        t_max: Maximum timestamp resolution for rotary embeddings.
    """

    def __init__(
        self,
        *,
        sequence_length: float,
        bin_width: float = 0.05,
        bin_step: float = 0.05,
        readout_specs: Dict[str, ModalitySpec] = MODALITY_REGISTRY,
        num_latents: int = 1,
        dim: int = 512,
        depth: int = 0,
        dim_head: int = 64,
        cross_heads: int = 1,
        self_heads: int = 8,
        ffn_dropout: float = 0.2,
        lin_dropout: float = 0.4,
        atn_dropout: float = 0.0,
        rnn_dim: int = 256,
        num_rnn_layers: int = 1,
        rnn_dropout: float = 0.2,
        output_ca_ctx_lim: int = 3,
        emb_init_scale: float = 0.02,
        t_min: float = 1e-4,
        t_max: float = 10.0,
    ):
        super().__init__()

        self.sequence_length = sequence_length
        self.bin_width = bin_width
        self.bin_step = bin_step
        self.num_latents = num_latents
        self.dim = dim
        self.output_ca_ctx_lim = output_ca_ctx_lim
        self.readout_specs = readout_specs

        # Precompute latent tokens (depend only on bin_width/num_latents, not
        # sequence_length, so constant across tokenize calls)
        latent_step = bin_width / num_latents
        self._latent_index, self._latent_timestamps = create_linspace_latent_tokens(
            0,
            bin_width,
            step=latent_step,
            num_latents_per_step=num_latents,
        )

        self.dropout = nn.Dropout(p=lin_dropout)

        # embeddings
        self.unit_emb = InfiniteVocabEmbedding(dim, init_scale=emb_init_scale)
        self.session_emb = InfiniteVocabEmbedding(dim, init_scale=emb_init_scale)
        self.token_type_emb = Embedding(4, dim, init_scale=emb_init_scale)
        self.task_emb = Embedding(
            len(readout_specs) + 1, dim, init_scale=emb_init_scale
        )
        self.latent_emb = Embedding(num_latents, dim, init_scale=emb_init_scale)
        self.rotary_emb = RotaryTimeEmbedding(
            head_dim=dim_head,
            rotate_dim=dim_head // 2,
            t_min=t_min,
            t_max=t_max,
        )

        # encoder
        self.enc_atn = RotaryCrossAttention(
            dim=dim,
            heads=cross_heads,
            dropout=atn_dropout,
            dim_head=dim_head,
            rotate_value=True,
        )
        self.enc_ffn = nn.Sequential(
            nn.LayerNorm(dim), FeedForward(dim=dim, dropout=ffn_dropout)
        )

        # processor
        self.proc_layers = nn.ModuleList([])
        for _ in range(depth):
            self.proc_layers.append(
                nn.ModuleList(
                    [
                        RotarySelfAttention(
                            dim=dim,
                            heads=self_heads,
                            dropout=atn_dropout,
                            dim_head=dim_head,
                            rotate_value=True,
                        ),
                        nn.Sequential(
                            nn.LayerNorm(dim),
                            FeedForward(dim=dim, dropout=ffn_dropout),
                        ),
                    ]
                )
            )

        # backbone (GRU)
        self.backbone = GRU(
            input_size=num_latents * dim,
            hidden_size=rnn_dim,
            num_layers=num_rnn_layers,
            dropout=rnn_dropout,
        )

        # decoder
        self.dec_atn = RotaryCrossAttention(
            dim=dim,
            context_dim=rnn_dim,
            heads=cross_heads,
            dropout=atn_dropout,
            dim_head=dim_head,
            rotate_value=True,
        )
        self.dec_ffn = nn.Sequential(
            nn.LayerNorm(dim), FeedForward(dim=dim, dropout=ffn_dropout)
        )

        # readout
        self.readout = MultitaskReadout(
            dim=dim,
            readout_specs=readout_specs,
        )

    def forward(
        self,
        *,
        # input sequence (binned)
        spike_unit_index: TensorType["n_intervals", "batch", "n_in", int],
        spike_timestamps: TensorType["n_intervals", "batch", "n_in", float],
        spike_type: TensorType["n_intervals", "batch", "n_in", int],
        input_mask: TensorType["n_intervals", "batch", "n_in", bool],
        n_intervals: TensorType["batch", int],
        # latent sequence
        latent_index: TensorType["batch", "n_latent", int],
        latent_timestamps: TensorType["batch", "n_latent", float],
        # output sequence
        output_session_index: TensorType["batch", "n_out", int],
        output_timestamps: TensorType["batch", "n_out", float],
        output_decoder_index: TensorType["batch", "n_out", int],
        output_bin_index: TensorType["batch", "n_out", int],
        unpack_output: bool = False,
    ) -> List[Dict[str, TensorType["*nqueries", "*nchannelsout"]]]:

        if self.unit_emb.is_lazy():
            raise ValueError(
                "Unit vocabulary has not been initialized, please use "
                "`model.unit_emb.initialize_vocab(unit_ids)`"
            )

        if self.session_emb.is_lazy():
            raise ValueError(
                "Session vocabulary has not been initialized, please use "
                "`model.session_emb.initialize_vocab(session_ids)`"
            )

        B = spike_unit_index.shape[0]
        num_intervals = max(n_intervals)
        latents = self.latent_emb(latent_index)

        # Build output queries
        output_queries = self.task_emb(output_decoder_index) + self.session_emb(
            output_session_index
        )
        output_timestamp_emb = self.rotary_emb(output_timestamps)

        # Batch encoder across all bins: fold [B, N_intervals, N_spikes]
        # into [B*N_intervals, N_spikes]
        spike_unit_index_flat = spike_unit_index.reshape(B * num_intervals, -1)
        spike_timestamps_flat = spike_timestamps.reshape(B * num_intervals, -1)
        spike_type_flat = spike_type.reshape(B * num_intervals, -1)
        input_mask_flat = input_mask.reshape(B * num_intervals, -1)

        inputs = self.unit_emb(spike_unit_index_flat) + self.token_type_emb(
            spike_type_flat
        )

        # Spike timestamps are bin-relative, latent timestamps are constant
        input_timestamp_emb = self.rotary_emb(spike_timestamps_flat)
        latent_timestamp_emb = self.rotary_emb(
            latent_timestamps[:1].expand(B * num_intervals, -1)
        )

        # Repeat latents for each bin
        latents_rep = latents.repeat(num_intervals, 1, 1)

        # Encoder cross-attention (single batched call)
        ca_output = latents_rep + self.enc_atn(
            latents_rep,
            inputs,
            latent_timestamp_emb,
            input_timestamp_emb,
            input_mask_flat,
        )

        # Processor layers
        for self_attn, self_ff in self.proc_layers:
            ca_output = ca_output + self.dropout(
                self_attn(ca_output, latent_timestamp_emb)
            )
            ca_output = ca_output + self.dropout(self_ff(ca_output))

        # FFN after proc_layers (original ordering)
        ca_output = ca_output + self.enc_ffn(ca_output)

        # Flatten latents and reshape back to [B, N_intervals, num_latents*dim]
        ca_output = torch.flatten(ca_output, start_dim=1)
        outputs = ca_output.reshape(B, num_intervals, -1)

        # Backbone (GRU)
        outputs = self.backbone(outputs)  # [B, n_intervals, rnn_dim]

        # Build causal context mask for decoder cross-attention.
        # Each output query attends only to the current and recent bins.
        # Padded outputs (bin_index=0) are clamped to 1 to avoid empty masks;
        # they are ignored downstream since no readout spec has id=0.
        bins = torch.arange(num_intervals, device=output_timestamps.device)
        bin_idx = output_bin_index.clamp(min=1).unsqueeze(-1)  # [B, N_out, 1]
        ctx_mask = (bins < bin_idx) & (bins >= bin_idx - self.output_ca_ctx_lim)

        output_latent_timestamps = torch.arange(
            self.bin_width,
            num_intervals * self.bin_width + 1e-6,
            self.bin_width,
            device=output_timestamps.device,
        )
        olt_emb = self.rotary_emb(
            output_latent_timestamps.unsqueeze(0).repeat(output_timestamps.shape[0], 1)
        )

        # Decoder cross-attention
        l = output_queries + self.dec_atn(
            output_queries,
            outputs,
            output_timestamp_emb,
            olt_emb,
            context_mask=ctx_mask,
        )
        output_latents = l + self.dec_ffn(l)

        # Multitask readout
        return self.readout(
            output_embs=output_latents,
            output_readout_index=output_decoder_index,
            unpack_output=unpack_output,
        )

    @staticmethod
    def _bin_spikes(
        spike_unit_index, spike_timestamps, n_intervals, interval_start, interval_end
    ):
        """Assign spikes to temporal bins using binary search and vectorized scatter.

        Handles both overlapping (bin_step < bin_width) and non-overlapping bins.
        Replaces the O(n_intervals * n_spikes) broadcast + Python loop with
        O(n_spikes * log(n_intervals)) search + vectorized scatter.

        Returns [n_intervals, max_spikes] tensors for unit_index, timestamps
        (bin-relative), type (zeros), and attention mask. Empty bins get a single
        blank token (unit=0, ts=0.0, visible to attention).
        """
        # Find the range of bins each spike belongs to via binary search.
        # For non-overlapping bins each spike maps to exactly one bin.
        first_bin = np.searchsorted(interval_end, spike_timestamps, side="right")
        last_bin = np.searchsorted(interval_start, spike_timestamps, side="right") - 1
        bins_per_spike = np.maximum(last_bin - first_bin + 1, 0)
        total = int(bins_per_spike.sum())

        if total == 0:
            # No valid spike-bin assignments: return blank tokens for every bin
            return (
                torch.zeros(n_intervals, 1, dtype=torch.long),
                torch.zeros(n_intervals, 1),
                torch.zeros(n_intervals, 1, dtype=torch.long),
                torch.ones(n_intervals, 1, dtype=torch.bool),
            )

        # Expand to one row per (spike, bin) pair
        spike_sel = np.repeat(np.arange(len(spike_timestamps)), bins_per_spike)
        cum = np.cumsum(bins_per_spike)
        bin_idx = np.repeat(first_bin, bins_per_spike) + (
            np.arange(total) - np.repeat(cum - bins_per_spike, bins_per_spike)
        )

        # Sort by bin and compute within-bin positions
        order = np.argsort(bin_idx, kind="stable")
        bin_sorted = bin_idx[order]
        counts = np.bincount(bin_idx, minlength=n_intervals)
        cum_counts = np.empty(n_intervals + 1, dtype=np.intp)
        cum_counts[0] = 0
        np.cumsum(counts, out=cum_counts[1:])
        pos = np.arange(total) - cum_counts[bin_sorted]

        # Scatter into padded [n_intervals, max_spikes] tensors
        max_spikes = max(int(counts.max()), 1)
        b = torch.from_numpy(bin_sorted.astype(np.int64))
        p = torch.from_numpy(pos.astype(np.int64))

        unit_2d = torch.zeros(n_intervals, max_spikes, dtype=torch.long)
        ts_2d = torch.zeros(n_intervals, max_spikes)
        type_2d = torch.zeros(n_intervals, max_spikes, dtype=torch.long)
        mask_2d = torch.zeros(n_intervals, max_spikes, dtype=torch.bool)

        unit_2d[b, p] = torch.from_numpy(
            spike_unit_index[spike_sel[order]].astype(np.int64)
        )
        ts_2d[b, p] = torch.from_numpy(
            (spike_timestamps[spike_sel[order]] - interval_start[bin_sorted]).astype(
                np.float32
            )
        )
        mask_2d[b, p] = True
        mask_2d[counts == 0, 0] = True  # blank token for empty bins

        return unit_2d, ts_2d, type_2d, mask_2d

    def tokenize(self, data: Data) -> Dict:
        r"""Tokenizer used to tokenize Data for the POSSM model.

        This tokenizer can be called as a transform. If you are applying multiple
        transforms, make sure to apply this one last.

        This code runs on CPU. Do not access GPU tensors inside this function.
        """

        start, end = 0, self.sequence_length

        # Compute bin structure
        end_rounded = round(end, 5)
        n_intervals = int(
            np.floor(np.round((end_rounded - self.bin_width) / self.bin_step, 5)) + 1
        )
        interval_end = np.round(
            np.arange(self.bin_width, end_rounded + 1e-6, self.bin_step), 5
        )
        interval_start = np.round(interval_end - self.bin_width, 5)
        interval_end[-1] = end_rounded + 1e-6

        # Prepare input
        unit_ids = data.units.id
        spike_unit_index = data.spikes.unit_index
        spike_timestamps = data.spikes.timestamps

        # Map local unit indices to global unit indices
        local_to_global_map = np.array(self.unit_emb.tokenizer(unit_ids))
        spike_unit_index = local_to_global_map[spike_unit_index]

        # Prepare outputs
        session_index = self.session_emb.tokenizer(data.session.id)

        (
            output_timestamps,
            output_values,
            output_task_index,
            output_weights,
            output_eval_mask,
        ) = prepare_for_multitask_readout(
            data,
            self.readout_specs,
        )

        session_index = np.repeat(session_index, len(output_timestamps))

        # Bin spikes using vectorized binary search + scatter
        spike_unit_index_2d, spike_timestamps_2d, spike_type_2d, input_mask_2d = (
            self._bin_spikes(
                spike_unit_index,
                spike_timestamps,
                n_intervals,
                interval_start,
                interval_end,
            )
        )

        # Compute output_bin_index: which bin each output belongs to (1-based)
        output_bin_index = (
            np.searchsorted(interval_end, output_timestamps, side="right") + 1
        )

        data_dict = {
            "model_inputs": {
                # input sequence (binned)
                "spike_unit_index": pad2d(spike_unit_index_2d),
                "spike_timestamps": pad2d(spike_timestamps_2d),
                "spike_type": pad2d(spike_type_2d),
                "input_mask": track_mask2d(input_mask_2d),
                "n_intervals": n_intervals,
                # latent sequence
                "latent_index": self._latent_index,
                "latent_timestamps": self._latent_timestamps,
                # output sequence
                "output_session_index": pad8(session_index),
                "output_timestamps": pad8(output_timestamps),
                "output_decoder_index": pad8(output_task_index),
                "output_bin_index": pad8(output_bin_index),
            },
            # ground truth targets
            "target_values": chain(output_values, allow_missing_keys=True),
            "target_weights": chain(output_weights, allow_missing_keys=True),
            # extra fields for evaluation
            "session_id": data.session.id,
            "absolute_start": data.absolute_start,
            "eval_mask": chain(output_eval_mask, allow_missing_keys=True),
        }

        return data_dict

    def freeze_middle(self) -> List[nn.Module]:
        """Freeze all parameters except unit and session embeddings."""
        frozen_modules = []
        banned_modules = [self.unit_emb, self.session_emb]
        for module in self.children():
            if module in banned_modules:
                continue
            for param in module.parameters():
                param.requires_grad = False
            frozen_modules.append(module)
        return frozen_modules

    def freeze_middle_newTask(self) -> List[nn.Module]:
        """Freeze all parameters except unit/session embeddings and readout."""
        frozen_modules = []
        banned_modules = [self.unit_emb, self.session_emb, self.readout]
        for module in self.children():
            if module in banned_modules:
                continue
            for param in module.parameters():
                param.requires_grad = False
            frozen_modules.append(module)
        return frozen_modules

    def unfreeze_middle(self) -> None:
        """Unfreeze all parameters."""
        for module in self.children():
            for param in module.parameters():
                param.requires_grad = True
