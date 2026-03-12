from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
from torch import cat, tensor, int64
from einops import rearrange, repeat
import torch.nn as nn
from torchtyping import TensorType
from temporaldata import Data

from torch_brain.data import chain, pad8, track_mask8
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
    create_start_end_unit_tokens,
)

class CaPOYO(nn.Module):
    """
    CaPOYO (Calcium POYO+) model from `Azabou et al. 2025, Multi-session, multi-task neural decoding
    from distinct cell-types and brain regions <https://openreview.net/forum?id=IuU0wcO0mo>`_.

    CaPOYO is a transformer-based model for neural decoding from calcium imaging recordings.
    It extends the POYO+ architecture with a calcium value map.
    """

    def __init__(
        self,
        *,
        sequence_length: float,
        readout_specs: Dict[str, ModalitySpec] = MODALITY_REGISTRY,
        latent_step: float,
        num_latents_per_step: int = 64,
        dim: int = 512,
        depth: int = 2,
        dim_head: int = 64,
        cross_heads: int = 1,
        self_heads: int = 8,
        ffn_dropout: float = 0.2,
        lin_dropout: float = 0.4,
        atn_dropout: float = 0.0,
        emb_init_scale: float = 0.02,
        t_min: float = 1e-4,
        t_max: float = 2.0627,
        data_modality: str = "calcium",  # "calcium" or "eeg"
        eeg_config: Optional[Dict] = None,  # EEG-specific config (patch_size, num_channels, etc.)
    ):
        super().__init__()

        self._validate_params(sequence_length, latent_step)

        self.latent_step = latent_step
        self.num_latents_per_step = num_latents_per_step
        self.sequence_length = sequence_length
        self.readout_specs = readout_specs
        self.data_modality = data_modality

        # Store EEG-specific config
        if data_modality == "eeg":
            self.eeg_config = eeg_config
        else:
            self.eeg_config = None

        # input value map
        self.input_value_map = nn.Linear(1, dim // 2)
        nn.init.trunc_normal_(self.input_value_map.weight, 0, emb_init_scale)
        nn.init.zeros_(self.input_value_map.bias)
        # ^ initialize weights for faster convergence

        # embeddings
        self.unit_emb = InfiniteVocabEmbedding(dim // 2, init_scale=emb_init_scale)
        self.session_emb = InfiniteVocabEmbedding(dim, init_scale=emb_init_scale)
        self.token_type_emb = Embedding(4, dim, init_scale=emb_init_scale)
        self.task_emb = Embedding(len(readout_specs), dim, init_scale=emb_init_scale)
        self.latent_emb = Embedding(
            num_latents_per_step, dim, init_scale=emb_init_scale
        )
        self.rotary_emb = RotaryTimeEmbedding(
            head_dim=dim_head,
            rotate_dim=dim_head // 2,
            t_min=t_min,
            t_max=t_max,
        )

        self.dropout = nn.Dropout(p=lin_dropout)

        # encoder layer
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

        # process layers
        self.proc_layers = nn.ModuleList([])
        for i in range(depth):
            self.proc_layers.append(
                nn.Sequential(
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
                )
            )

        # decoder layer
        self.dec_atn = RotaryCrossAttention(
            dim=dim,
            heads=cross_heads,
            dropout=atn_dropout,
            dim_head=dim_head,
            rotate_value=False,
        )
        self.dec_ffn = nn.Sequential(
            nn.LayerNorm(dim), FeedForward(dim=dim, dropout=ffn_dropout)
        )

        # Output projections + loss
        self.readout = MultitaskReadout(
            dim=dim,
            readout_specs=readout_specs,
        )

        self.dim = dim

    def forward(
        self,
        inputs: Optional[TensorType["batch", "n_in", "dim"]] = None,
        *,
        # input sequence
        input_unit_index: Optional[TensorType["batch", "n_in", int]] = None,
        input_timestamps: Optional[TensorType["batch", "n_in", float]] = None,
        input_values: Optional[TensorType["batch", "n_in", float]] = None,
        input_mask: Optional[TensorType["batch", "n_in", bool]] = None,
        # latent sequence
        latent_index: Optional[TensorType["batch", "n_latent", int]] = None,
        latent_timestamps: Optional[TensorType["batch", "n_latent", float]] = None,
        # output sequence
        output_session_index: Optional[TensorType["batch", "n_out", int]] = None,
        output_timestamps: Optional[TensorType["batch", "n_out", float]] = None,
        output_decoder_index: Optional[TensorType["batch", "n_out", int]] = None,
        unpack_output: bool = False,
    ) -> Tuple[List[Dict[str, TensorType["*nqueries", "*nchannelsout"]]]]:
        """Forward pass of the POYO+ model.

        The model processes input spike sequences through its encoder-processor-decoder
        architecture to generate task-specific predictions.

        Args:
            inputs: Pre-embedded input tokens (B, n_in, dim) - for EEG mode with DIVER features
            input_unit_index: Indices of input units (for calcium mode)
            input_timestamps: Timestamps of input tokens
            input_values: Calcium values of input sequence (for calcium mode)
            input_mask: Mask for input sequence
            latent_index: Indices for latent tokens
            latent_timestamps: Timestamps for latent tokens
            output_session_index: Index of the recording session
            output_timestamps: Timestamps for output predictions
            output_decoder_index: Indices indicating which decoder to use
            unpack_output: Whether to unpack output

        Returns:
            Output predictions for each task
        """

        # Mode 1: Pre-embedded inputs (EEG mode with DIVER features)
        if inputs is not None:
            # inputs: (B, C*N, dim) - DIVER features
            # Need to convert to POYO-style format (value_emb + channel_emb)

            if self.data_modality == "eeg" and input_unit_index is not None:
                # EEG mode with channel identity: split DIVER features and add channel embedding
                # This creates a POYO-compatible representation:
                # - First d_model//2: DIVER features (value-like)
                # - Last d_model//2: Channel identity embedding

                # Project DIVER features to half dimension
                if not hasattr(self, 'diver_feature_proj'):
                    self.diver_feature_proj = nn.Linear(self.dim, self.dim // 2)
                    nn.init.xavier_uniform_(self.diver_feature_proj.weight)
                    nn.init.zeros_(self.diver_feature_proj.bias)
                    self.diver_feature_proj = self.diver_feature_proj.to(inputs.device)

                value_features = self.diver_feature_proj(inputs)  # (B, C*N, dim//2)

                # Get channel identity embedding
                if self.unit_emb.is_lazy():
                    raise ValueError(
                        "Unit vocabulary has not been initialized. For EEG mode with DIVER features, "
                        "please initialize channel IDs via `model.unit_emb.initialize_vocab(channel_ids)`"
                    )
                channel_emb = self.unit_emb(input_unit_index)  # (B, C*N, dim//2)

                # Combine in POYO style
                inputs = cat([value_features, channel_emb], dim=-1)  # (B, C*N, dim)
            else:
                # Direct pre-embedded inputs (already in POYO format or don't need conversion)
                pass
        # Mode 2: Calcium mode (original CaPOYO)
        else: 
            # input embedding
            inputs = cat(
                (self.input_value_map(input_values), self.unit_emb(input_unit_index)),
                dim=-1,
            )
        input_timestamp_emb = self.rotary_emb(input_timestamps)

        # latents
        latents = self.latent_emb(latent_index)
        latent_timestamp_emb = self.rotary_emb(latent_timestamps)

        # outputs
        output_queries = self.session_emb(output_session_index) + self.task_emb(
            output_decoder_index
        )
        output_timestamp_emb = self.rotary_emb(output_timestamps)

        # encode
        latents = latents + self.enc_atn(
            latents,
            inputs,
            latent_timestamp_emb,
            input_timestamp_emb,
            input_mask,
        )
        latents = latents + self.enc_ffn(latents)

        # process
        for self_attn, self_ff in self.proc_layers:
            latents = latents + self.dropout(self_attn(latents, latent_timestamp_emb))
            latents = latents + self.dropout(self_ff(latents))

        # decode
        output_queries = output_queries + self.dec_atn(
            output_queries,
            latents,
            output_timestamp_emb,
            latent_timestamp_emb,
        )
        output_latents = output_queries + self.dec_ffn(output_queries)

        # multitask readout layer, each task has a seperate linear readout layer
        output = self.readout(
            output_embs=output_latents,
            output_readout_index=output_decoder_index,
            unpack_output=unpack_output,
        )

        return output

    def tokenize(self, data: Data) -> Dict:
        r"""Tokenizer used to tokenize Data for the POYO+ model.

        This tokenizer can be called as a transform. If you are applying multiple
        transforms, make sure to apply this one last.

        This code runs on CPU. Do not access GPU tensors inside this function.

        Supports two data modalities:
        - "calcium": Original CaPOYO tokenization for calcium imaging
        - "eeg": EEG tokenization that returns raw (C, L) format
        """

        if self.data_modality == "calcium":
            return self._tokenize_calcium(data)
        elif self.data_modality == "eeg":
            return self._tokenize_eeg(data)

    def tokenize_from_shape(self, C, N, data_info_list):
        r"""Create timestamps and latents from shape information (for EEG forward pass).

        This is used when DIVER encoder has already processed the data and we need
        to create timestamps/latents that match the DIVER patch structure, without
        going through the full tokenize() path that expects raw Data objects.

        Args:
            C: Number of channels
            N: Number of patches
            data_info_list: List of dicts, each containing:
                - 'session_id': str
                - 'task_configs': List[Dict] with 'task_name' and 'timestamps'

        Returns:
            Dict with keys:
                - 'input_timestamps': (C*N,) numpy array
                - 'latent_index': (n_latent,) numpy array
                - 'latent_timestamps': (n_latent,) numpy array
                - 'output_session_index': (n_out,) numpy array
                - 'output_timestamps': (n_out,) numpy array
                - 'output_decoder_index': (n_out,) numpy array
        """
        # Create input timestamps and latents using helpers
        input_timestamps = self._create_input_timestamps_eeg(C, N)
        latent_index, latent_timestamps = self._create_latent_tokens()

        # Prepare outputs from data_info_list
        # Collect per-batch and pad to create (B, max_queries) tensors
        batch_output_timestamps = []
        batch_output_session_indices = []
        batch_output_task_indices = []

        # First pass: collect per-batch data and find max queries
        max_queries = 0
        batch_data = []

        for info in data_info_list:
            session_id = info['session_id']
            task_configs = info.get('task_configs', [])

            # Get session index
            session_idx = self.session_emb.tokenizer(session_id)

            # Collect for this batch element
            elem_timestamps = []
            elem_sessions = []
            elem_tasks = []

            for task_config in task_configs:
                task_name = task_config['task_name']
                task_timestamps = task_config['timestamps']
                task_idx = self.readout_specs[task_name]['id']

                num_queries = len(task_timestamps)
                elem_timestamps.extend(task_timestamps)
                elem_sessions.extend([session_idx] * num_queries)
                elem_tasks.extend([task_idx] * num_queries)

            batch_data.append({
                'timestamps': elem_timestamps,
                'sessions': elem_sessions,
                'tasks': elem_tasks,
                'num_queries': len(elem_timestamps)
            })
            max_queries = max(max_queries, len(elem_timestamps))

        # Second pass: pad to max_queries
        for data in batch_data:
            n_queries = data['num_queries']
            pad_len = max_queries - n_queries

            # Pad with zeros (will be ignored by masking in readout)
            batch_output_timestamps.append(
                data['timestamps'] + [0.0] * pad_len
            )
            batch_output_session_indices.append(
                data['sessions'] + [0] * pad_len
            )
            batch_output_task_indices.append(
                data['tasks'] + [0] * pad_len
            )

        # Convert to numpy arrays with shape (B, max_queries)
        output_timestamps = np.array(batch_output_timestamps, dtype=np.float32)
        output_session_index = np.array(batch_output_session_indices, dtype=np.int64)
        output_decoder_index = np.array(batch_output_task_indices, dtype=np.int64)

        return {
            'input_timestamps': input_timestamps,  # (C*N,) - shared across batch
            'latent_index': latent_index,  # (n_latent,) - shared across batch
            'latent_timestamps': latent_timestamps,  # (n_latent,) - shared across batch
            'output_session_index': output_session_index,  # (B, max_queries)
            'output_timestamps': output_timestamps,  # (B, max_queries)
            'output_decoder_index': output_decoder_index,  # (B, max_queries)
        }

    def _tokenize_calcium(self, data: Data) -> Dict:
        """Original calcium imaging tokenization."""
        # context window
        start, end = 0, self.sequence_length

        ### prepare input
        calcium_traces = data.calcium_traces
        unit_ids = data.units.id

        T, N = calcium_traces.df_over_f.shape
        input_timestamps = repeat(calcium_traces.timestamps, "T -> (T N)", T=T, N=N)

        ### prepare calcium values
        T, N = calcium_traces.df_over_f.shape
        input_values = rearrange(calcium_traces.df_over_f, "T N -> (T N) 1", T=T, N=N)

        # input unit indices
        local_to_global_map = np.array(self.unit_emb.tokenizer(unit_ids))
        input_unit_index = [local_to_global_map[i] for i in range(len(unit_ids))]
        input_unit_index = tensor(input_unit_index, dtype=int64)
        input_unit_index = repeat(input_unit_index, "N -> (T N)", T=T, N=N)

        ### prepare latents
        latent_index, latent_timestamps = self._create_latent_tokens()

        ### prepare outputs
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

        data_dict = {
            "model_inputs": {
                # input sequence
                "input_unit_index": pad8(input_unit_index),
                "input_timestamps": pad8(input_timestamps),
                "input_values": pad8(input_values),
                "input_mask": track_mask8(input_unit_index),
                # latent sequence
                "latent_index": latent_index,
                "latent_timestamps": latent_timestamps,
                # output sequence
                "output_session_index": pad8(session_index),
                "output_timestamps": pad8(output_timestamps),
                "output_decoder_index": pad8(output_task_index),
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

    def _create_input_timestamps_eeg(self, C, N):
        """Create input timestamps for EEG patches (C*N tokens).

        Args:
            C: Number of channels
            N: Number of patches

        Returns:
            input_timestamps: (C*N,) numpy array
        """
        patch_duration = self.sequence_length / N
        patch_centers = np.arange(N, dtype=np.float32) * patch_duration + patch_duration / 2
        # Repeat for each channel: (N,) -> (C, N) -> (C*N,)
        input_timestamps = np.tile(patch_centers, C)
        return input_timestamps

    def _create_latent_tokens(self):
        """Create latent tokens using linspace.

        Returns:
            latent_index: (n_latent,) numpy array
            latent_timestamps: (n_latent,) numpy array
        """
        start, end = 0, self.sequence_length
        latent_index, latent_timestamps = create_linspace_latent_tokens(
            start, end,
            step=self.latent_step,
            num_latents_per_step=self.num_latents_per_step,
        )
        return latent_index, latent_timestamps

    def _tokenize_eeg(self, data: Data) -> Dict:
        """EEG tokenization that returns raw ormat for DIVER processing."""
        ### Extract EEG data
        eeg_data = data.eeg.values 
        x = eeg_data.T  

        C, L = x.shape
        patch_size = self.eeg_config["patch_size"]

        N = L // patch_size  # num_patches

        ### Create input timestamps and latents using helpers
        input_timestamps = self._create_input_timestamps_eeg(C, N)
        latent_index, latent_timestamps = self._create_latent_tokens()

        ### Prepare input_values and input_unit_index for POYO compatibility
        # Flatten (C, L) -> (C*L, 1) for input_values
        input_values = rearrange(x, 'c l -> (c l) 1')  # (C*L, 1)

        # Create input_unit_index: channel IDs repeated L times each
        # [0,0,..,0 (L times), 1,1,..,1 (L times), , C-1,C-1,..C-1]
        input_unit_index = np.repeat(np.arange(C, dtype=np.int64), L)  # (C*L,)

        # Create per-sample timestamps for input_values
        # Each sample gets timestamp based on its position in the sequence
        sample_rate = L / self.sequence_length  # samples per second
        sample_timestamps = np.arange(L, dtype=np.float32) / sample_rate  # (L,)
        input_timestamps_full = np.tile(sample_timestamps, C)  # (C*L,) - repeat for each channel

        ### prepare outputs
        session_index = self.session_emb.tokenizer(data.session.id)

        # Extract label/target information
        if hasattr(data, 'drifting_gratings') and hasattr(data.drifting_gratings, 'orientation_id'):
            label = data.drifting_gratings.orientation_id
            label_timestamps = data.drifting_gratings.timestamps
        else:
            # Fallback: single label at sequence center
            label = np.array([0])
            label_timestamps = np.array([self.sequence_length / 2])

        # Build output info (simplified for EEG)
        # This assumes a single classification task
        output_timestamps = label_timestamps
        output_task_index = np.zeros(len(label_timestamps), dtype=np.int64)
        session_index = np.repeat(session_index, len(output_timestamps))

        data_dict = {
            # Raw EEG for DIVER encoder
            "x": x,  # (C, L) - used by DIVERCaPOYOFineTuneModel
            "model_inputs": {
                # Input sequence - POYO-compatible format
                "input_values": pad8(input_values),  # (C*L, 1)
                "input_unit_index": pad8(input_unit_index),  # (C*L,)
                "input_timestamps": pad8(input_timestamps_full),  # (C*L,) - per-sample timestamps
                "input_mask": track_mask8(input_unit_index),  # (C*L,)
                # Latent sequence
                "latent_index": latent_index,
                "latent_timestamps": latent_timestamps,
                # Output sequence
                "output_session_index": pad8(session_index),
                "output_timestamps": pad8(output_timestamps),
                "output_decoder_index": pad8(output_task_index),
            },
            # Ground truth targets
            "target_values": {"label": label},  # Simple dict for single task
            "session_id": data.session.id,
        }

        return data_dict


    def _validate_params(self, sequence_length, latent_step):
        r"""Ensure: sequence_length, and latent_step are floating point numbers greater
        than zero. And sequence_length is a multiple of latent_step.
        """

        if not isinstance(sequence_length, float):
            raise ValueError("sequence_length must be a float")
        if not sequence_length > 0:
            raise ValueError("sequence_length must be greater than 0")
        self.sequence_length = sequence_length

        if not isinstance(latent_step, float):
            raise ValueError("latent_step must be a float")
        if not latent_step > 0:
            raise ValueError("latent_step must be greater than 0")
        self.latent_step = latent_step

        # check if sequence_length is a multiple of latent_step
        if abs(sequence_length % latent_step) > 1e-10:
            logging.warning(
                f"sequence_length ({sequence_length}) is not a multiple of latent_step "
                f"({latent_step}). This is a simple warning, and this behavior is allowed."
            )
