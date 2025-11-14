#%%
import os
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN

from utils.config_utils import DictConfig
from models.model_output import ModelOutput
from utils.mask import random_mask, get_spikes_mask_from_mask
import random

from utils.pos_embed import * 
from models.self_attention_layer import *


#torchbrain imports
from temporaldata import Interval
from torch_brain.data import Dataset
from torch_brain.utils.binning import bin_spikes


from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
from temporaldata import Data


#%%
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

@dataclass
class MAE_Output(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    regularization_loss: Optional[torch.FloatTensor] = None
    preds: Optional[torch.FloatTensor] = None
    targets: Optional[torch.FloatTensor] = None


class NeuralStitcher_cross_att(nn.Module):
    def __init__(self, 
                session_list:list,
                area_ind_list_list:list,
                areaoi_ind: np.ndarray,
                config: DictConfig):
        
        super().__init__()
        unit_embed_dict = {}

        for area_ind_list, session_ind in zip(area_ind_list_list, session_list):
            unit_embed_dict[str(session_ind)] = nn.Embedding(len(area_ind_list), config.unit_embed_dim)

        self.unit_embed_dict = nn.ModuleDict({k: v for k, v in unit_embed_dict.items()})
        self.dropout = nn.Dropout(config.dropout_x)

        self.region_to_indx = {r: i for i,r in enumerate(areaoi_ind)}
        self.region_embed = nn.Embedding(len(self.region_to_indx), config.region_embed_dim) # integer to 1xd vector

        self.hemisphere_embed = nn.Embedding(2, config.hemisphere_embed_dim) # hemisphere to 1xd vector

        #from (N,T) to (C,h)
        self.q = nn.Parameter(torch.randn(config.n_channels_per_region, config.hidden_size))
        self.key = nn.Linear(config.T + config.unit_embed_dim + config.region_embed_dim + config.hemisphere_embed_dim, config.hidden_size, bias=False)
        self.value = nn.Linear(config.T + config.unit_embed_dim + config.region_embed_dim + config.hemisphere_embed_dim, config.hidden_size, bias=False)
        self.n_channels_per_region = config.n_channels_per_region
        self.cross_att_dropout = nn.Dropout(config.dropout_cross_att)

        self.ln1 = nn.LayerNorm(config.T + config.unit_embed_dim + config.region_embed_dim + config.hemisphere_embed_dim)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.mlp = NeuralMLP(config.hidden_size, 2*config.hidden_size, 'gelu', True, dropout=0.5)

        #from (C,h) to (C,T)
        self.ff1 = nn.Linear(config.hidden_size, config.hidden_size*2)
        self.act = ACT2FN['gelu']
        self.ff2 = nn.Linear(config.hidden_size*2, config.T)


    def forward(self, x, eid, neuron_regions, is_left):
        # x: (B, T, N)
        B, T, N = x.size()
        
        x = self.dropout(x)
        
        unit_embed = self.unit_embed_dict[eid]
        unit_id = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        unit_vec = unit_embed(unit_id) # (B, N, D1)

        hemisphere_vec = self.hemisphere_embed(is_left) # (B, N, D3)

        region_id = torch.tensor([self.region_to_indx[r.item()] for r in neuron_regions[0]], device=x.device)
        region_id = region_id.unsqueeze(0).expand(B, -1)
        region_vec = self.region_embed(region_id) # (B, N, D2)

        x_transpose = x.transpose(1, 2) # (B, N, T)
        x_concat = torch.cat([x_transpose, unit_vec, region_vec, hemisphere_vec], dim=2) # (B, N, T+D1+D2+D3)

        x_concat = self.ln1(x_concat)

        latent_list = []
        area_ind_unique = neuron_regions[0].unique()
        q = self.q.unsqueeze(0).expand(B, -1, -1) # (B, C, h)
        for area_ind in area_ind_unique:
            neuron_indices = torch.where(neuron_regions[0] == area_ind)[0]
            x1 = x_concat[:,neuron_indices,:] 

            k = self.key(x1) # (B, N, h)
            v = self.value(x1) # (B, N, h)

            out = F.scaled_dot_product_attention(q, k, v, is_causal=False) # (B, C, h)

            latent = q + self.cross_att_dropout(out)
            latent = latent + self.mlp(self.ln2(latent)) #(B, C, h)
            latent_list.append(latent)

        latent_all = torch.stack(latent_list, dim=2) # (B, C, R, h)

        latent_all = self.ff2(self.act(self.ff1(latent_all))) # (B, C, R, T)

        latent_all = latent_all.transpose(1, 3).contiguous() # (B, T, R, C)

        return latent_all # (B, T, R, C)


class StitchDecoder(nn.Module):

    def __init__(self,
                 session_list:list,
                 n_channels:int,
                 area_ind_list_list:list,
                 areaoi_ind: Optional[np.ndarray] = None,
                 pr_max_dict: Optional[dict] = None):
    
        super().__init__()

        stitch_decoder_dict = {}
        linear_layer_dict = {}

        for i, area_ind in enumerate(areaoi_ind):
            n_channels_this_region = pr_max_dict[area_ind] + 10 # set # of latent factors based on upper bound of dim calculated from data
            linear_layer_dict[str(area_ind)] = nn.Linear(n_channels, n_channels_this_region)

        for area_ind_list, session_ind in zip(area_ind_list_list, session_list):
            stitch_decoder_dict[str(session_ind)] = {}
            area_ind_unique = np.unique(area_ind_list)

            for area_ind in area_ind_unique:
                neuron_indices = np.where(area_ind_list == area_ind)[0]
                n_channels_this_region = pr_max_dict[area_ind.item()] + 10 # set # of latent factors based on upper bound of dim calculated from data
                stitch_decoder_dict[str(session_ind)][str(area_ind.item())] = nn.Linear(n_channels_this_region, len(neuron_indices))
        
        self.stitch_decoder_dict = nn.ModuleDict({k: nn.ModuleDict(v) for k, v in stitch_decoder_dict.items()})
        self.linear_layer_dict = nn.ModuleDict(linear_layer_dict)
        
        self.register_buffer("areaoi_ind", torch.tensor(areaoi_ind))


    def forward(self, x, eid, neuron_regions):
        
        # x is (B, T, R_all, H)
        stitch_decoder = self.stitch_decoder_dict[eid]
        area_ind_unique = neuron_regions[0].unique()
        
        x1_dict = {}
        for area_ind in area_ind_unique:
            x_tmp = x[:,:,self.areaoi_ind==area_ind,:] # (B, T, H)
            x1_dict[str(area_ind.item())] = self.linear_layer_dict[str(area_ind.item())](x_tmp) # (B, T, n_channels_this_region)

        N = neuron_regions.size(1)
        pred_fr = torch.zeros(x.size(0), x.size(1), N, device=x.device)

        for area_ind in area_ind_unique:
            x2 = stitch_decoder[str(area_ind.item())](x1_dict[str(area_ind.item())]) # (B, T, 1, n_channels_this_region)
            neuron_indices = torch.where(neuron_regions[0] == area_ind)[0]
            pred_fr[:,:,neuron_indices] = torch.squeeze(x2, dim=2)
        return pred_fr # (B, T, N). for neurons not in the areaoi_ind, the prediction is 0
    

# mask input
class NeuralMaskLayer(nn.Module):

    def __init__(self, hidden_size, n_channels, config: DictConfig):
        super().__init__()

        self.bias = config.bias
        self.n_channels = n_channels    
        self.input_dim = self.n_channels*config.mult

        # One common embedding layer (common to all timesteps and areas)
        self.ff1 = nn.Linear(self.n_channels, self.input_dim, bias=config.bias)
        self.ff2 = nn.Linear(self.input_dim, hidden_size)
        self.act = ACT2FN[config.act] 

        # Embed prompt token
        self.use_prompt = config.use_prompt
        if self.use_prompt:
            self.mask_types = ['time', 'region', 'time_region']
            self.mask_to_indx = {r: i for i,r in enumerate(self.mask_types)}
            self.embed_prompt = nn.Embedding(len(self.mask_types), hidden_size) 

        # Regularization
        self.dropout = nn.Dropout(config.dropout)
        self.t_ratio = config.t_ratio
        
    def forward(
            self, 
            spikes:           torch.FloatTensor,      # (B, T, R, C)
            masking_mode:     Optional[str] = None,
            force_mask:       Optional[dict] = None
        ) -> Tuple[torch.FloatTensor,torch.LongTensor,torch.LongTensor]:   # (B, T_unmasked*R_unmasked, H1),  (T), (R) # here H1 = transformer.hidden_size - encoder.region_embed_dim

        B, _, _, C = spikes.size()
        
        #add masking, and only embed kept data after masking
        if force_mask:
            mask = force_mask['mask'].unsqueeze(-1).repeat(1,1,1,C) # (B, T, R, C)
            mask_R = force_mask['mask_R']
            mask_T = force_mask['mask_T']
            ids_restore_R = force_mask['ids_restore_R'] 
            ids_restore_T = force_mask['ids_restore_T']

            T_kept = torch.sum(mask_T[0,:]==0)
            R_kept = torch.sum(mask_R[0,:]==0)

            spikes_masked = spikes[mask==0].view(B, T_kept, R_kept, C)

        else:
            r_ratio = random.uniform(0, 0.6)
            if r_ratio < 0.05:
                r_ratio = 0
            
            spikes_masked,  mask, mask_R, mask_T, ids_restore_R, ids_restore_T  = random_mask(masking_mode, spikes, t_ratio = self.t_ratio, r_ratio = r_ratio)
            
            _, T_kept, R_kept, _ = spikes_masked.size()

        # from (B, T_kept, R_kept, C) to (B, T_kept, R_kept, H1)
        x = self.ff1(spikes_masked)
        x = self.act(x)
        x = self.ff2(x)

        x = x.view(B, T_kept*R_kept, -1) # (B, T_kept*R_kept, H1)

        # Prepend prompt token 
        if self.use_prompt:
            mask_idx = torch.tensor(self.mask_to_indx[masking_mode], dtype=torch.int64, device=spikes.device)
            x = torch.cat((self.embed_prompt(mask_idx)[None,None,:].expand(B,-1,-1), x), dim=1) 

        return self.dropout(x), mask, mask_R, mask_T, ids_restore_R, ids_restore_T

################stitchers + masker + encoder################
class NeuralEncoder(nn.Module):

    def __init__(
        self, 
        config: DictConfig,
        **kwargs
    ):
        super().__init__() 

        self.hidden_size = config.transformer.hidden_size
        self.n_layers = config.transformer.n_layers
        self.max_F = config.transformer.max_F #max number of tokens for RoPE

        # Build stitcher
        self.stitchers = NeuralStitcher_cross_att(kwargs['eids'], kwargs['area_ind_list_list'], kwargs['areaoi_ind'], config.stitcher)
        self.use_prompt = config.masker.use_prompt

        # Mask layer
        self.H1 = self.hidden_size-config.region_embed_dim #- config.trial_type_embed_dim
        self.masker = NeuralMaskLayer(self.H1, config.stitcher.n_channels_per_region, config.masker)
        self.R_all = len(kwargs['areaoi_ind'])
        self.region_embed = nn.Parameter(torch.randn(self.R_all, config.region_embed_dim))

        #self.trial_type_num = len(kwargs['trial_type_values'])
        #self.trial_type_embed = nn.Embedding(self.trial_type_num, config.trial_type_embed_dim)

        self.register_buffer("areaoi_ind", torch.tensor(kwargs['areaoi_ind']))

        #mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.H1))

        # Transformer
        self.layers = nn.ModuleList([Block(idx, self.max_F, config.transformer) for idx in range(self.n_layers)])
        self.ln = nn.LayerNorm(self.hidden_size) 


    def forward(
            self, 
            spikes:           torch.FloatTensor,  # (B, T, N)
            spikes_timestamp: torch.LongTensor,   # (B, T)
            area_ind_unique:  torch.LongTensor,   # (R,)
            neuron_regions:   Optional[torch.LongTensor] = None,  # (B, N)
            is_left:         Optional[torch.LongTensor] = None, # (B, N)
            trial_type:       Optional[torch.LongTensor] = None, # (B, )
            masking_mode:     Optional[str] = None,
            eid:              Optional[str] = None,
            force_mask:       Optional[dict] = None
    ) -> torch.FloatTensor:                     # (B, T_kept*R_kept+1, hidden_size)
        
        # stitcher
        x = self.stitchers(spikes, str(eid.item()), neuron_regions, is_left) # (B, T, R, n_channels)

        B, T, R, _ = x.size()

        # mask neural data and add prompt token if use_prompt
        x, mask, mask_R, mask_T, ids_restore_R, ids_restore_T = self.masker(
            x, masking_mode, force_mask
        ) # x is (B, T_kept*R_kept, H-region_embed_dim) if no prompt token, otherwise (B, T_kept*R_kept+1, H-region_embed_dim)

        R_kept = torch.sum(mask_R[0,:]==0)
        T_kept = torch.sum(mask_T[0,:]==0)

        if self.use_prompt:
            x_ = x[:, 1:, :]
        else:
            x_ = x

        x_ = x_.view(B, T_kept, R_kept, -1)
        #add masked tokens
        if ids_restore_R is not None:
            R_mask = R-R_kept
            mask_tokens_region = self.mask_token.repeat(x.shape[0], R_mask, 1)
            x_ = torch.cat((x_, mask_tokens_region.unsqueeze(1).repeat(1, T_kept, 1, 1)), dim=2)

        if ids_restore_T is not None:
            T_mask = T-T_kept
            mask_tokens_time = self.mask_token.repeat(x.shape[0], T_mask, 1)
            x_ = torch.cat((x_, mask_tokens_time.unsqueeze(2).repeat(1, 1, x_.size(2), 1)), dim=1)            

        #restore tokens order
        if ids_restore_R is not None:
            x_ = torch.gather(x_, dim=2, index=ids_restore_R.unsqueeze(1).unsqueeze(-1).repeat(1, T, 1, self.H1))
        if ids_restore_T is not None:
            x_ = torch.gather(x_, dim=1, index=ids_restore_T.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, R, self.H1))

        #x_ is (B, T, R, H- region_embed_dim)
        # add tokens for unrecorded regions
        x_all = self.mask_token.repeat(B, T*self.R_all, 1)
        x_all = x_all.view(B, T, self.R_all, -1)

        area_ind_unique_ind = torch.tensor([torch.where(self.areaoi_ind == val)[0].item() for val in area_ind_unique])

        x_all[:,:,area_ind_unique_ind,:] = x_
    
        # embed regions
        region_embed_vec = self.region_embed # (R_all, region_embed_dim)
        x_all = torch.cat((x_all, region_embed_vec[None, None, :, :].expand(B, T, -1, -1)), dim=-1) # (B, T, R_all, H )
        
        # embed trial types
        #trial_type_embed_vec = self.trial_type_embed(trial_type) # (B, trial_type_embed_dim)
        #x_all = torch.cat((x_all, trial_type_embed_vec[:, None, None, :].expand(-1, T, self.R_all, -1)), dim=-1) # (B, T, R_all, H)

        x_all = x_all.view(B, T*self.R_all, -1) # (B, T*R_all, H)

        #calculate timestamps for each kept token                                     
        timestamp_kept_expand = spikes_timestamp.unsqueeze(2).repeat(1, 1, self.R_all) # (B, T, R_all)
        timestamp = timestamp_kept_expand.view(B, -1) # (B, T*R_all)

        if self.use_prompt:
            timestamp = torch.cat((torch.zeros(B,1, device=spikes.device, dtype=torch.int64), timestamp+1), dim=1)
            x_all = torch.cat(x[:,0:1,:], x_all, dim=1)
        
        x = x_all
        
        # Forward transformer
        for layer in self.layers:
            x = layer(x, timestamp=timestamp)
        x = self.ln(x)

        if self.use_prompt:
            x = x[:, 1:, :]

        x = x.view(B, T, self.R_all, -1)

        return x, mask, mask_R, mask_T, ids_restore_R, ids_restore_T


class MAE_with_region_stitcher(nn.Module):

    def __init__(
        self, 
        config: DictConfig,
        **kwargs
    ):
        super().__init__()
        
        # Build encoder
        self.encoder = NeuralEncoder(config.encoder, **kwargs)
        self.use_prompt = config.encoder.masker.use_prompt

        #stitcher
        self.stitch_decoder = StitchDecoder(kwargs['eids'], 
                                            self.encoder.hidden_size, 
                                            kwargs['area_ind_list_list'], 
                                            kwargs['areaoi_ind'],
                                            kwargs['pr_max_dict'])

        # Build loss function
        self.loss_fn = nn.PoissonNLLLoss(reduction="none", log_input=True)

        # for torchbrain tokenize()
        session_id_strs = kwargs['session_id_strs']
        self.session_id_str_to_ind = {session_id_strs[i]: i for i in range(len(session_id_strs))}
        self.areaoi = kwargs['areaoi'] # list of area names of interest
        self.region_to_ind = {region: i for i, region in enumerate(self.areaoi)}
        #

    def forward(
        self, 
        spikes:           torch.FloatTensor,  # (bs, seq_len, N)
        spikes_timestamps: torch.LongTensor,   # (bs, seq_len)
        neuron_regions:   Optional[torch.LongTensor] = None,   # (bs, N)
        is_left:         Optional[torch.LongTensor] = None, # (bs, N)
        trial_type:       Optional[torch.LongTensor] = None, # (bs, )
        masking_mode:     Optional[str] = None,
        eid:              Optional[str] = None,
        with_reg:       Optional[bool] = False,
        force_mask:       Optional[dict] = None
    ) -> MAE_Output:  

        targets = spikes.clone()

        area_ind_unique = neuron_regions[0].unique()
        # Encode neural data: stitcher - masker - add masked token & embed region & time - transformer
        x, mask, mask_R, mask_T, ids_restore_R, ids_restore_T = self.encoder(spikes, spikes_timestamps, area_ind_unique, neuron_regions, is_left, trial_type, masking_mode, eid, force_mask)
        #mask is (B, T, R, H)
        #x is (B, T, R_all, H)

        regularization_loss = torch.mean(torch.abs(torch.diff(x, dim=1)))*0.1

        #decoder-side stitcher
        outputs = self.stitch_decoder(x, str(eid.item()), neuron_regions) 

        if with_reg:
            loss = torch.nanmean(self.loss_fn(outputs, targets)) + regularization_loss
        else:
            loss = torch.nanmean(self.loss_fn(outputs, targets))
        
        #n_examples = mask.sum()

        return MAE_Output(
            loss=loss,
            regularization_loss = regularization_loss,
            preds=outputs,
            targets=targets
        )
    
    def tokenize(self, data: Data) -> Dict:
        # held out data from a randomly selected area TO-DO? 

        spikes_data = bin_spikes(data.spikes,len(data.units.acronym), bin_size=0.01, right=True)
        spikes_data = spikes_data.T.contiguous() # (T,N)
        T = spikes_data.shape[0]
        area_acronym_list = data.units.acronym.copy() #area name acronym list for all neurons
        
        #this function needs to be simplied. 
        area_acronym_list = self.update_area_ind_list_and_areaoi_ind(area_acronym_list, self.areaoi) #update the acronym
        unit_filter_mask = np.isin(area_acronym_list, self.areaoi) #boolean mask for all neurons

        # Q: is there a way to do it only once for all trials in the data? 
        #data.units = data.units.select_by_mask(unit_filter_mask)

        spikes_data = spikes_data[:, unit_filter_mask]

        #only include spikes_data from neurons in areaoi
        area_acronym_list = area_acronym_list[unit_filter_mask]

        #combine some areas
        unit_rename_mask = np.isin(area_acronym_list, [b'VISa', b'VISam'])
        area_acronym_list[unit_rename_mask] = b'VISa'
        
        #get area ind from area acronym
        area_ind_list = np.array(list(map(self.region_to_ind.get, area_acronym_list)), dtype=np.int64)
        N = spikes_data.shape[1]
        #turn session id str to session ind
        session_id_str = data.session.id
        eid = self.session_id_str_to_ind[session_id_str]

        #get trial type
        trial_type = data.trials.choice #1 or 0 (instead of -1 or 1)

        data_dict = {
            "spikes_data": spikes_data,
            "spikes_timestamps": np.arange(T),
            "neuron_regions": area_ind_list,
            "is_left": np.zeros((N,)),
            "trial_type": trial_type,
            "masking_mode": 'region',
            "eid": eid,
        }
        return data_dict
