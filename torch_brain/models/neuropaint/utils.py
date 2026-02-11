import torch    

#########RoPE##########
# Copied from hf Llama
# Precompute cos and sin for RoPE
def get_cos_sin(dim, max_F, base=10000, dtype=torch.get_default_dtype(), device=None):
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        t = torch.arange(max_F, device=device, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

# Rotates half the hidden dims of the input.
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), -1)

# Applies RoPE to the query and key tensors.
def apply_rotary_pos_emb(q, k, pos_ids, cos, sin, unsqueeze_dim=1):

    cos = cos[pos_ids].unsqueeze(unsqueeze_dim) # Bx1xTxD
    sin = sin[pos_ids].unsqueeze(unsqueeze_dim) # Bx1xTxD

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed
##########################



""" Wrapper for config dict to access it using dot notation
"""
class DictConfig(dict):

    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = DictConfig(value)
        return value

    def get_dict(self):
        return dict(self)

#####


def random_mask(masking_mode, x, t_ratio = 0.3, r_ratio = 0.1):
    '''
    Adapted from https://github.com/facebookresearch/mae/blob/main/models_mae.py
    Perform per-sample random masking by per-sample shuffling. 

    in the paper, we only used masking_mode = 'region'.

    x: [B, T, R, D]
    t_ratio: ratio of time steps to mask
    r_ratio: ratio of brain regions to mask

    Returns:
    x_masked: [B, T_kept, R_kept, D]
    mask: [B, T, R, D]
    mask_R: [B, R]
    mask_T: [B, T]
    ids_restore_R: [B, R]
    ids_restore_T: [B, T]

    '''
    B, T, R, D = x.shape

    if masking_mode == 'time':
        # Mask along T dimension
        len_keep = int(T * (1 - t_ratio))
        noise = torch.rand(B, T, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_restore_T = ids_restore
        ids_restore_R = None

        ids_keep = ids_shuffle[:, :len_keep]  # [B, len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, R, D))
        
        mask = torch.ones([B, T], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        mask_T = mask
        mask_R = torch.zeros([B, R], device=x.device)

        # Expand mask to match x's shape
        mask = mask.unsqueeze(-1).unsqueeze(-1).expand(B, T, R, D)

    elif masking_mode == 'region':
        # Mask along R dimension
        len_keep = int(R * (1 - r_ratio))

        if len_keep < 1:
            print('Warning: len_keep < 1, setting len_keep = 1')
            len_keep = 1 # Ensure at least one region is kept

        noise = torch.rand(B, R, device=x.device)  # [B, R]
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_restore_R = ids_restore
        ids_restore_T = None
        
        ids_keep = ids_shuffle[:, :len_keep]  # [B, len_keep]
        x_masked = torch.gather(x, dim=2, index=ids_keep.unsqueeze(1).unsqueeze(-1).repeat(1, T, 1, D))
        
        mask = torch.ones([B, R], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        mask_R = mask
        mask_T = torch.zeros([B, T], device=x.device)

        # Expand mask to match x's shape
        mask = mask.unsqueeze(1).unsqueeze(-1).expand(B, T, R, D)

    elif masking_mode == 'time_region':
        
        # Mask along both T and R dimensions
        len_keep_T = int(T * (1 - t_ratio))
        len_keep_R = int(R * (1 - r_ratio))

        if len_keep_R < 1:
            print('Warning: len_keep_R < 1, setting len_keep_R = 1')
            len_keep_R = 1
        
        noise_T = torch.rand(B, T, device=x.device)  # [B, T]
        noise_R = torch.rand(B, R, device=x.device)  # [B, R]

        ids_shuffle_T = torch.argsort(noise_T, dim=1)
        ids_shuffle_R = torch.argsort(noise_R, dim=1)
        
        ids_restore_T = torch.argsort(ids_shuffle_T, dim=1)
        ids_restore_R = torch.argsort(ids_shuffle_R, dim=1)

        ids_keep_T = ids_shuffle_T[:, :len_keep_T]
        ids_keep_R = ids_shuffle_R[:, :len_keep_R]

        x_masked = torch.gather(
            x, 
            dim=1, 
            index=ids_keep_T.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, R, D)
        )
        x_masked = torch.gather(
            x_masked, 
            dim=2, 
            index=ids_keep_R.unsqueeze(1).unsqueeze(-1).repeat(1, len_keep_T, 1, D)
        )

        mask_T = torch.ones([B, T], device=x.device)
        mask_T[:, :len_keep_T] = 0
        mask_T = torch.gather(mask_T, dim=1, index=ids_restore_T)

        mask_R = torch.ones([B, R], device=x.device)
        mask_R[:, :len_keep_R] = 0
        mask_R = torch.gather(mask_R, dim=1, index=ids_restore_R)

        # Combine masks
        mask = mask_T.unsqueeze(-1).expand(B, T, R) | mask_R.unsqueeze(1).expand(B, T, R) #mask elements that are masked by either mask
        mask = mask.unsqueeze(-1).expand(B, T, R, D)

    else:
        raise ValueError(f"Invalid masking mode: {masking_mode}")

    return  x_masked, mask, mask_R, mask_T, ids_restore_R, ids_restore_T



def get_spikes_mask_from_mask(mask_R, mask_T, area_ind_unique, neuron_regions):
    '''
    Given a mask in the T/R space, which is (B,T) and (B, R)
    return a mask that masks out spikes, which is (B,T,N)
    
    mask_R: [B, R] #mask of regions
    mask_T: [B, T] #mask of time steps
    area_ind_unique: [R,] # index of R regions
    neuron_regions: [B, N] # region index of each neuron
    '''

    B, T = mask_T.shape
    N = neuron_regions.size(1)

    #mask time steps
    spikes_mask = mask_T.unsqueeze(-1).repeat(1,1,N)

    #mask neurons in the masked regions
    for b in range(B):
        mask_R_tmp = mask_R[b] # [R,]
        area_ind_masked_tmp = torch.where(mask_R_tmp == 1)[0] # [R_masked,]
        area_ind_masked = area_ind_unique[area_ind_masked_tmp]

        for r in area_ind_masked:
            neuron_masked = torch.where(neuron_regions[b] == r)[0]
            spikes_mask[b,:,neuron_masked] = 1
    
    return spikes_mask

def get_force_mask(B, T, R, t_mask_ind, r_mask_ind, device):
    '''
    B: batch size
    T: number of time steps
    R: number of regions
    t_mask_ind: the time step to be masked
    r_mask_ind: the region to be masked

    now only support mask 1 time step or 1 region

    '''

    mask_T = torch.zeros([B, T], device=device)
    mask_R = torch.zeros([B, R], device=device)
    ids_restore_R = None
    ids_restore_T = None

    if t_mask_ind is not None:
        mask_T[:, t_mask_ind] = 1
        tmp_T = torch.arange(T).to(device)
        # put t_mask_ind at the end and the rest at the beginning
        ids_shuffle_T = torch.cat([tmp_T[:t_mask_ind], tmp_T[t_mask_ind+1:], tmp_T[t_mask_ind].unsqueeze(0)], dim=0)
        ids_shuffle_T = ids_shuffle_T.unsqueeze(0).repeat(B, 1)
        ids_restore_T = torch.argsort(ids_shuffle_T, dim=1).to(device)
    
    if r_mask_ind is not None:
        mask_R[:, r_mask_ind] = 1
        tmp_R = torch.arange(R).to(device)
        # put r_mask_ind at the end and the rest at the beginning
        ids_shuffle_R = torch.cat([tmp_R[:r_mask_ind], tmp_R[r_mask_ind+1:], tmp_R[r_mask_ind].unsqueeze(0)], dim=0)
        ids_shuffle_R = ids_shuffle_R.unsqueeze(0).repeat(B, 1)
        ids_restore_R = torch.argsort(ids_shuffle_R, dim=1).to(device)

    mask = torch.logical_or(mask_T.unsqueeze(-1).repeat(1, 1, R), mask_R.unsqueeze(1).repeat(1, T, 1))

    force_mask ={'mask': mask, 
                 'mask_R': mask_R, 
                 'mask_T': mask_T, 
                 'ids_restore_R': ids_restore_R, 
                 'ids_restore_T': ids_restore_T}

    return force_mask