import torch
import numpy as np
import os
import warnings
from torch_brain.models.neuropaint import MAE_with_region_stitcher, NeuralStitcher_cross_att
#TO-DO add utils files to neuropaint_utils
from utils.config_utils import config_from_kwargs, update_config
from utils.utils import set_seed


from accelerate import Accelerator
from torch.optim.lr_scheduler import OneCycleLR
from trainer import trainer
import argparse
import pickle

#for torch_brain 
from torch_brain.data import Dataset, collate
import copy
from temporaldata import Interval
from torch_brain.data.sampler import TrialSampler

warnings.simplefilter("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["WANDB_IGNORE_COPY_ERR"] = "true"


#create dataset and dataloader
def make_loader(base_path, example_path, batch_size = 16):
    '''

    '''

    dataset = Dataset(
        root= base_path + "/processed",
        config= f"{example_path}/config/dataset/ibl_bwm_1sess.yaml",
    )
    
    areaoi = [b"PO", b"LP", b"DG", b"CA1", b"VISa", b"VPM", b"APN", b"MRN"]
    region_to_ind = {region: i for i, region in enumerate(areaoi)}
    train_sampling_intervals_dict = {}
    val_sampling_intervals_dict = {}
    area_ind_list_dict = {}
    heldout_area_dict = {}
    unit_filter_dict = {}
    unit_heldout_filter_dict = {}
    
    for session_idx, session_id in enumerate(dataset.get_session_ids()):
        data = copy.deepcopy(dataset.get_recording_data(session_id))

        samp_interval = Interval(
            start=data.trials.stim_on_time - 0.5,
            end=data.trials.stim_on_time + 1.5,
        )
        train_intervals, val_intervals = samp_interval.split([0.8, 0.2])
        train_sampling_intervals_dict[session_id] = train_intervals
        val_sampling_intervals_dict[session_id] = val_intervals

        #get area_ind_list
        area_acronym_list = data.units.acronym.copy() #area name acronym list for all neurons
        unit_filter_mask = np.isin(area_acronym_list, areaoi) #boolean mask for all neurons

        #held out one region randomly
        np.random.seed(session_idx+1000)
        heldout_region_ind = np.random.choice(areaoi_ind_exist, 1)[0]
        heldout_area_dict[session_id] = areaoi[heldout_region_ind]
        unit_heldout_filter = (area_ind_list != heldout_region_ind)
        unit_heldout_filter_dict[session_id] = unit_heldout_filter

        unit_filter_mask = unit_filter_mask & unit_heldout_filter

        #only include spikes_data from neurons in areaoi
        area_acronym_list = area_acronym_list[unit_filter_mask]

        #combine some areas
        unit_rename_mask = np.isin(area_acronym_list, [b'VISa', b'VISam'])
        area_acronym_list[unit_rename_mask] = b'VISa'
        
        #get area ind from area acronym
        area_ind_list = np.array(list(map(region_to_ind.get, area_acronym_list)), dtype=np.int64)
        area_ind_list_dict[session_id] = area_ind_list

        # choose only the neurons in the area of interest (grouped)
        area_ind_unique = np.unique(area_ind_list)
        areaoi_ind_exist = np.intersect1d(area_ind_unique, areaoi)
        unit_filter_dict[session_id] = unit_filter_mask


    train_sampler = TrialSampler(
    sampling_intervals=train_sampling_intervals_dict,
    generator=torch.Generator().manual_seed(0),
    shuffle=True,
    )

    val_sampler = TrialSampler(
        sampling_intervals=val_sampling_intervals_dict,
        generator=torch.Generator().manual_seed(0),
        shuffle=True,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=train_sampler,
        batch_size=16,
        collate_fn=collate,         # the collator
        num_workers=4,              # data sample processing (slicing, transforms, tokenization) happens in parallel; this sets the amount of that parallelization
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=val_sampler,
        batch_size=16,
        collate_fn=collate,         # the collator
        num_workers=4,              # data sample processing (slicing, transforms, tokenization) happens in parallel; this sets the amount of that parallelization
    )

    dataloader = {'train': train_loader, 'val': val_loader}
    areaoi_ind = np.arange(len(areaoi))
    trial_type_dict = {'left': 0, 'right':1}

    return dataloader, areaoi_ind, area_ind_list_dict, heldout_area_dict, trial_type_dict, unit_heldout_filter_dict, unit_filter_dict

def main(with_reg, with_consistency):

    print(f"with_reg: {with_reg}")
    print(f"with_consistency: {with_consistency}")

    torch.cuda.empty_cache()

    #%% set arguments
    multi_gpu = False
    consistency = False
    load_previous_model = False

    base_path = '/work/hdd/bdye/jxia4/code/NeuroPaint_torchbrain/'
    example_path = base_path + 'torch_brain/examples/neuropaint/'
    num_train_sessions = len(eids)
    train = True

    mask_mode = 'region' # 'time' or 'region' or 'time_region'

    region_channel_num_encoder = 48 # number of region channels in encoder
    unit_embed_dim = 50
    n_layers = 5

    num_epochs = 1000
    batch_size = 16
    use_wandb = False

    #%%
    kwargs = {
        "model": f"include: {example_path}/configs/model/mae_with_hemisphere_embed_and_diff_dim_per_area_ibl.yaml",
    }

    config = config_from_kwargs(kwargs)
    config = update_config(f"{example_path}/configs/finetune_sessions_trainer.yaml", config)

    config['model']['encoder']['masker']['mask_mode'] = mask_mode
    config['model']['encoder']['stitcher']['n_channels_per_region'] = region_channel_num_encoder
    config['model']['encoder']['stitcher']['unit_embed_dim'] = unit_embed_dim
    config['training']['num_epochs'] = num_epochs
    config['wandb']['use'] = use_wandb
    config['wandb']['project'] = 'torchbrain-neuropaint-ibl'
    
    config['model']['encoder']['transformer']['n_layers'] = n_layers

    meta_data = {}

    # set accelerator
    if multi_gpu:
        print("Using multi-gpu training.")
        from accelerate.utils import DistributedDataParallelKwargs
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(kwargs_handlers=[kwargs]) 

        global_batch_size = batch_size * accelerator.num_processes
        config['optimizer']['lr'] = 1e-3 * global_batch_size / 256

    else:
        accelerator = Accelerator()
        
    print(f"Accelerator device: {accelerator.device}")
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    dataloader, areaoi_ind, area_ind_list_dict, heldout_area_dict, trial_type_dict, unit_heldout_filter_dict, unit_filter_dict = make_loader(base_path, example_path, batch_size)
    train_dataloader = dataloader['train']
    val_dataloader = dataloader['val']

    set_seed(config.seed)  

    meta_data['area_ind_list_list'] = area_ind_list_list
    meta_data['areaoi_ind'] = areaoi_ind
    meta_data['num_sessions'] = len(eids)
    meta_data['eids'] = [eid_idx for eid_idx, eid in enumerate(eids)]

    #load pr_max_dict
    pr_max_dict_path = f"{example_path}/configs/dataset/pr_max_dict_ibl.pkl"
    with open(pr_max_dict_path, 'rb') as f:
        pr_max_dict = pickle.load(f)

    for k, v in pr_max_dict.items():
        pr_max_dict[k] = int(v)

    meta_data['pr_max_dict'] = pr_max_dict

    trial_type_values = list(trial_type_dict.values())
    meta_data['trial_type_values'] = trial_type_values

    #for torch_brain
    meta_data['areaoi'] = [b"PO", b"LP", b"DG", b"CA1", b"VISa", b"VPM", b"APN", b"MRN"]
    meta_data['session_id_strs'] = eids


    config = update_config(config, meta_data) # so that everything is saved in the config file

    train_dataloader = dataloader['train']
    val_dataloader = dataloader['val']    


#############
    #test if the model can run forward pass
    model = MAE_with_region_stitcher(config['model'], **meta_data)
    model.to(accelerator.device)
    for batch in val_dataloader:
        with torch.no_grad():
            outputs = model(batch)
        print('forward pass successful')
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--with_reg', action='store_true', help='whether to use regularization')
    parser.add_argument('--with_consistency', action='store_true', help='whether to use consistency loss')

    args = parser.parse_args()
    main(args.eids, args.with_reg, args.with_consistency)

