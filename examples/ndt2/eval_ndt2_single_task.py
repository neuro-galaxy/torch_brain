import argparse
import logging
import os

import lightning as L
import numpy as np
import torch
from hydra import compose, initialize
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.loggers import WandbLogger
from model import BhvrDecoder, ContextManager, Encoder, NDT2Model, SpikesPatchifier
from omegaconf import OmegaConf, open_dict
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from train import TrainWrapper
from transforms import Ndt2Tokenizer

from torch_brain.data import Dataset, collate
from torch_brain.data.sampler import SequentialFixedWindowSampler

logging.basicConfig(level=logging.INFO)


def move_to_gpu(d, device):
    for k, v in d.items():
        if isinstance(v, dict):
            move_to_gpu(v, device)
        elif isinstance(v, torch.Tensor):
            d[k] = v.to(device)


def _one_hot(arr, T):
    uni = np.sort(np.unique(arr))
    ret = np.zeros((len(arr), T, len(uni)))
    for i, _uni in enumerate(uni):
        ret[:, :, i] = arr == _uni
    return ret


def get_ctx_vocab(self, ctx_keys):
    return {k: getattr(self.dataset, f"get_{k}_ids")() for k in ctx_keys}


# -------
# SET UP
# -------
ap = argparse.ArgumentParser()
ap.add_argument("--eid", type=str, default="03d9a098-07bf-4765-88b7-85f8d8f620cc")
ap.add_argument("--behavior", type=str, default="whisker")
ap.add_argument("--model_path", type=str, default="./logs/lightning_logs/")
ap.add_argument(
    "--ckpt_name", type=str, default="/home/hice1/aandre8/scratch/alex/checkpoints/"
)
ap.add_argument(
    "--save_file",
    type=str,
    default="/home/hice1/aandre8/scratch/alex/results/results.txt",
)
ap.add_argument("--config_path", type=str, default="./ibl_configs")
ap.add_argument("--config_name_prefix", type=str, default="probe_")
ap.add_argument("--ckpt_postfix", type=str, default="")
args = ap.parse_args()

save_file = args.save_file
config_path = args.config_path
model_path = args.model_path
config_name_prefix = args.config_name_prefix
ckpt_postfix = args.ckpt_postfix
ckpt_name = args.ckpt_name
eid = args.eid
bhvr = args.behavior
config_name = f"{config_name_prefix}{bhvr}"

logging.info(f"Evaluating session: {eid}")

params = {
    "interval_len": 2,
    "binsize": 0.02,
    "single_region": False,
    "align_time": "stimOn_times",
    "time_window": (-0.5, 1.5),
    "fr_thresh": 0.5,
}

with initialize(version_base="1.3", config_path=config_path):
    cfg = compose(config_name=config_name)


# ---------
# LOAD DATA
# ---------
dataset_cfg = OmegaConf.create(
    [{"selection": [{"brainset": "ibl", "sessions": [eid]}]}]
)

dataset = Dataset(root=cfg.data_root, split="test", config=dataset_cfg)

# ----------
# LOAD MODEL
# ----------

L.seed_everything(cfg.seed)
# seed_everything(cfg.seed)


with open_dict(cfg):
    # Adjust batch size for multi-gpu
    num_gpus = torch.cuda.device_count()
    cfg.batch_size_per_gpu = cfg.batch_size // num_gpus
    cfg.superv_batch_size = cfg.superv_batch_size or cfg.batch_size
    cfg.superv_batch_size_per_gpu = cfg.superv_batch_size // num_gpus
#     log.info(f"Number of GPUs: {num_gpus}")
#     log.info(f"Batch size per GPU: {cfg.batch_size_per_gpu}")
#     log.info(f"Superv batch size per GPU: {cfg.superv_batch_size_per_gpu}")

# wandb_logger = set_wandb(cfg, log)
wandb_logger = None

dim = cfg.model.dim

# Mask manager (for MAE SSL)
mae_mask_manager = None

# Context manager
ctx_manager = ContextManager(dim, cfg.ctx_keys)

# Spikes patchifier
spikes_patchifier = SpikesPatchifier(dim, cfg.patch_size)

# Encoder
encoder = Encoder(
    dim=dim,
    max_time_patches=cfg.model.max_time_patches,
    max_space_patches=cfg.model.max_space_patches,
    **cfg.model.encoder,
)

# Decoder
decoder = BhvrDecoder(
    dim=dim,
    max_time_patches=cfg.model.max_time_patches,
    max_space_patches=cfg.model.max_space_patches,
    bin_time=cfg.bin_time,
    **cfg.model.bhv_decoder,
)

# Model wrap everithing
model = NDT2Model(mae_mask_manager, ctx_manager, spikes_patchifier, encoder, decoder)


bhvr_dim = cfg.model.bhv_decoder["behavior_dim"]

# Load from checkpoint
# TODO update this

path = f"{ckpt_name}probe_{bhvr}{ckpt_postfix}-{eid}.ckpt"

ckpt = torch.load(path)
model.ctx_manager.load_state_dict(ckpt["context_manager_state_dict"])
model.spikes_patchifier.load_state_dict(ckpt["spikes_patchifier_state_dict"])
model.encoder.load_state_dict(ckpt["encoder_state_dict"])
model.decoder.load_state_dict(ckpt["decoder_state_dict"])

ctx_tokenizer = ctx_manager.get_ctx_tokenizer()
tokenizer = Ndt2Tokenizer(
    ctx_time=cfg.ctx_time,
    bin_time=cfg.bin_time,
    patch_size=cfg.patch_size,
    pad_val=cfg.pad_val,
    ctx_tokenizer=ctx_tokenizer,
    unsorted=cfg.unsorted,
    is_ssl=cfg.is_ssl,
    bhvr_key=cfg.get("bhvr_key"),
    bhvr_dim=bhvr_dim,
    ibl_binning=cfg.get("ibl_binning", False),
)

test_dataset = Dataset(root=cfg.data_root, config=dataset_cfg, split="test")
inter = test_dataset.get_sampling_intervals()
eval_sampler = SequentialFixedWindowSampler(
    sampling_intervals=inter, window_length=cfg.ctx_time, drop_short=True
)

dataset = Dataset(
    root=cfg.data_root, config=dataset_cfg, split=None, transform=tokenizer
)
eval_loader = DataLoader(
    dataset=dataset,
    batch_size=cfg.batch_size,
    sampler=eval_sampler,
    collate_fn=collate,
    num_workers=1,
)


# Set up trainer
trainer = L.Trainer(
    logger=wandb_logger,
    default_root_dir=cfg.log_dir,
    check_val_every_n_epoch=cfg.eval_epochs,
    max_epochs=cfg.epochs,
    log_every_n_steps=1,
    callbacks=None,
    accelerator="gpu",
    precision=cfg.precision,
    num_sanity_val_steps=0,
    strategy="single_device",
)

# Train wrapper
# TODO not confident here
train_wrapper = TrainWrapper(cfg, model)


# ---------
# INFERENCE
# ---------
session_timestamp = {}
session_subtask_index = {}
session_gt_output = {}
session_pred_output = {}

pred_outputs = None
gt_outputs = None

for batch in tqdm(eval_loader):
    # Autocast is explicitly set based on the precision specified by the user.
    # By default, torch autocasts to float16 for 16-bit inference.
    # This behavior is overridden to use bfloat16 if specified in trainer.precision.
    # If 16-bit inference is not enabled, autocast is not used.
    def get_autocast_args(trainer):
        if trainer.precision.startswith("bf16"):
            return torch.bfloat16, True
        elif trainer.precision.startswith("16"):
            return torch.float16, True
        else:
            return None, False

    dtype, enabled = get_autocast_args(trainer)

    # forward pass
    with torch.cuda.amp.autocast(enabled=enabled, dtype=dtype):
        with torch.inference_mode():
            decoder_out = model(batch, "bhv")

    pred_output = decoder_out["pred"].detach().cpu()

    # TODO maybe gt could be obtained in another way
    # e.g.: session_id = f"ibl/{eid}"
    # choice = test_dataset._data_objects[session_id].choice.choice
    gt_output = batch["bhvr"].detach().cpu()
    if pred_outputs is None:
        pred_outputs = pred_output
        gt_outputs = gt_output
    else:
        pred_outputs = torch.cat((pred_outputs, pred_output), dim=0)
        gt_outputs = torch.cat((gt_outputs, gt_output), dim=0)


data = {"gt": gt_outputs, "pred": pred_outputs}

np.save(f"/home/hice1/aandre8/scratch/alex/results/{eid}_{bhvr}.npy", data)

# -----
# EVAL
# -----
results = {bhvr: {}}


if bhvr == "choice" or bhvr == "block":
    pred = pred_outputs.argmax(-1)
    target = gt_outputs.argmax(-1)
    results[bhvr]["accuracy"] = accuracy_score(target, pred)
    results[bhvr]["balanced_accuracy"] = balanced_accuracy_score(target, pred)

from ibl.eval_utils import compute_R2_main, viz_single_cell
from sklearn.metrics import r2_score

# Eval continuous behaviors
if bhvr == "wheel" or bhvr == "whisker":
    results[bhvr]["r2"] = r2_score(gt_outputs.flatten(), pred_outputs.flatten())

    # would be useful to have the viz
    # T = 100
    # gt = np.array(pred_outputs).reshape(-1, T, 1)
    # pred = np.array(gt_outputs).reshape(-1, T, 1)

    # test_dataset.disable_data_leakage_check()
    # data = test_dataset.get_recording_data(f"ibl/{eid}")
    # choice = data.choice.choice
    # reward = data.reward.reward
    # block = data.block.block

    # X = np.concatenate(
    #     (
    #         _one_hot(choice.reshape(-1, 1), T),
    #         _one_hot(reward.reshape(-1, 1), T),
    #         _one_hot(block.reshape(-1, 1), T),
    #     ),
    #     axis=2,
    # )

    # var_name2idx = {"block": [2], "choice": [0], "reward": [1]}
    # var_value2label = {
    #     "block": {
    #         (0,): "p(left)=0.2",
    #         (1,): "p(left)=0.5",
    #         (2,): "p(left)=0.8",
    #     },
    #     "choice": {(0,): "right", (1,): "left"},
    #     "reward": {
    #         (0,): "no reward",
    #         (1,): "reward",
    #     },
    # }
    # var_tasklist = ["block", "choice", "reward"]
    # var_behlist = []

    # if args.behavior == "wheel":
    #     avail_beh = "wheel-speed"
    # elif args.behavior == "whisker":
    #     avail_beh = "whisker-motion-energy"

    # y = gt[:, :, [0]]
    # y_pred = pred[:, :, [0]]
    # y = y.reshape(-1, T)
    # y_pred = y_pred.reshape(-1, T)
    # _, _ = viz_single_cell(
    #     X,
    #     y,
    #     y_pred,
    #     var_name2idx,
    #     var_tasklist,
    #     var_value2label,
    #     var_behlist,
    #     subtract_psth="task",
    #     aligned_tbins=[],
    #     neuron_idx=avail_beh,
    #     neuron_region="",
    #     method="poyo",
    #     save_path="/home/hice1/aandre8/scratch/alex/results/",
    #     save_plot=True,
    # )

print(results)


with open(save_file, "a") as f:
    if bhvr == "choice" or bhvr == "block":
        f.write(f"{eid}, {results[bhvr]['accuracy']}\n")
    else:
        f.write(f"{eid}, {results[bhvr]['r2']}\n")
