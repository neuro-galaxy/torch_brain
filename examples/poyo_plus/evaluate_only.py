import copy
import logging
from collections import defaultdict
from typing import Callable, Dict

import hydra
import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from omegaconf import DictConfig, OmegaConf
from temporaldata import Data
from torch.utils.data import DataLoader

from torch_brain.data import Dataset, collate
from torch_brain.data.sampler import (
    DistributedStitchingFixedWindowSampler,
    RandomFixedWindowSampler,
)
from torch_brain.optim import SparseLamb
from torch_brain.models import POYOPlus
from torch_brain.registry import MODALITY_REGISTRY
from torch_brain.transforms import Compose
from torch_brain.utils import callbacks as tbrain_callbacks
from torch_brain.utils import seed_everything
from torch_brain.utils.stitcher import (
    MultiTaskDecodingStitchEvaluator,
    DataForMultiTaskDecodingStitchEvaluator,
)


# higher speed on machines with tensor cores
torch.set_float32_matmul_precision("medium")

logger = logging.getLogger(__name__)


class TrainWrapper(L.LightningModule):
    def __init__(
        self,
        model: POYOPlus,
        cfg: DictConfig,
    ):
        super().__init__()

        self.model = model
        self.cfg = cfg
        self.save_hyperparameters(OmegaConf.to_container(cfg))

    def configure_optimizers(self):
        max_lr = self.cfg.optim.base_lr * self.cfg.batch_size  # linear scaling rule

        special_emb_params = (
            list(self.model.unit_emb.parameters())
            + list(self.model.session_emb.parameters())
            + list(self.model.readout.parameters())
        )

        remaining_params = [
            p
            for n, p in self.model.named_parameters()
            if "unit_emb" not in n and "session_emb" not in n and "readout" not in n
        ]

        optimizer = SparseLamb(
            [
                {"params": special_emb_params, "sparse": True},
                {"params": remaining_params},
            ],
            lr=max_lr,
            weight_decay=self.cfg.optim.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.cfg.optim.lr_decay_start,
            anneal_strategy="cos",
            div_factor=1,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def training_step(self, batch, batch_idx):

        # forward pass
        output_values = self.model(**batch["model_inputs"], unpack_output=False)

        # compute loss
        target_values = batch["target_values"]
        target_weights = batch["target_weights"]
        loss = torch.tensor(0, device=self.device, dtype=torch.float32)
        taskwise_loss = {}
        for readout_id in output_values.keys():
            output = output_values[readout_id]
            target = target_values[readout_id]

            spec = self.model.readout.readout_specs[readout_id]

            weights = 1.0
            if readout_id in target_weights and target_weights[readout_id] is not None:
                weights = target_weights[readout_id]
                        
            taskwise_loss[readout_id] = spec.loss_fn(output, target, weights)

            # count the number of sequences in the batch that have the current task
            num_sequences_with_current_task = torch.any(
                batch["model_inputs"]["output_decoder_index"]
                == MODALITY_REGISTRY[readout_id].id,
                dim=1,
            ).sum()
            loss = loss + taskwise_loss[readout_id] * num_sequences_with_current_task

        batch_size = batch["model_inputs"]["input_unit_index"].shape[0]
        # TODO change batch_size when POYOPlusEfficient is used
        loss = loss / batch_size

        self.log("train_loss", loss, prog_bar=True)
        self.log_dict({f"losses/{k}": v for k, v in taskwise_loss.items()})

        # Log batch statistics
        # for name in target_values.keys():
        #     preds = torch.cat([pred[name] for pred in output if name in pred])
        #     self.log(f"predictions/mean_{name}", preds.mean())
        #     self.log(f"predictions/std_{name}", preds.std())

        #     targets = target_values[name].float()
        #     self.log(f"targets/mean_{name}", targets.mean())
        #     self.log(f"targets/std_{name}", targets.std())

        unit_index = batch["model_inputs"]["input_unit_index"].float()
        self.log("inputs/mean_unit_index", unit_index.mean())
        self.log("inputs/std_unit_index", unit_index.std())

        return loss

    def validation_step(self, batch, batch_idx):

        # forward pass
        output_values = self.model(**batch["model_inputs"], unpack_output=True)

        # prepare data for evaluator
        # (goes to MultiTaskDecodingStitchEvaluator.on_validation_batch_end)
        data_for_eval = DataForMultiTaskDecodingStitchEvaluator(
            timestamps=batch["model_inputs"]["output_timestamps"],
            preds=output_values,
            targets=batch["target_values"],
            decoder_indices=batch["model_inputs"]["output_decoder_index"],
            eval_masks=batch["eval_mask"],
            session_ids=batch["session_id"],
            absolute_starts=batch["absolute_start"],
        )

        return data_for_eval

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)


class DataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.log = logging.getLogger(__name__)

    def setup_dataset_and_link_model(self, model: POYOPlus):
        r"""Setup Dataset objects, and update a given model's embedding vocabs (session
        and unit_emb)
        """
        self.sequence_length = model.sequence_length

        # prepare transforms
        train_transforms = hydra.utils.instantiate(self.cfg.train_transforms)

        # compose transforms, tokenizer is always the last transform
        train_transform = Compose([*train_transforms, model.tokenize])

        self.train_dataset = Dataset(
            root=self.cfg.data_root,
            config=self.cfg.dataset,
            split="train",
            transform=train_transform,
        )
        self.train_dataset.disable_data_leakage_check()

        self._init_model_vocab(model)

        eval_transforms = hydra.utils.instantiate(self.cfg.eval_transforms)

        # validation and test datasets require a tokenizer that is in eval mode
        self.val_dataset = Dataset(
            root=self.cfg.data_root,
            config=self.cfg.dataset,
            split="valid",
            transform=Compose([*eval_transforms, model.tokenize]),
        )
        self.val_dataset.disable_data_leakage_check()

        self.test_dataset = Dataset(
            root=self.cfg.data_root,
            config=self.cfg.dataset,
            split="test",
            transform=Compose([*eval_transforms, model.tokenize]),
        )
        self.test_dataset.disable_data_leakage_check()

    def _init_model_vocab(self, model: POYOPlus):
        # TODO: Add code for finetuning situation (when model already has a vocab)
        model.unit_emb.initialize_vocab(self.get_unit_ids())
        model.session_emb.initialize_vocab(self.get_session_ids())

    def get_session_ids(self):
        return self.train_dataset.get_session_ids()

    def get_unit_ids(self):
        return self.train_dataset.get_unit_ids()

    def get_recording_config_dict(self):
        return self.train_dataset.get_recording_config_dict()

    def get_multitask_readout_registry(self):
        config_dict = self.train_dataset.get_recording_config_dict()

        custum_readout_registry = {}
        for recording_id in config_dict.keys():
            config = config_dict[recording_id]
            multitask_readout = config["multitask_readout"]

            for readout_config in multitask_readout:
                readout_id = readout_config["readout_id"]
                if readout_id not in MODALITY_REGISTRY:
                    raise ValueError(
                        f"Readout {readout_id} not found in modality registry, please register it "
                        "using torch_brain.register_modality()"
                    )
                custum_readout_registry[readout_id] = MODALITY_REGISTRY[readout_id]
        return custum_readout_registry

    def get_metrics(self):
        dataset_config_dict = self.get_recording_config_dict()
        metrics = defaultdict(lambda: defaultdict(dict))
        # setup the metrics
        for recording_id, recording_config in dataset_config_dict.items():
            for readout_config in recording_config["multitask_readout"]:
                readout_id = readout_config["readout_id"]
                for metric_config in readout_config["metrics"]:
                    metric = hydra.utils.instantiate(metric_config["metric"])
                    metrics[recording_id][readout_id][str(metric)] = metric
        return metrics

    def train_dataloader(self):
        train_sampler = RandomFixedWindowSampler(
            sampling_intervals=self.train_dataset.get_sampling_intervals(),
            window_length=self.sequence_length,
            generator=torch.Generator().manual_seed(self.cfg.seed + 1),
        )

        train_loader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            collate_fn=collate,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True if self.cfg.num_workers > 0 else False,
            prefetch_factor=2 if self.cfg.num_workers > 0 else None,
        )

        self.log.info(f"Training on {len(train_sampler)} samples")
        self.log.info(f"Training on {len(self.train_dataset.get_unit_ids())} units")
        self.log.info(f"Training on {len(self.get_session_ids())} sessions")

        return train_loader

    def val_dataloader(self):
        batch_size = self.cfg.eval_batch_size or self.cfg.batch_size

        val_sampler = DistributedStitchingFixedWindowSampler(
            sampling_intervals=self.val_dataset.get_sampling_intervals(),
            window_length=self.sequence_length,
            step=self.sequence_length / 2,
            batch_size=batch_size,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
        )

        val_loader = DataLoader(
            self.val_dataset,
            sampler=val_sampler,
            shuffle=False,
            batch_size=batch_size,
            collate_fn=collate,
            num_workers=0,
            drop_last=False,
        )

        self.log.info(f"Expecting {len(val_sampler)} validation steps")
        self.val_sequence_index = val_sampler.sequence_index

        return val_loader

    def test_dataloader(self):
        batch_size = self.cfg.eval_batch_size or self.cfg.batch_size

        test_sampler = DistributedStitchingFixedWindowSampler(
            #todo - maybe replace it with validation here?
            # todo - fix it somehow
            # sampling_intervals=self.test_dataset.get_sampling_intervals(),
            sampling_intervals=self.val_dataset.get_sampling_intervals(),
            window_length=self.sequence_length,
            step=self.sequence_length / 2,
            batch_size=batch_size,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
        )

        test_loader = DataLoader(
            #todo - fix this somehow
            self.val_dataset,
            # self.test_dataset,
            sampler=test_sampler,
            shuffle=False,
            batch_size=batch_size,
            collate_fn=collate,
            num_workers=4,
            drop_last=False,
        )

        self.log.info(f"Testing on {len(test_sampler)} samples")
        self.test_sequence_index = test_sampler.sequence_index

        return test_loader


@hydra.main(version_base="1.3", config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig):
    logger.info("POYO+!")
    # fix random seed, skipped if cfg.seed is None
    seed_everything(cfg.seed)

    # setup loggers
    log = logging.getLogger(__name__)
    

    # make model and datamodule
    # TODO: resolve the readout_id from dataset, only build readouts needed
    model = hydra.utils.instantiate(cfg.model, readout_specs=MODALITY_REGISTRY)
    data_module = DataModule(cfg)
    data_module.setup_dataset_and_link_model(model)

    # Lightning train wrapper
    wrapper = TrainWrapper(cfg=cfg, model=model)

    evaluator = MultiTaskDecodingStitchEvaluator(metrics=data_module.get_metrics())

    callbacks = [
        evaluator,
        ModelSummary(max_depth=2),  # Displays the number of parameters in the model.
        tbrain_callbacks.MemInfo(),
        tbrain_callbacks.EpochTimeLogger(),
        tbrain_callbacks.ModelWeightStatsLogger(),
    ]

    trainer = L.Trainer(
        logger=None,
        default_root_dir=cfg.log_dir,
        check_val_every_n_epoch=cfg.eval_epochs,
        max_epochs=cfg.epochs,
        log_every_n_steps=1,
        strategy=(
            "ddp_find_unused_parameters_true" if torch.cuda.is_available() else "auto"
        ),
        callbacks=callbacks,
        precision=cfg.precision,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.gpus,
        num_nodes=cfg.nodes,
        limit_val_batches=None,  # Ensure no limit on validation batches
        num_sanity_val_steps=-1 if cfg.sanity_check_validation else 0,
    )

    log.info(
        f"Local rank/node rank/world size/num nodes: "
        f"{trainer.local_rank}/{trainer.node_rank}/{trainer.world_size}/{trainer.num_nodes}"
    )


    # Test
    checkpoint_pre_path = '/scratch-grete/projects/nim00012/adapted_torch_brain/torch_brain/examples/poyo_plus/checkpoints/'
    trainer.test(wrapper, data_module, 
                #  ckpt_path=f"{checkpoint_pre_path}capoyo_8_mice_bs_256_ep_300_depth_12_4_workers/epoch=259-step=28080.ckpt",
                #  ckpt_path=f"{checkpoint_pre_path}capoyo_8_mice_bs_256_ep_300_depth_18/epoch=39-step=4320.ckpt",
                #  ckpt_path=f"{checkpoint_pre_path}capoyo_8_mice_bs_128_ep_300_dim_256_dim_head_128/epoch=259-step=56160.ckpt",
                #  ckpt_path=f"{checkpoint_pre_path}capoyo_8_mice_bs_256_ep_300_dim_256_dim_head_64/epoch=74-step=8100.ckpt",
                #  ckpt_path=f"{checkpoint_pre_path}capoyo_8_mice_bs_256_ep_300_dim_64_dim_head_32/epoch=249-step=27000.ckpt",
                #  ckpt_path=f"{checkpoint_pre_path}capoyo_8_mice_bs_256_ep_300_16_latents/epoch=19-step=2160.ckpt",
                #  ckpt_path=f"{checkpoint_pre_path}capoyo_8_mice_bs_256_ep_300_8_workers/epoch=244-step=26460.ckpt",
                #  ckpt_path=f"{checkpoint_pre_path}capoyo_8_mice_fixed_checkpoints_bs_128_ep_300_1_sec_depth_6_ca_heads_2/epoch=130-step=28296.ckpt",
                #  ckpt_path=f"{checkpoint_pre_path}capoyo_8_mice_fixed_checkpoints_bs_128_ep_300_2_sec/epoch=178-step=18616.ckpt",
                #  ckpt_path=f"{checkpoint_pre_path}capoyo_8_mice_fixed_checkpoints_bs_128_ep_300", # still training!,
                # ckpt_path=f"{checkpoint_pre_path}poyo_single_sesssion_conf_bs_256_ep_300_seq_len_2_max_unit_500_min_100_mode_250_dim_128/epoch=204-step=10660.ckpt",
                # ckpt_path=f"{checkpoint_pre_path}poyo_single_sesssion_conf_bs_256_ep_300_seq_len_2_max_unit_500_min_100_mode_250/epoch=179-step=9360.ckpt",
                # ckpt_path=f"{checkpoint_pre_path}poyo_single_sesssion_conf_bs_256_ep_300_seq_len_2/epoch=234-step=12220.ckpt",
                # ckpt_path=f"{checkpoint_pre_path}poyo_single_sesssion_conf_bs_256_ep_300_seq_len_1/epoch=109-step=11880.ckpt",
                # ckpt_path=f"{checkpoint_pre_path}poyo_single_sesssion_conf_bs_256_ep_300_seq_len_2_max_unit_1000/epoch=269-step=14040.ckpt",
                ckpt_path=f"{checkpoint_pre_path}poyo_single_sesssion_conf_bs_256_ep_300_seq_len_2_max_unit_500_min_10_mode_50_with_new_norm/epoch=39-step=2080.ckpt",
                # ckpt_path=f"{checkpoint_pre_path}",
                 )


if __name__ == "__main__":
    main()
