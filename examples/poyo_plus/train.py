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
            sampling_intervals=self.test_dataset.get_sampling_intervals(),
            window_length=self.sequence_length,
            step=self.sequence_length / 2,
            batch_size=batch_size,
            num_replicas=self.trainer.world_size,
            rank=self.trainer.global_rank,
        )

        test_loader = DataLoader(
            self.test_dataset,
            sampler=test_sampler,
            shuffle=False,
            batch_size=batch_size,
            collate_fn=collate,
            num_workers=0,
        )

        self.log.info(f"Testing on {len(test_sampler)} samples")
        self.test_sequence_index = test_sampler.sequence_index

        return test_loader



class DataModuleForDiver(DataModule):
    """DataModule for DIVER that uses DatasetFromLmdb and produces POYO-style batches.
    
    This class inherits from DataModule but:
    - Takes DIVER's argparse.Namespace instead of Hydra DictConfig
    - Uses DatasetFromLmdb instead of HDF5-based Dataset
    - Uses a standalone tokenizer (not dependent on POYO model)
    
    The dataloader produces batches with keys:
        dict_keys(['model_inputs', 'target_values', 'target_weights', 'session_id', 'absolute_start', 'eval_mask'])
    """
    
    def __init__(self, params, task_config: dict = None):
        """Initialize DataModuleForDiver.
        
        Args:
            params: argparse.Namespace from DIVER containing:
                - lmdb_root: path to LMDB data root
                - batch_size: batch size for training
                - num_workers: number of dataloader workers
                - seed: random seed
            task_config: dict from FINAL_TASK_DICT containing task-specific settings:
                - brainset_id: identifier for the brainset
                - session_id: identifier for the session
                - sampling_rate: sampling rate in Hz
                - sequence_length: length of each sequence in seconds
                - num_targets: number of output classes
                - multitask_readout: list of readout configurations
        """
        # Skip DataModule's __init__ which expects DictConfig
        L.LightningDataModule.__init__(self)
        import pdb; pdb.set_trace()
        self.params = params
        self.task_config = task_config or {}
        self.log = logging.getLogger(__name__)
        
        # Get parameters from task_config (with fallbacks from params or defaults)
        self.sequence_length = self.task_config.get('sequence_length', getattr(params, 'sequence_length', 1.0))
        self.batch_size = getattr(params, 'batch_size', 32)
        self.num_workers = 0  # TODO for debug, CHANGE LATER: getattr(params, 'num_workers', 0)
        self.seed = getattr(params, 'seed', 42)
        self.num_targets = self.task_config.get('num_targets', 9)
        
        # Get POYO-specific config from task_config
        brainset_id = self.task_config.get('brainset_id', 'lmdb_dataset')
        session_id = self.task_config.get('session_id', 'session_001')
        sampling_rate = self.task_config.get('sampling_rate', 500.0)
        multitask_readout = self.task_config.get('multitask_readout', [])
        
        # LMDB path from params
        lmdb_path = getattr(params, 'datasets_dir', '/pscratch/sd/a/ahhyun/data/FACED/FACED_LMDB')
        
        # Create LmdbConfig for DatasetFromLmdb
        from torch_brain.data import DatasetFromLmdb, LmdbConfig
        
        lmdb_cfg = LmdbConfig(
            lmdb_path=lmdb_path,
            brainset_id=brainset_id,
            session_id=session_id,
            sampling_rate=sampling_rate,
            multitask_readout=multitask_readout,
        )
        
        # Create datasets for each split - same lmdb_config, different split parameter
        self.train_dataset = DatasetFromLmdb(
            lmdb_config=lmdb_cfg,
            split='train',
            transform=self._create_tokenizer(),
        )
        
        self.val_dataset = DatasetFromLmdb(
            lmdb_config=lmdb_cfg,
            split='valid',  # Note: might be 'valid' or 'val' depending on LMDB keys
            transform=self._create_tokenizer(),
        )
        
        self.test_dataset = DatasetFromLmdb(
            lmdb_config=lmdb_cfg,
            split='test',
            transform=self._create_tokenizer(),
        )
        
        self.log.info(f"DataModuleForDiver initialized with LMDB path: {lmdb_path}")
    
    def _create_tokenizer(self): #!TODO : patch -> CNN -> backbone.
        """Create a standalone tokenizer that produces POYO-style batch format.
        
        Returns a callable that transforms temporaldata.Data -> POYO batch dict
        """
        sequence_length = self.sequence_length
        
        def tokenize(data): #* TODO : compare with orig. POYO tokenizer + DIVER tokenizer(patchfy?)
            """Transform temporaldata.Data to POYO-style dict format.
            
            Produces:
                dict_keys(['model_inputs', 'target_values', 'target_weights', 'session_id', 'absolute_start', 'eval_mask'])
            """
            import numpy as np
            from torch_brain.data import pad8, track_mask8, chain
            from torch_brain.utils import create_linspace_latent_tokens
            
            start, end = 0, sequence_length
            
            # === Prepare input (EEG data as continuous signal) ===
            # data.eeg has shape (time, channels) 
            eeg_data = data.eeg.values  # (T, N_channels)
            timestamps = data.eeg.timestamps  # (T,)
            unit_ids = data.units.id  # channel IDs
            
            T, N = eeg_data.shape
            
            # Flatten to (T*N,) for input - each timepoint x channel is a token
            input_values = eeg_data.flatten().reshape(-1, 1)  # (T*N, 1)
            
            # Create timestamps for each (time, channel) pair
            input_timestamps = np.repeat(timestamps, N)  # (T*N,)
            
            # Create unit indices - cycle through channels for each timepoint
            input_unit_index = np.tile(np.arange(N), T)  # (T*N,)
            
            # === Prepare latents ===
            latent_index, latent_timestamps = create_linspace_latent_tokens(
                start, end,
                step=0.1,  # 100ms latent step
                num_latents_per_step=16,
            )
            
            # === Prepare outputs ===
            session_index = 0  # Single session for DIVER
            
            # Get label from drifting_gratings (mapped from LMDB label)
            if hasattr(data, 'drifting_gratings') and hasattr(data.drifting_gratings, 'orientation_id'):
                label = data.drifting_gratings.orientation_id
                label_timestamps = data.drifting_gratings.timestamps
            else:
                # Fallback: use middle of sequence
                label = np.array([0])
                label_timestamps = np.array([sequence_length / 2])
            
            output_timestamps = label_timestamps
            output_values = {'label': label}
            output_weights = {'label': np.ones_like(label, dtype=np.float32)}
            output_decoder_index = np.zeros_like(label)  # Single decoder
            
            output_session_index = np.repeat(session_index, len(output_timestamps))
            
            # === Build the POYO-style dict ===
            data_dict = {
                "model_inputs": {
                    # Input sequence (keys/values for encoder)
                    "input_unit_index": pad8(input_unit_index),
                    "input_timestamps": pad8(input_timestamps),
                    "input_values": pad8(input_values),
                    "input_mask": track_mask8(input_unit_index),
                    # Latent sequence
                    "latent_index": latent_index,
                    "latent_timestamps": latent_timestamps,
                    # Output query sequence (queries for decoder)
                    "output_session_index": pad8(output_session_index),
                    "output_timestamps": pad8(output_timestamps),
                    "output_decoder_index": pad8(output_decoder_index),
                },
                # Ground truth targets
                "target_values": chain(output_values, allow_missing_keys=True),
                "target_weights": chain(output_weights, allow_missing_keys=True),
                # Extra data for evaluation
                "session_id": data.session.id,
                "absolute_start": getattr(data, 'absolute_start', 0.0),
                "eval_mask": chain({'label': np.array([True] * len(label))}, allow_missing_keys=True),
            }
            
            return data_dict
        
        return tokenize
    
    def setup_dataset_and_link_model(self, model=None):
        """Override parent - datasets are already set up in __init__, model not needed."""
        self.log.info("DataModuleForDiver: Datasets already initialized, skipping model linking.")
        pass
    
    def _init_model_vocab(self, model=None):
        """Override parent - no model vocab initialization needed."""
        pass
    
    # Note: get_session_ids, get_unit_ids, get_recording_config_dict 
    # are inherited from DataModule (identical implementation)
    
    def get_multitask_readout_registry(self):
        """Return readout registry from task_config's multitask_readout.
        
        Returns:
            dict: Mapping from readout_id to readout config
        """
        multitask_readout = self.task_config.get('multitask_readout', [])
        registry = {}
        for readout_config in multitask_readout:
            readout_id = readout_config.get('readout_id', 'label')
            registry[readout_id] = {
                'type': 'classification',
                'num_classes': readout_config.get('num_classes', self.num_targets),
                'weight': readout_config.get('weight', 1.0),
            }
        # Fallback if no multitask_readout configured
        if not registry:
            registry = {'label': {'type': 'classification', 'num_classes': self.num_targets}}
        return registry
    
    def get_metrics(self):
        """Return metrics for evaluation from task_config's multitask_readout.
        
        Returns:
            dict: Nested dict of {session_id: {readout_id: {metric_name: metric}}}
        """
        from collections import defaultdict
        import torchmetrics
        import hydra
        
        metrics = defaultdict(lambda: defaultdict(dict))
        multitask_readout = self.task_config.get('multitask_readout', [])
        
        for session_id in self.get_session_ids():
            for readout_config in multitask_readout:
                readout_id = readout_config.get('readout_id', 'label')
                metric_configs = readout_config.get('metrics', [])
                
                for metric_cfg in metric_configs:
                    metric_def = metric_cfg.get('metric', {})
                    # Try hydra instantiate, fallback to manual creation
                    try:
                        metric = hydra.utils.instantiate(metric_def)
                    except Exception:
                        # Manual fallback for common metrics
                        target = metric_def.get('_target_', 'torchmetrics.Accuracy')
                        num_classes = metric_def.get('num_classes', self.num_targets)
                        task = metric_def.get('task', 'multiclass')
                        if 'Accuracy' in target:
                            metric = torchmetrics.Accuracy(task=task, num_classes=num_classes)
                        elif 'F1Score' in target:
                            average = metric_def.get('average', 'macro')
                            metric = torchmetrics.F1Score(task=task, num_classes=num_classes, average=average)
                        else:
                            metric = torchmetrics.Accuracy(task=task, num_classes=num_classes)
                    
                    metrics[session_id][readout_id][str(metric)] = metric
        
        # Fallback if no multitask_readout configured
        if not multitask_readout:
            for session_id in self.get_session_ids():
                metrics[session_id]['label']['Accuracy'] = torchmetrics.Accuracy(
                    task='multiclass', num_classes=self.num_targets
                )
        
        return metrics
    
    def train_dataloader(self):
        from torch_brain.data.sampler import RandomFixedWindowSampler
        
        train_sampler = RandomFixedWindowSampler(
            sampling_intervals=self.train_dataset.get_sampling_intervals(),
            window_length=self.sequence_length,
            generator=torch.Generator().manual_seed(self.seed + 1),
        )
        
        train_loader = DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            collate_fn=collate,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )
        
        self.log.info(f"Training on {len(train_sampler)} samples")
        return train_loader
    
    def val_dataloader(self):
        from torch_brain.data.sampler import SequentialFixedWindowSampler # TODO: it's DistributedStitchingFixedWindowSampler for DataModule(POYO). think it needs fix if we are to do ddp?
        
        val_sampler = SequentialFixedWindowSampler(
            sampling_intervals=self.val_dataset.get_sampling_intervals(),
            window_length=self.sequence_length,
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            sampler=val_sampler,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=collate,
            num_workers=0,
            drop_last=False,
        )
        
        self.log.info(f"Validating on {len(val_sampler)} samples")
        return val_loader
    
    def test_dataloader(self):
        from torch_brain.data.sampler import SequentialFixedWindowSampler # TODO: it's DistributedStitchingFixedWindowSampler for DataModule(POYO). think it needs fix if we are to do ddp?
        
        test_sampler = SequentialFixedWindowSampler(
            sampling_intervals=self.test_dataset.get_sampling_intervals(),
            window_length=self.sequence_length,
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            sampler=test_sampler,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=collate,
            num_workers=0,
        )
        
        self.log.info(f"Testing on {len(test_sampler)} samples")
        return test_loader


@hydra.main(version_base="1.3", config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig):
    logger.info("POYO+!")
    # fix random seed, skipped if cfg.seed is None
    seed_everything(cfg.seed)

    # setup loggers
    log = logging.getLogger(__name__)
    wandb_logger = None
    if cfg.wandb.enable:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            save_dir=cfg.log_dir,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            project=cfg.wandb.project,
            log_model=cfg.wandb.log_model,
        )

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
        ModelCheckpoint(
            save_last=True,
            monitor="average_val_metric",
            mode="max",
            save_on_train_epoch_end=True,
            every_n_epochs=cfg.eval_epochs,
        ),
        LearningRateMonitor(
            logging_interval="step"
        ),  # Create a callback to log the learning rate.
        tbrain_callbacks.MemInfo(),
        tbrain_callbacks.EpochTimeLogger(),
        tbrain_callbacks.ModelWeightStatsLogger(),
    ]

    trainer = L.Trainer(
        logger=wandb_logger,
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

    # Train
    trainer.fit(wrapper, data_module, ckpt_path=cfg.ckpt_path)

    # Test
    trainer.test(wrapper, data_module, ckpt_path="best", weights_only=False)


if __name__ == "__main__":
    main()
