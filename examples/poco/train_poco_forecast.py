import logging
import hydra
import lightning as L
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from torch_brain.data import Dataset, collate
from torch_brain.data.sampler import RandomFixedWindowSampler, SequentialFixedWindowSampler
from torch_brain.optim import SparseLamb
from torch_brain.registry import MODALITY_REGISTRY, ModalitySpec
from torch_brain.utils import seed_everything

from torch_brain.transforms import Compose
from torch_brain.transforms.patchify import PatchTokenize  # <-- the file above
from torch_brain.models.poco import POCO                                    # <-- the model above

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("medium")

# ---------------- LightningModule ----------------
class TrainWrapper(L.LightningModule):
    def __init__(self, cfg: DictConfig, model: nn.Module, modality_spec: ModalitySpec):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.modality_spec = modality_spec
        self.mse = nn.MSELoss(reduction="none")
        self.mae = nn.L1Loss(reduction="none")
        self.save_hyperparameters(OmegaConf.to_container(cfg))

    def configure_optimizers(self):
        max_lr = self.cfg.optim.base_lr * self.cfg.batch_size
        special_emb = list(self.model.unit_emb.parameters()) + list(self.model.session_emb.parameters())
        others = [p for n, p in self.model.named_parameters() if ("unit_emb" not in n and "session_emb" not in n)]
        optim = SparseLamb([{"params": special_emb, "sparse": True}, {"params": others}],
                           lr=max_lr, weight_decay=self.cfg.optim.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optim, max_lr=max_lr, total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.cfg.optim.lr_decay_start, anneal_strategy="cos", div_factor=1
        )
        return {"optimizer": optim, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def _masked_metrics(self, preds, targets, mask):
        preds = preds[mask]
        targets = targets[mask]
        weights = weights[mask]
        mse = (self.mse(preds, targets).mean(dim=-1) ).mean()
        mae = (self.mae(preds, targets).mean(dim=-1) ).mean()
        return mse, mae

    def training_step(self, batch, batch_idx):
        out = self.model(**batch["model_inputs"])
        mse, mae = self._masked_metrics(out, batch["target_values"],
                                        batch["model_inputs"]["output_mask"])
        self.log("train/mse", mse, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/mae", mae, prog_bar=False, on_step=True, on_epoch=True)
        return mse

    def validation_step(self, batch, batch_idx):
        out = self.model(**batch["model_inputs"])
        mse, mae = self._masked_metrics(out, batch["target_values"],
                                        batch["model_inputs"]["output_mask"])
        self.log("val/mse", mse, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/mae", mae, prog_bar=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        out = self.model(**batch["model_inputs"])
        mse, mae = self._masked_metrics(out, batch["target_values"],
                                        batch["model_inputs"]["output_mask"])
        self.log("test/mse", mse, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("test/mae", mae, prog_bar=False, on_epoch=True, sync_dist=True)

# ---------------- DataModule (no stitch) ----------------
class DataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.sequence_length = float(cfg.context_window) + float(cfg.forecast_window)

    def setup_dataset_and_link_model(self, model: POCO, readout_spec: ModalitySpec):
        # transforms: patchify first, then model.tokenize LAST (POYO pattern)
        train_tf = Compose([PatchTokenize(key=self.cfg.signal_field, patch_size=self.cfg.patch_size), model.tokenize])
        eval_tf  = Compose([PatchTokenize(key=self.cfg.signal_field, patch_size=self.cfg.patch_size), model.tokenize])

        self.train_dataset = Dataset(root=self.cfg.data_root, config=self.cfg.dataset, split="train", transform=train_tf)
        self.val_dataset   = Dataset(root=self.cfg.data_root, config=self.cfg.dataset, split="valid", transform=eval_tf)
        self.test_dataset  = Dataset(root=self.cfg.data_root, config=self.cfg.dataset, split="test",  transform=eval_tf)

        # vocab init same as your train.py
        model.unit_emb.initialize_vocab(self.train_dataset.get_unit_ids())
        model.session_emb.initialize_vocab(self.train_dataset.get_session_ids())

    def train_dataloader(self):
        sampler = RandomFixedWindowSampler(
            sampling_intervals=self.train_dataset.get_sampling_intervals(),
            window_length=self.sequence_length,
            generator=torch.Generator().manual_seed(self.cfg.seed + 1 if self.cfg.seed is not None else 1234),
        )
        return DataLoader(self.train_dataset, sampler=sampler, collate_fn=collate,
                          batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers,
                          drop_last=True, pin_memory=True,
                          persistent_workers=True if self.cfg.num_workers > 0 else False,
                          prefetch_factor=2 if self.cfg.num_workers > 0 else None)

    def val_dataloader(self):
        sampler = SequentialFixedWindowSampler(
            sampling_intervals=self.val_dataset.get_sampling_intervals(),
            window_length=self.sequence_length,
            step=self.sequence_length,  # non-overlapping eval windows
        )
        return DataLoader(self.val_dataset, sampler=sampler, collate_fn=collate,
                          batch_size=self.cfg.eval_batch_size or self.cfg.batch_size,
                          num_workers=self.cfg.num_workers, drop_last=False, pin_memory=True)

    def test_dataloader(self):
        sampler = SequentialFixedWindowSampler(
            sampling_intervals=self.test_dataset.get_sampling_intervals(),
            window_length=self.sequence_length,
            step=self.sequence_length,
        )
        return DataLoader(self.test_dataset, sampler=sampler, collate_fn=collate,
                          batch_size=self.cfg.eval_batch_size or self.cfg.batch_size,
                          num_workers=self.cfg.num_workers, drop_last=False, pin_memory=True)

# ---------------- Hydra entry ----------------
@hydra.main(version_base="1.3", config_path="./configs", config_name="train_poco_forecast.yaml")
def main(cfg: DictConfig):
    logger.info("POCO forecasting")

    seed_everything(cfg.seed)

    # readout spec (same pattern as your train.py)  :contentReference[oaicite:7]{index=7}
    readout_id = cfg.dataset[0].config.readout.readout_id
    readout_spec = MODALITY_REGISTRY[readout_id]

    # instantiate model via Hydra (readout_spec is injected here)
    model = hydra.utils.instantiate(
        cfg.model,
        readout_spec=readout_spec,
    )

    #TODO: ADD THE OPTIMIZER HERE

    dm = DataModule(cfg)
    dm.setup_dataset_and_link_model(model, readout_spec)

    wrapper = TrainWrapper(cfg, model, readout_spec)
    callbacks = [
        stitch_evaluator,
        ModelSummary(max_depth=2),  # Displays the number of parameters in the model.
        ModelCheckpoint(
            save_last=True,
            monitor="average_val_metric",
            mode="max",
            save_on_train_epoch_end=True,
            every_n_epochs=cfg.eval_epochs,
        ),
        LearningRateMonitor(logging_interval="step"),
        tbrain_callbacks.MemInfo(),
        tbrain_callbacks.EpochTimeLogger(),
        tbrain_callbacks.ModelWeightStatsLogger(),
    ]

    trainer = L.Trainer(
        default_root_dir=cfg.log_dir,
        max_epochs=cfg.epochs,
        check_val_every_n_epoch=cfg.eval_epochs,
        log_every_n_steps=1,
        callbacks=[],
        precision=cfg.precision,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.gpus,
        num_nodes=cfg.nodes,
        num_sanity_val_steps=0,
    )
    trainer.fit(wrapper, dm)
    trainer.test(wrapper, dm)

if __name__ == "__main__":
    main()
