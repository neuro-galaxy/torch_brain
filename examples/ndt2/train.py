import logging

import hydra
import lightning as L
import torch
from data_loader_generator import DataLoaderGenerator
from lightning.pytorch.utilities import CombinedLoader
from model import (
    NDT2_MaskManager,
    NDT2_Patchifier,
    NDT2_Predictor,
    NDT2_TransformerDecoder,
    NDT2_TransformerEncoder,
)
from omegaconf import OmegaConf, open_dict
from torch import optim
from torchmetrics import R2Score

# from sklearn.metrics import balanced_accuracy_score
from utils import balanced_accuracy_score, set_callbacks, wandb_set_up

from torch_brain.utils.validation import avg_pool, gt_pool

log = logging.getLogger(__name__)


class TrainWrapper(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.mask_manager = NDT2_MaskManager(cfg.mask_ratio)

        self.patchifier = NDT2_Patchifier(
            dim=cfg.model.dim,
            patch_size=cfg.patch_size,
            max_time_patches=cfg.model.max_time_patches,
            max_space_patches=cfg.model.max_space_patches,
            **cfg.model.patchifier,
        )

        self.encoder = NDT2_TransformerEncoder(
            dim=cfg.model.dim,
            **cfg.model.encoder,
        )

        self.predictor = NDT2_Predictor(
            dim=cfg.model.dim,
            patch_size=cfg.patch_size,
            max_time_patches=cfg.model.max_time_patches,
            max_space_patches=cfg.model.max_space_patches,
            **cfg.model.predictor,
        )

        self.bhv_decoder = NDT2_TransformerDecoder(
            dim=cfg.model.dim,
            max_time_patches=cfg.model.max_time_patches,
            max_space_patches=cfg.model.max_space_patches,
            **cfg.model.bhv_decoder,
        )

        self.mae_loss_fn = torch.nn.PoissonNLLLoss(log_input=True)
        self.bhv_loss_fn = torch.nn.MSELoss()

        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

    def training_step(self, batch, batch_idx):
        mae_loss = 0.0
        if "ssl" in batch:
            mae_loss = self.mae_step(batch["ssl"], batch_idx)

        bhv_loss = 0.0
        if "superv" in batch:
            bhv_loss, _ = self.decoder_step(batch["superv"], batch_idx)

        loss = bhv_loss + mae_loss
        return loss

    def mae_step(self, batch, batch_idx, log=True):
        batch = self.mask_manager(batch)

        # patchify input
        x = self.patchifier(
            x=batch["spike_tokens"].clone(),
            time_idx=batch["time_idx"],
            space_idx=batch["space_idx"],
            session_idx=batch["session_idx"],
            session_token_idx=batch["spike_tokens_seqlen"].cumsum(0) - 1,
            # session token is at the end of each sample's token sequence
        )
        NxT, D = x.shape

        # -- encoder
        mask = batch["mask"]
        encoder_input = x[~mask]
        encoder_input_seqlen = batch["spike_tokens_seqlen"] - batch["mask_seqlen"]
        encoder_out = self.encoder(encoder_input, encoder_input_seqlen)

        # -- predictor
        predictor_input = x.new_empty((NxT, D))
        predictor_input[~mask] = encoder_out
        predictor_out = self.predictor(
            x=predictor_input,
            time_idx=batch["time_idx"],
            space_idx=batch["space_idx"],
            seqlen=batch["spike_tokens_seqlen"],
            mask=mask,
        )

        # -- loss
        targets = batch["spike_tokens"][mask]
        preds = predictor_out[mask]
        # remove padding tokens
        pad_val_mask = targets != self.cfg.pad_val
        targets = targets[pad_val_mask]
        preds = preds[pad_val_mask]
        #
        loss = self.mae_loss_fn(preds, targets.float())

        if log:
            self.log("train_loss", loss, prog_bar=True, on_step=True)

        return loss

    def decoder_step(self, batch, batch_idx, log=True):
        def encode(batch):
            x = self.patchifier(
                x=batch["spike_tokens"].clone(),
                time_idx=batch["time_idx"],
                space_idx=batch["space_idx"],
                session_idx=batch["session_idx"],
                session_token_idx=batch["spike_tokens_seqlen"].cumsum(0) - 1,
                # session token is at the end of each sample's token sequence
            )
            x = self.encoder(x=x, seqlen=batch["spike_tokens_seqlen"])
            return x

        if self.cfg.encoder_finetune:
            x = encode(batch)
        else:
            x = encode(batch).detach()

        # decoder
        decoder_out, loss = self.bhv_decoder(
            x=x,
            spike_seqlen=batch["spike_tokens_seqlen"],
            time_idx=batch["time_idx"],
            space_idx=batch["space_idx"],
            output_time_idx=batch["output_time_idx"],
            output_weights=batch["output_weights"],
            output_seqlen=batch["output_seqlen"],
            targets=batch["output_values"],
        )

        if self.cfg.model.bhv_decoder.loss_type == "regression":
            mask = slice(0, None)
            if self.cfg.subtask_idx is not None:
                mask = batch["output_subtask_idx"] == self.cfg.subtask_idx
            r2 = R2Score(num_outputs=decoder_out.size(-1)).to(decoder_out.device)
            pred = decoder_out[mask]
            target = batch["output_values"][mask]
            if len(pred) > 2:
                r2_score = r2(pred, target)
                if log:
                    self.log("train/r2", r2_score, prog_bar=True)

        self.log("bhv_loss", loss, prog_bar=True)
        return loss, decoder_out

    def configure_optimizers(self):
        cfg = self.cfg

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
        )

        linearLR = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=cfg.optimizer.start_factor,
            total_iters=cfg.optimizer.warmup_steps,
        )
        cosineAnnealingLR = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.optimizer.decay_steps,
            eta_min=cfg.optimizer.lr_min,
        )
        scheduler = optim.lr_scheduler.ChainedScheduler([linearLR, cosineAnnealingLR])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_validation_epoch_start(self):
        self.val_mae_losses = []

        self.val_predictions = []
        self.val_targets = []
        self.val_times = []
        self.val_losses = []

    @torch.inference_mode()
    def validation_step(self, batch, batch_idx):
        if batch.get("ssl", None) is not None:
            mae_loss = self.mae_step(batch["ssl"], batch_idx, log=False)
            self.val_mae_losses.append(torch.tensor([mae_loss]))

        if batch.get("superv", None) is not None:
            bhv_loss, bhv_pred = self.decoder_step(
                batch["superv"], batch_idx, log=False
            )

            mask = slice(0, None)
            if self.cfg.subtask_idx is not None:
                mask = batch["superv"]["output_subtask_idx"] == self.cfg.subtask_idx

            self.val_predictions.append(bhv_pred[mask])
            self.val_targets.append(batch["superv"]["output_values"][mask])
            self.val_times.append(batch["superv"]["output_absolute_time"][mask])
            self.val_losses.append(bhv_loss.view(1))

    def on_validation_epoch_end(self, phase="val"):
        def all_gather_list(data):
            if self.trainer.world_size == 1:
                return data
            x = [x.cpu() for x in data]
            ans = [None for _ in range(self.trainer.world_size)]
            torch.distributed.all_gather_object(ans, x)
            ans = sum(ans, [])
            return ans

        # -- MAE
        if self.cfg.doing_ssl:
            self.val_mae_losses = torch.cat(all_gather_list(self.val_mae_losses))
            self.log("val/mae_loss", self.val_mae_losses.mean(), prog_bar=True)

        # -- Behavior
        if self.cfg.doing_superv:
            self.val_predictions = torch.cat(all_gather_list(self.val_predictions))
            self.val_targets = torch.cat(all_gather_list(self.val_targets))
            self.val_losses = torch.cat(all_gather_list(self.val_losses))
            self.val_times = torch.cat(all_gather_list(self.val_times))

            # -- Behavior
            task_type = self.cfg.model.bhv_decoder.loss_type
            if task_type == "regression":
                pred = avg_pool(self.val_times, self.val_predictions)
                targets = avg_pool(self.val_times, self.val_targets)
                r2_func = R2Score(pred.size(-1)).to(pred.device)
                r2 = r2_func(pred, targets)
                self.log(f"{phase}/r2", r2, prog_bar=True)
            elif task_type == "class":
                pred = avg_pool(self.val_times, self.val_predictions)
                pred = pred.argmax(dim=-1)
                targets = gt_pool(self.val_times, self.val_targets)
                acc = balanced_accuracy_score(targets, pred)
                self.log(f"{phase}/accuracy", acc, prog_bar=True)

            self.val_predictions = self.val_targets = self.val_times = None


def run_training(cfg):
    L.seed_everything(cfg.seed)

    if cfg.fast_dev_run:
        cfg.wandb.enable = False
        cfg.num_workers = 0

    with open_dict(cfg):
        cfg.doing_ssl = not cfg.superv_only
        cfg.doing_superv = not cfg.ssl_only

        # Adjust batch size for multi-gpu
        num_gpus = torch.cuda.device_count()
        cfg.batch_size_per_gpu = cfg.batch_size // num_gpus
        cfg.superv_batch_size = cfg.superv_batch_size or cfg.batch_size
        cfg.superv_batch_size_per_gpu = cfg.superv_batch_size // num_gpus
        log.info(f"Number of GPUs: {num_gpus}")
        log.info(f"Batch size per GPU: {cfg.batch_size_per_gpu}")
        log.info(f"Superv batch size per GPU: {cfg.superv_batch_size_per_gpu}")

    wandb_logger = wandb_set_up(cfg, log)

    # Train wrapper
    train_wrapper = TrainWrapper(cfg)

    # ssl
    ssl_cfg = OmegaConf.to_container(cfg.data_ssl.include)
    ssl_loader_generator = DataLoaderGenerator(cfg, ssl_cfg, train_wrapper)
    ssl_train_loader = ssl_loader_generator("train")
    ssl_val_loader = ssl_loader_generator("valid")

    # superv
    superv_cfg = OmegaConf.to_container(cfg.data_superv.include)
    superv_loader_generator = DataLoaderGenerator(cfg, superv_cfg, train_wrapper, False)
    superv_train_loader = superv_loader_generator("train")
    superv_val_loader = superv_loader_generator("valid")

    # combine loaders
    train_loader_dict, val_loader_dict = {}, {}
    if cfg.doing_ssl:
        train_loader_dict["ssl"] = ssl_train_loader
        val_loader_dict["ssl"] = ssl_val_loader
    if cfg.doing_superv:
        train_loader_dict["superv"] = superv_train_loader
        val_loader_dict["superv"] = superv_val_loader
    train_loader = CombinedLoader(train_loader_dict, mode="max_size_cycle")
    val_loader = CombinedLoader(val_loader_dict, mode="max_size")

    # Set up vocab
    ses_ids = ssl_loader_generator.session_ids
    superv_ses_ids = superv_loader_generator.session_ids
    ses_emb = train_wrapper.patchifier.ses_emb
    ses_emb.initialize_vocab(ses_ids)
    subjet_ids = ssl_loader_generator.subjet_ids
    superv_subjet_ids = superv_loader_generator.subjet_ids
    subj_emb = train_wrapper.patchifier.subj_emb
    subj_emb.initialize_vocab(subjet_ids)

    if cfg.doing_superv and cfg.encoder_finetune:
        ses_emb.extend_vocab(vocab=superv_ses_ids, exist_ok=True)
        subj_emb.extend_vocab(vocab=superv_subjet_ids, exist_ok=True)

    if cfg.superv_only:
        ses_emb.subset_vocab(vocab=superv_ses_ids)
        subj_emb.subset_vocab(vocab=superv_subjet_ids)

    log.info(f"SSL sessions: {len(ses_ids)}")
    log.info(f"Superv sessions: {len(superv_ses_ids)}")
    log.info(f"Vocab size: {len(train_wrapper.patchifier.ses_emb.vocab)}")

    # Callbacks
    callbacks = set_callbacks(cfg)

    # Set up trainer
    trainer = L.Trainer(
        logger=wandb_logger,
        default_root_dir=cfg.log_dir,
        check_val_every_n_epoch=cfg.eval_epochs,
        max_epochs=cfg.epochs,
        log_every_n_steps=cfg.log_every_n_steps,
        callbacks=callbacks,
        accelerator="gpu",
        precision=cfg.precision,
        fast_dev_run=cfg.fast_dev_run,
        num_sanity_val_steps=cfg.num_sanity_val_steps,
        strategy="ddp_find_unused_parameters_true",
    )

    if wandb_logger is not None:
        wandb_logger.watch(train_wrapper, log="all", log_freq=1)

    # Train model
    trainer.fit(train_wrapper, train_loader, val_loader)


@hydra.main(version_base="1.3", config_path="./configs", config_name="train.yaml")
def main(cfg):
    run_training(cfg)


if __name__ == "__main__":
    main()
