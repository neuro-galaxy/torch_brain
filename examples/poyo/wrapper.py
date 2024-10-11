import logging
from collections import defaultdict
from typing import Optional
from omegaconf import DictConfig, OmegaConf

from tqdm import tqdm
import numpy as np
import pandas as pd
from rich import print as rprint
import torch
import torch.nn as nn
import torch.distributed as dist
import lightning as L
from torch_optimizer import Lamb
import wandb

from brainsets.taxonomy import Decoder, OutputType, Task
from torch_brain.nn import compute_loss_or_metric
from torch_brain.utils.validation import (
    all_gather_dict_of_dict_of_tensor,
    avg_pool,
    gt_pool,
)

log = logging.getLogger(__name__)


class POYOTrainWrapper(L.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        model: nn.Module,
        dataset_config_dict: dict = None,
    ):
        super().__init__()

        self.cfg = cfg
        self.model = model
        self.dataset_config_dict = dataset_config_dict
        self.save_hyperparameters(OmegaConf.to_container(cfg))

    def configure_optimizers(self):
        max_lr = self.cfg.base_lr * self.cfg.batch_size  # linear scaling rule

        optimizer = Lamb(
            self.model.parameters(),
            lr=max_lr,
            weight_decay=self.cfg.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=self.cfg.epochs,
            steps_per_epoch=self.cfg.steps_per_epoch,
            pct_start=self.cfg.pct_start,
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
        output, loss, taskwise_loss = self.model(**batch)

        self.log("train_loss", loss, prog_bar=True)
        self.log_dict({f"losses/{k}": v for k, v in taskwise_loss.items()})

        # Log batch statistics
        for name in batch["output_values"].keys():
            output_predictions = torch.cat(
                [pred[name] for pred in output if name in pred], dim=0
            )
            self.log(
                f"predictions/mean_{name}", output_predictions.mean(), prog_bar=False
            )
            self.log(
                f"predictions/std_{name}", output_predictions.std(), prog_bar=False
            )
            self.log(
                f"targets/mean_{name}",
                batch["output_values"][name].to(torch.float).mean(),
                prog_bar=False,
            )
            self.log(
                f"targets/std_{name}",
                batch["output_values"][name].to(torch.float).std(),
                prog_bar=False,
            )

        if "unit_index" in batch:
            s = batch["unit_index"].to(torch.float)
            self.log("inputs/mean_unit_index", s.mean(), prog_bar=False)
            self.log("inputs/std_unit_index", s.std(), prog_bar=False)

        return loss

    def on_validation_epoch_start(self):
        # Create dictionaries to store the prediction and other information for all
        # validation data samples. All data dictionaries follow this heirarchy:
        # {
        #   "<session_id 1>": {
        #       "<taskname 1>": [tensor from sample 1, tensor from sample 2, ...],
        #       "<taskname 2>": [tensor from sample 1, tensor from sample 2, ...],
        #   }
        #   "<session_id 2>": {...}
        # }
        self.val_data = {
            "timestamps": defaultdict(lambda: defaultdict(list)),
            "subtask_index": defaultdict(lambda: defaultdict(list)),
            "ground_truth": defaultdict(lambda: defaultdict(list)),
            "pred": defaultdict(lambda: defaultdict(list)),
        }

    def validation_step(self, batch, batch_idx):

        absolute_starts = batch.pop("absolute_start")
        session_ids = batch.pop("session_id")
        output_subtask_index = batch.pop("output_subtask_index")

        # forward pass
        output, loss, taskwise_loss = self.model(**batch)

        # register all the data
        for task_index in torch.unique(batch["output_decoder_index"]):
            mask = batch["output_decoder_index"] == task_index
            taskname = Decoder(task_index.item()).name

            # token_sample_idx is the sample index that each token in the batch belongs to
            if "input_mask" in batch:  # => padded batch format
                token_sample_idx = torch.where(mask)[0]
            elif "input_seqlen" in batch:  # => chained batch format
                token_sample_idx = batch["output_batch_index"][mask]
            else:
                raise ValueError("Invalid batch format.")

            for i in torch.unique(token_sample_idx):
                session_id = session_ids[i]

                pred = output[i][taskname]
                self.val_data["pred"][session_id][taskname].append(pred.detach().cpu())

                timestamps = (
                    batch["output_timestamps"][mask][token_sample_idx == i]
                    + absolute_starts[i]
                )
                self.val_data["timestamps"][session_id][taskname].append(
                    timestamps.detach().cpu()
                )

                gt = batch["output_values"][taskname][token_sample_idx == i]
                self.val_data["ground_truth"][session_id][taskname].append(
                    gt.detach().cpu()
                )

                subtask_idx = output_subtask_index[taskname][token_sample_idx == i]
                self.val_data["subtask_index"][session_id][taskname].append(
                    subtask_idx.detach().cpu()
                )

    def on_validation_epoch_end(self, prefix="val"):
        # Aggregate all data
        for key, data_dict in self.val_data.items():
            # concatenate tensors
            for session_id, task_dict in data_dict.items():
                data_dict[session_id] = task_dict = dict(task_dict)
                for taskname, tensor_list in task_dict.items():
                    data_dict[session_id][taskname] = torch.cat(tensor_list)

            # all-gather across all processes
            if self.trainer.world_size > 1:
                self.val_data[key] = all_gather_dict_of_dict_of_tensor(dict(data_dict))

        # Compute metrics
        metrics = dict()
        session_ids = list(self.val_data["ground_truth"].keys())
        for session_id in tqdm(
            session_ids,
            desc=f"Compiling metrics @ Epoch {self.current_epoch}",
            disable=(self.local_rank != 0),
        ):
            decoders = self.dataset_config_dict[session_id]["multitask_readout"]

            for taskname in self.val_data["ground_truth"][session_id]:

                # Find the decoder and metrics for this task
                decoder = None
                for decoder_ in decoders:
                    if decoder_["decoder_id"] == taskname:
                        decoder = decoder_
                assert decoder is not None, f"Decoder not found for {taskname}"
                metrics_spec = decoder["metrics"]

                # Compute metrics for the task
                for metric in metrics_spec:
                    gt = self.val_data["ground_truth"][session_id][taskname]
                    pred = self.val_data["pred"][session_id][taskname]
                    timestamps = self.val_data["timestamps"][session_id][taskname]
                    subtask_index = self.val_data["subtask_index"][session_id][taskname]

                    metric_subtask = metric.get("subtask", None)
                    if metric_subtask is not None:
                        select_subtask_index = Task.from_string(metric_subtask).value
                        mask = subtask_index == select_subtask_index
                        gt = gt[mask]
                        pred = pred[mask]
                        timestamps = timestamps[mask]

                    # Pool data wherever timestamps overlap
                    output_type = self.model.readout.decoder_specs[taskname].type
                    if output_type == OutputType.CONTINUOUS:
                        pred = avg_pool(timestamps, pred)
                        gt = avg_pool(timestamps, gt)
                    elif output_type in [
                        OutputType.BINARY,
                        OutputType.MULTINOMIAL,
                        OutputType.MULTILABEL,
                    ]:
                        gt = gt_pool(timestamps, gt)
                        pred = avg_pool(timestamps, pred)
                    else:
                        raise NotImplementedError

                    # Resolve the appropriate loss function.
                    metrics[
                        f"{prefix}_{session_id}_{str(taskname.lower())}_{metric['metric']}"
                    ] = compute_loss_or_metric(
                        metric["metric"], output_type, pred, gt, 1.0
                    ).item()

        # Add average of all metrics
        metrics[f"average_{prefix}_metric"] = np.array(list(metrics.values())).mean()

        # Logging
        self.log_dict(metrics)
        logging.info(f"Logged {len(metrics)} {prefix} metrics.")

        metrics_data = []
        for metric_name, metric_value in metrics.items():
            metrics_data.append({"metric": metric_name, "value": metric_value})

        metrics_df = pd.DataFrame(metrics_data)
        if self.local_rank == 0:
            for logger in self.loggers:
                if isinstance(logger, L.pytorch.loggers.TensorBoardLogger):
                    logger.experiment.add_text(
                        f"{prefix}_metrics", metrics_df.to_markdown()
                    )
                if isinstance(logger, L.pytorch.loggers.WandbLogger):
                    logger.experiment.log(
                        {f"{prefix}_metrics": wandb.Table(dataframe=metrics_df)}
                    )

        rprint(metrics_df)

        # Reset the validation data
        self.val_data = None

    def on_test_epoch_start(self):
        self.on_validation_epoch_start()

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end(prefix="test")
