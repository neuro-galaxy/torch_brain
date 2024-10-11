import logging
from collections import defaultdict
from typing import Optional

from tqdm import tqdm
import numpy as np
import pandas as pd
from rich import print as rprint
import torch
import torch.nn as nn
import torch.distributed as dist
import lightning as L
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
        model: nn.Module,
        dataset_config_dict: dict = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ):
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.dataset_config_dict = dataset_config_dict

    def configure_optimizers(self):
        return [self.optimizer], [{"scheduler": self.scheduler, "interval": "step"}]

    def training_step(self, batch, batch_idx):
        output, loss, taskwise_loss = self.model(**batch)

        # Compute the mean and std of the output.
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

        self.log("train_loss", loss, prog_bar=True)
        self.log_dict({f"losses/{k}": v for k, v in taskwise_loss.items()})

        return loss

    def on_validation_epoch_start(self):
        # Create dictionaries to store the prediction and other information for all
        # validation data samples. All dictionaries follow this heirarchy:
        # {
        #   "<session_id 1>": {
        #       "<taskname 1>": [tensor from sample 1, tensor from sample 2, ...],
        #       "<taskname 2>": [tensor from sample 1, tensor from sample 2, ...],
        #   }
        #   "<session_id 2>": {...}
        # }
        self.timestamps = defaultdict(lambda: defaultdict(list))
        self.subtask_index = defaultdict(lambda: defaultdict(list))
        self.ground_truth = defaultdict(lambda: defaultdict(list))
        self.pred = defaultdict(lambda: defaultdict(list))

    def validation_step(self, batch, batch_idx):

        absolute_starts = batch.pop("absolute_start")
        session_ids = batch.pop("session_id")
        output_subtask_index = batch.pop("output_subtask_index")
        batch_format = None
        if "input_mask" in batch:
            batch_format = "padded"
        elif "input_seqlen" in batch:
            batch_format = "chained"
        else:
            raise ValueError("Invalid batch format.")

        # forward pass
        output, loss, taskwise_loss = self.model(**batch)

        # we need to get the timestamps, the ground truth values, the task ids as well
        # as the subtask ids. since the batch is padded and chained, this is a bit tricky
        # tldr: this extracts the ground truth in the same format as the model output
        batch_size = len(output)
        # get gt_output and timestamps to be in the same format as pred_output
        timestamps = [{} for _ in range(batch_size)]
        subtask_index = [{} for _ in range(batch_size)]
        gt_output = [{} for _ in range(batch_size)]

        for taskname, spec in self.model.readout.decoder_specs.items():
            # get the mask of tokens that belong to this task
            taskid = Decoder.from_string(taskname).value
            mask = batch["output_decoder_index"] == taskid

            # there is not a single token for this task, so we skip
            if not mask.any():
                continue

            if batch_format == "padded":
                token_batch = torch.where(mask)[0]
            elif batch_format == "chained":
                token_batch = batch["output_batch_index"][mask]

            batch_i, token_batch = torch.unique(token_batch, return_inverse=True)
            for i in range(len(batch_i)):
                timestamps[batch_i[i]][taskname] = (
                    batch["output_timestamps"][mask][token_batch == i]
                    + absolute_starts[batch_i[i]]
                )
                subtask_index[batch_i[i]][taskname] = output_subtask_index[taskname][
                    (token_batch == i).detach().cpu()
                ]
                gt_output[batch_i[i]][taskname] = batch["output_values"][taskname][
                    token_batch == i
                ]

        # register all the data
        for i in range(batch_size):
            session_id = session_ids[i]

            for taskname, pred_value in output[i].items():
                self.pred[session_id][taskname].append(pred_value.detach().cpu())
                self.ground_truth[session_id][taskname].append(
                    gt_output[i][taskname].detach().cpu()
                )
                self.timestamps[session_id][taskname].append(
                    timestamps[i][taskname].detach().cpu()
                )
                self.subtask_index[session_id][taskname].append(
                    subtask_index[i][taskname].detach().cpu()
                )

    def on_validation_epoch_end(self, prefix="val"):
        # Aggregate all data
        for dict_obj in [
            self.timestamps,
            self.subtask_index,
            self.ground_truth,
            self.pred,
        ]:
            for session_id, task_dict in dict_obj.items():
                for taskname, tensor_list in task_dict.items():
                    dict_obj[session_id][taskname] = torch.cat(tensor_list)

            if self.trainer.world_size > 1:
                dict_obj = all_gather_dict_of_dict_of_tensor(dict_obj)

        metrics = dict()
        for session_id in tqdm(
            self.ground_truth,
            desc=f"Compiling metrics @ Epoch {self.current_epoch}",
            disable=(self.local_rank != 0),
        ):
            for taskname in self.ground_truth[session_id]:
                decoders = self.dataset_config_dict[session_id]["multitask_readout"]

                decoder = None
                for decoder_ in decoders:
                    if decoder_["decoder_id"] == taskname:
                        decoder = decoder_

                assert decoder is not None, f"Decoder not found for {taskname}"
                metrics_spec = decoder["metrics"]
                for metric in metrics_spec:
                    gt = self.ground_truth[session_id][taskname]
                    pred = self.pred[session_id][taskname]
                    timestamps = self.timestamps[session_id][taskname]
                    subtask_index = self.subtask_index[session_id][taskname]

                    metric_subtask = metric.get("subtask", None)
                    if metric_subtask is not None:
                        select_subtask_index = Task.from_string(metric_subtask).value
                        mask = subtask_index == select_subtask_index
                        gt = gt[mask]
                        pred = pred[mask]
                        timestamps = timestamps[mask]

                    # pool
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

                    # Resolve the appropriate loss function.
                    metrics[
                        f"{prefix}_{session_id}_{str(taskname.lower())}_{metric['metric']}"
                    ] = compute_loss_or_metric(
                        metric["metric"], output_type, pred, gt, 1.0
                    ).item()

        # Add average of all metrics
        # TODO: Clean this up so we get average-metric per task-type
        metrics[f"average_{prefix}_metric"] = np.array(list(metrics.values())).mean()

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

        del self.timestamps
        del self.subtask_index
        del self.ground_truth
        del self.pred

    def on_test_epoch_start(self):
        self.on_validation_epoch_start()

    def test_step(self, data, data_idx):
        self.validation_step()

    def on_test_epoch_end(self):
        self.on_validation_epoch_end(prefix="test")
