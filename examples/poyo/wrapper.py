import time
import subprocess
import logging
from typing import Optional

import torch
import torch.nn as nn
from lightning import LightningModule
import lightning.pytorch.loggers as pl_loggers
from lightning.pytorch.callbacks import Callback

from torch_brain.utils.validation_wrapper import CustomValidator

log = logging.getLogger(__name__)


class POYOTrainWrapper(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["optimizer", "scheduler"])

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def configure_optimizers(self):
        return [self.optimizer], [{"scheduler": self.scheduler, "interval": "step"}]

    def training_step(self, data, data_idx):
        output, loss, taskwise_loss = self.model(**data)

        # Compute the mean and std of the output.
        for name in data["output_values"].keys():
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
                data["output_values"][name].to(torch.float).mean(),
                prog_bar=False,
            )
            self.log(
                f"targets/std_{name}",
                data["output_values"][name].to(torch.float).std(),
                prog_bar=False,
            )

        if "unit_index" in data:
            s = data["unit_index"].to(torch.float)
            self.log("inputs/mean_unit_index", s.mean(), prog_bar=False)
            self.log("inputs/std_unit_index", s.std(), prog_bar=False)

        self.log("train_loss", loss, prog_bar=True)
        self.log_dict({f"losses/{k}": v for k, v in taskwise_loss.items()})

        return loss

    def validation_step(self, data, data_idx):
        # Necessary to trick PyTorch Lightning into running the custom validator.
        pass

    def test_step(self, data, data_idx):
        # Necessary to trick PyTorch Lightning into running the custom validator.
        pass
