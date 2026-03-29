import torch
from .base_scheduler import BaseScheduler
from torch.optim.lr_scheduler import StepLR
from warmup_scheduler import GradualWarmupScheduler


class RampUp(BaseScheduler):
    """
    RampUp scheduler with warmup followed by step decay.

    Ported from PopT-BYD-BTB implementation.
    Uses GradualWarmupScheduler for warmup period, then StepLR for decay.
    """

    def __init__(self, cfg, optim):
        """
        Initialize RampUp scheduler.

        Args:
            cfg: Scheduler configuration dict with:
                - warmup: Fraction of total_steps for warmup (default 0.025)
                - total_steps: Total number of training steps
                - gamma: Decay factor for StepLR (default 0.99)
            optim: PyTorch optimizer
        """
        super(RampUp, self).__init__()
        self.cfg = cfg
        warmup = int(self.cfg.warmup * self.cfg.total_steps)
        step_size = (self.cfg.total_steps - warmup) / 100

        gamma = 0.99
        if "gamma" in self.cfg:
            gamma = self.cfg.gamma

        scheduler_steplr = StepLR(optim, step_size=int(step_size), gamma=gamma)
        scheduler_warmup = GradualWarmupScheduler(
            optim, multiplier=1, total_epoch=warmup, after_scheduler=scheduler_steplr
        )

        # This zero gradient update is needed to avoid a warning message, issue #8.
        optim.zero_grad()
        optim.step()
        self.scheduler = scheduler_warmup

    def step(self, loss=None):
        """
        Step the scheduler.

        Args:
            loss: Optional loss value (not used for RampUp, but kept for compatibility)
        """
        self.scheduler.step()
