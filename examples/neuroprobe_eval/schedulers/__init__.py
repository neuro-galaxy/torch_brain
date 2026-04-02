from .ramp_up import RampUp

__all__ = ["build_scheduler", "RampUp"]


def build_scheduler(cfg, optim):
    """
    Build a learning rate scheduler from configuration.

    Args:
        cfg: Scheduler configuration dict with 'name' key
        optim: PyTorch optimizer

    Returns:
        Scheduler instance or None if name is 'none' or not specified
    """
    if cfg is None:
        return None

    name = cfg.get("name", None)
    if name is None or name == "none":
        return None

    if name == "ramp_up":
        return RampUp(cfg, optim)
    else:
        raise ValueError(
            f"Scheduler name '{name}' not found. Supported: 'ramp_up', 'none'"
        )
