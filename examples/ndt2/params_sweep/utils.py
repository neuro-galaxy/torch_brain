def get_sweep_run_name(cfg):
    """
    Returns the name of the sweep's run based on the hyperparameters being monitored.

    Args:
        cfg (Config): The configuration object containing the hyperparameters.

    Returns:
        str: The name of the sweep's run.

    Notes:
        This helper function can be modified as per the user's requirements and the hparams that are being monitored.
        name the sweep's run based on the hyperparameters being monitored.
    """
    lr = cfg.optimizer.lr
    mask_ratio = cfg.mask_ratio
    weight_decay = cfg.optimizer.weight_decay
    emb_dim = cfg.model.dim
    patch_size = cfg.patch_size
    canonical_name = f"sweep/lr:{lr:.1e}-emb_dim:{emb_dim}-patch_size:{patch_size[0]}-weight_decay:{weight_decay:.1e}-mask_ratio:{mask_ratio:.1f}"
    return canonical_name
