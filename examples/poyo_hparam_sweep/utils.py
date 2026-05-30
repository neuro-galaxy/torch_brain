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
    lr = cfg.base_lr
    model_name = "poyo-plus"
    dataset_name = cfg.dataset[0].selection[0].dandiset
    canonical_name = f"sweep/lr:{lr:.2e}/{dataset_name}/{model_name}"
    return canonical_name
