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
    psz = cfg.model.patch_size
    l_dim = cfg.model.num_latents
    e_dim = cfg.model.dim
    model_name = "capoyo"
    dataset_name = cfg.dataset[0].selection[0].sortset
    canonical_name = f"sweep/lr:{lr:.2e}/psz:{psz}/latent:{l_dim}/dim:{e_dim}"
    return canonical_name
