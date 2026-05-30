# Hyperparameter Sweep with W&B Sweeps and Hydra
A default hyperparameter sweep config file in `wandb_sweep.yaml` that uses [W&B Sweeps](https://wandb.com/sweeps) in combination with Hydra.

First initialize a sweep with:
```bash
wandb sweep --name <sweep-name> wandb_sweep.yaml
```

Then run the sweep agent with the `<sweep-id>` provided in the above command:
```bash
wandb agent <sweep-id>
```

The above command will spawn a sweep agent in wandb's server that will generate hyperparamters as well as run the training script command according to the provided `wandb_sweep.yaml` file.

The included `train.py` uses the same `train.run_training` module. It overrides the `cfg.name` of each sweep run using `utils.get_sweep_run_name()` to dynamically give an appropriate name to each run.

_Pro-tip_: You can run `CUDA_VISIBLE_DEVICES=X wandb agent <sweep-id>` on parallel terminal sessions to run multiple agents in parallel.

For more information on how to use W&B sweeps along with Hydra, refer [this useful report](https://wandb.ai/adrishd/hydra-example/reports/Configuring-W-B-Projects-with-Hydra--VmlldzoxNTA2MzQw?galleryTag=posts) and [W&B's official guide](https://docs.wandb.ai/guides/integrations/hydra).