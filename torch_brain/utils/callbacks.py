import time
import subprocess
import logging

try:
    import lightning as L

    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False

    class DummyCallback:
        pass

    L = type("L", (), {"Callback": DummyCallback})


def _check_lightning_available(cls):
    """Raise an error if Lightning is not available."""
    if not LIGHTNING_AVAILABLE:
        raise ImportError(
            f"Lightning is not installed. Please install it with "
            f"`pip install lightning~=2.3` to use {cls.__name__}."
        )


class EpochTimeLogger(L.Callback):
    r"""Lightning callback to log the time taken for each epoch.
    Args:
        enable (bool, optional): Whether to enable the callback. Defaults to `True`.
    """

    def __init__(self, enable=True):
        _check_lightning_available(self.__class__)
        self.enable = enable

    def on_train_epoch_start(self, trainer, pl_module):
        if self.enable:
            self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        if self.enable:
            epoch_time = time.time() - self.start_time
            pl_module.log("epoch_time", epoch_time, sync_dist=True)


class ModelWeightStatsLogger(L.Callback):
    r"""Lightning callback to log the mean and standard deviation of the weights and
    gradients of the model.
    Args:
        enable (bool, optional): Whether to enable the callback. Defaults to `True`.
        grads (bool, optional): Whether to log the statistics of gradients.
            Defaults to `True`.
        module_name (str, optional): The name of the :class:`torch.nn.Module` object inside your
            :class:`lightning.LightningModule` object to log the statistics of. Defaults to "model".
            This name is prefixed to the keys in the logs, allowing logging the statistics
            of multiple modules with multiple instances of this callback.
    """

    def __init__(self, enable=True, grads=True, module_name="model", every_n_epoch=1):
        _check_lightning_available(self.__class__)
        self.enable = enable
        self.grads = grads
        self.module_name = module_name
        self.every_n_epoch = every_n_epoch

    def on_train_epoch_end(self, trainer, pl_module):
        if self.enable:
            if pl_module.current_epoch % self.every_n_epoch != 0:
                return

            model = getattr(pl_module, self.module_name)
            for tag, value in model.named_parameters():
                pl_module.log(
                    f"{self.module_name}_weights/mean_{tag}",
                    value.mean(),
                    sync_dist=True,
                )
                if len(value) > 1:
                    pl_module.log(
                        f"{self.module_name}_weights/std_{tag}",
                        value.std(),
                        sync_dist=True,
                    )
                if self.grads and value.grad is not None:
                    pl_module.log(
                        f"{self.module_name}_grads/mean_{tag}",
                        value.grad.mean(),
                        sync_dist=True,
                    )


class MemInfo(L.Callback):
    r"""Lightning callback to print the memory information of the system at the
    beginning of the training. This uses the `cat /proc/meminfo` command to get the
    memory information.
    """

    def on_train_start(self, trainer, pl_module):
        _check_lightning_available(self.__class__)
        # Log the output of `cat /proc/meminfo` using a shell script.
        try:
            # Execute the command and capture its output
            result = subprocess.run(
                ["cat", "/proc/meminfo"],
                capture_output=True,
                text=True,
                check=True,
            )
            result = result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Command failed with error: {e}")
            result = ""

        # Log the output
        logging.info(f"Memory info: \n{result}")
