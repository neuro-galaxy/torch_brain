def _deprecated_import_error(name):
    raise ImportError(
        f"`{name}` has been moved to `torch_brain.models.poyo_plus.multitask_readout`. "
        f"Please update your import to: `from torch_brain.models.poyo_plus.multitask_readout import {name}`"
    )


class MultitaskReadout:
    def __init__(self, *args, **kwargs):
        _deprecated_import_error("MultitaskReadout")


def prepare_for_multitask_readout(*args, **kwargs):
    _deprecated_import_error("prepare_for_multitask_readout")
