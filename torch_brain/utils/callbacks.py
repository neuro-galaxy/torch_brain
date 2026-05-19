def _deprecated_import_error(name, dest="models.poyo_plus.callbacks"):
    raise ImportError(
        f"`{name}` has been moved to `torch_brain.{dest}`. "
        f"Please update your import to: `from torch_brain.{dest} import {name}`"
    )


class EpochTimeLogger:
    def __init__(self, *args, **kwargs):
        _deprecated_import_error("EpochTimeLogger")


class ModelWeightStatsLogger:
    def __init__(self, *args, **kwargs):
        _deprecated_import_error("ModelWeightStatsLogger")


class MemInfo:
    def __init__(self, *args, **kwargs):
        _deprecated_import_error("MemInfo")


class DataForDecodingStitchEvaluator:
    def __init__(self, *args, **kwargs):
        _deprecated_import_error("DataForDecodingStitchEvaluator")


class DecodingStitchEvaluator:
    def __init__(self, *args, **kwargs):
        _deprecated_import_error("DecodingStitchEvaluator")


class DataForMultiTaskDecodingStitchEvaluator:
    def __init__(self, *args, **kwargs):
        _deprecated_import_error("DataForMultiTaskDecodingStitchEvaluator")


class MultiTaskDecodingStitchEvaluator:
    def __init__(self, *args, **kwargs):
        _deprecated_import_error("MultiTaskDecodingStitchEvaluator")
