from typing import Optional
import numpy as np
from torch_brain.transforms import TransformType
from torch_brain.utils import np_string_prefix
from temporaldata import Data, Interval

from .dataset import Dataset, DatasetIndex, _ensure_index_has_namespace
from .mixins import SpikingDatasetMixin


class NestedDataset(Dataset):
    """Dataset that composes multiple :class:`Dataset` instances under a single interface.

    Each child dataset is namespaced by a string prefix (its *dataset name*).
    Exposed `recording_id`s therefore take the form
    ``"<dataset_name>/<recording_id>"``. The nested dataset behaves like a
    regular `Dataset`, dispatching all operations to the appropriate child
    dataset based on this prefix.
    """

    def __init__(
        self,
        datasets: list[Dataset] | dict[str, Dataset],
        transform: Optional[TransformType] = None,
    ):
        """Create a `NestedDataset`.

        Args:
            datasets: Either
                - a mapping from dataset name to `Dataset` instance, or
                - a list/tuple of `Dataset` instances.

                When a list/tuple is given, dataset names are inferred from the
                class names of the datasets. In this case, duplicate class names
                are not allowed.
            transform: Optional transform that is applied to samples in :meth:`__getitem__`.

        """
        if _is_dict_like(datasets):
            # this check supports Python dicts, OmegaConf.DictConfig, etc.
            # Why do we want to support OmegaConf.DictConfig? It is received when using
            # hydra.utils.instantiate() to instantiate the NestedDataset
            dataset_dict = datasets
        elif isinstance(datasets, (list, tuple)):
            dataset_names = [x.__class__.__name__ for x in datasets]
            if len(dataset_names) != len(set(dataset_names)):
                raise ValueError(
                    "Duplicate dataset class names found in provided datasets."
                    " Please use a dictionary instead to specify dataset names explicitly."
                )
            dataset_dict = {name: x for name, x in zip(dataset_names, datasets)}
        else:
            raise TypeError(
                f"datasets must be a list/tuple or a dict-like object"
                f" (got {type(datasets)})"
            )

        self._datasets = dataset_dict
        rec_ids = []
        for name, dataset in self._datasets.items():
            rec_ids.extend(np_string_prefix(name + "/", dataset.recording_ids))
        self._recording_ids = np.sort(rec_ids)

        self.transform = transform

    @property
    def datasets(self) -> dict[str, Dataset]:
        """The underlying mapping from dataset name to `Dataset`."""
        return self._datasets

    def get_recording(self, recording_id: str, _namespace: str = "") -> Data:
        """Return a full `Data` recording from the appropriate child dataset.

        Args:
            recording_id: Recording identifier of the form ``"<dataset_name>/<recording_id>"``.
            _namespace: Internal namespace string propagated to child datasets. End users
                normally do not need to pass this explicitly.

        Returns:
            Data: The selected Data object.

        Raises:
            ValueError: If the `recording_id` does not contain a dataset prefix.
        """
        if "/" not in recording_id:
            raise ValueError(
                f"recording_id '{recording_id}' missing dataset prefix. "
                f"Expected format: 'dataset_name/recording_id'."
            )
        dataset_name, recording_id = recording_id.split("/", 1)
        _namespace = _namespace_join(_namespace, dataset_name)
        data = self.datasets[dataset_name].get_recording(recording_id, _namespace)
        self.get_recording_hook(data)
        return data

    def __getitem__(self, index: DatasetIndex) -> Data:
        """Return a sample specified by a `DatasetIndex`.

        The `index.recording_id` must include a dataset prefix of the form
        ``"<dataset_name>/<recording_id>"``. The index is rewritten to strip
        the prefix and forwarded to the selected child dataset. If a
        `transform` was provided at construction time, it is applied to the
        resulting sample before returning it.

        Args:
            index: DatasetIndex containing the full nested recording_id.

        Returns:
            Data: The sampled Data object from the correct sub-dataset.

        Raises:
            ValueError: If `index.recording_id` does not contain a dataset prefix.
        """
        index = _ensure_index_has_namespace(index)

        if "/" not in index.recording_id:
            raise ValueError(
                f"recording_id '{index.recording_id}' missing dataset prefix. "
                f"Expected format: 'dataset_name/recording_id'."
            )
        dataset_name, recording_id = index.recording_id.split("/", 1)
        new_index = DatasetIndex(
            recording_id=recording_id,
            start=index.start,
            end=index.end,
            _namespace=_namespace_join(index._namespace, dataset_name),
        )
        sample = self.datasets[dataset_name][new_index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def get_sampling_intervals(self, *args, **kwargs) -> dict[str, Interval]:
        """Return sampling intervals for all recordings across child datasets.

        Any positional and keyword arguments are forwarded to the underlying
        datasets' :meth:`get_sampling_intervals` methods. Keys in the returned
        dictionary are prefixed with the corresponding dataset name so that
        they match the nested ``"<dataset_name>/<recording_id>"`` convention.

        Returns:
            dict[str, Interval]: Mapping from nested recording id to interval for all contained datasets.
        """
        ans = {}
        for dataset_name, dataset in self.datasets.items():
            samp_intervals = dataset.get_sampling_intervals(*args, **kwargs)
            for rid, interval in samp_intervals.items():
                ans[dataset_name + "/" + rid] = interval
        return ans


class NestedSpikingDataset(SpikingDatasetMixin, NestedDataset):
    """Spiking variant of :class:`NestedDataset`.

    This class combines the nesting behavior of `NestedDataset` with the
    spike-specific utilities provided by :class:`SpikingDatasetMixin`.
    """

    ...


def _is_dict_like(obj) -> bool:
    """Return ``True`` if *obj* behaves like a mapping."""
    return hasattr(obj, "keys") and hasattr(obj, "__getitem__")


def _namespace_join(a: str, b: str) -> str:
    """Join two namespace components with a slash, skipping empty prefixes."""
    return a + "/" + b if a else b
