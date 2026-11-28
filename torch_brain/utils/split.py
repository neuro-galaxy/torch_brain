__all__ = [
    "generate_stratified_folds",
    "generate_string_kfold_assignment",
    "generate_temporal_folds",
    "temporal_metadata",
    "temporal_balance_rng",
]

# Drives the generated API reference; see docs/source/api_reference.py.
__api_ref__ = {
    "description": None,
    "sections": [{"autosummary": __all__}],
}

import hashlib
from collections.abc import Callable
from numbers import Real

import numpy as np

from torch_brain.data import Data, Interval

PROTOCOL_VERSION = "shared-temporal-v1"


def _create_interval_split(intervals: Interval, indices: np.ndarray) -> Interval:
    """Create an Interval subset from indices and sort it."""
    mask = np.zeros(len(intervals), dtype=bool)
    mask[indices] = True
    split = intervals.select_by_mask(mask)
    split.sort()
    return split


def generate_stratified_folds(
    intervals: Interval,
    stratify_by: str,
    n_folds: int = 5,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> list[Data]:
    """
    Generates stratified train/valid/test splits using a two-stage splitting process.

    The splitting is performed in two stages:
        1. Outer split (StratifiedKFold): The intervals are divided into n_folds,
           where each fold uses one partition as the test set and the remaining
           partitions as train+valid. Stratification ensures each fold maintains
           the class distribution of the original data.
        2. Inner split (StratifiedShuffleSplit): The train+valid portion of each fold
           is further split into train and valid sets using val_ratio, while preserving
           the class distribution.

    Args:
        intervals: The intervals to split.
        n_folds: Number of folds for cross-validation.
        val_ratio: Ratio of validation set relative to train+valid combined.
        seed: Random seed.
        stratify_by: The attribute name to use for stratification (e.g., "id", "label",
            "class"). The intervals must have this attribute.

    Returns:
        List of Data objects, one for each fold.

    Raises:
        ValueError: If the intervals don't have the specified stratify_by attribute.
        ValueError: If there are fewer samples than n_folds.
    """
    try:
        from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
    except ImportError as err:
        raise ImportError(
            "This function requires the scikit-learn library which you can install with "
            "`pip install scikit-learn`"
        ) from err

    if not hasattr(intervals, stratify_by):
        raise ValueError(
            f"Intervals must have a '{stratify_by}' attribute for stratification."
        )

    class_labels = getattr(intervals, stratify_by)
    if len(class_labels) < n_folds:
        raise ValueError(
            f"Not enough samples ({len(class_labels)}) for {n_folds} folds."
        )

    outer_splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    sample_indices = np.arange(len(intervals))

    for fold_idx, (train_val_indices, test_indices) in enumerate(
        outer_splitter.split(sample_indices, class_labels)
    ):
        test_split = _create_interval_split(intervals, test_indices)

        train_val_labels = class_labels[train_val_indices]
        inner_splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=val_ratio, random_state=seed + fold_idx
        )

        for train_indices, val_indices in inner_splitter.split(
            train_val_indices, train_val_labels
        ):
            train_original_indices = train_val_indices[train_indices]
            val_original_indices = train_val_indices[val_indices]

            train_split = _create_interval_split(intervals, train_original_indices)
            val_split = _create_interval_split(intervals, val_original_indices)

            combined_domain = train_split | val_split | test_split

            fold_data = Data(
                train=train_split,
                valid=val_split,
                test=test_split,
                domain=combined_domain,
            )

            folds.append(fold_data)

    return folds


def generate_string_kfold_assignment(
    string_id: str,
    n_folds: int = 3,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> list[str]:
    """Generate deterministic per-fold train/valid/test assignments for one ID.

    The assignment is independent for each fold index ``k``, but follows a
    deterministic two-step rule:

    1. Compute a global bucket from ``md5(f"{string_id}_{seed}") % n_folds``.
       The fold whose index equals this bucket is labeled ``"test"``.
    2. For every other fold, compute a fold-specific hash
       ``md5(f"{string_id}_{seed}_{k}")`` and map it to ``[0, 1)``.
       If that value is below ``val_ratio``, the fold is ``"valid"``,
       otherwise it is ``"train"``.

    As a result, each ``string_id`` appears in the test split for exactly one
    fold and is never in test for the remaining folds. This makes the output
    reproducible across runs and safe for parallel processing.

    Args
    ----
    string_id : str
        String identifier (e.g., "S001", "sub-01", or "sub-01_ses-01").
    n_folds : int
        Number of folds for cross-validation. Default is 3.
    val_ratio : float
        Ratio of validation set relative to train+valid combined. Default is 0.2.
    seed : int
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    List[str]
        List of fold assignments where index ``k`` corresponds to fold ``k`` and
        each value is one of ``"train"``, ``"valid"``, or ``"test"``.
        Exactly one entry is ``"test"``.

    Examples
    --------
    >>> assignments = generate_string_kfold_assignment("sub-01", n_folds=3)  # doctest: +SKIP
    >>> assignments  # doctest: +SKIP
    ['train', 'test', 'train']

    >>> generate_string_kfold_assignment("sub-01_ses-01", n_folds=3)  # doctest: +SKIP
    ['valid', 'train', 'test']
    """
    if not isinstance(string_id, str) or not string_id:
        raise ValueError("string_id must be a non-empty string")
    if n_folds < 1:
        raise ValueError(f"n_folds must be at least 1, got {n_folds}")
    if not (0.0 <= val_ratio <= 1.0):
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

    base_str = f"{string_id}_{seed}"
    hash_int = _get_integer_hash_from_string(base_str)
    bucket = hash_int % n_folds

    assignments: list[str] = []
    for k in range(n_folds):
        if bucket == k:
            assignments.append("test")
        else:
            fold_str = f"{base_str}_{k}"
            fold_hash_int = _get_integer_hash_from_string(fold_str)
            normalized_hash = (fold_hash_int % 10000) / 10000.0
            if normalized_hash < val_ratio:
                assignments.append("valid")
            else:
                assignments.append("train")
    return assignments


def _get_integer_hash_from_string(string: str) -> int:
    """
    Compute a deterministicinteger hash from a string using MD5.

    Parameters
    ----------
    string : str
        The string to hash.

    Returns
    -------
    int
        The integer representation of the MD5 hash of the input string.
    """
    base_bytes = string.encode("utf-8")
    hash_obj = hashlib.md5(base_bytes)
    hash_int = int(hash_obj.hexdigest(), 16)
    return hash_int


def _temporal_blocks(
    bounds: np.ndarray, fold: int, *, gap_s: float
) -> dict[str, tuple[float, float]]:
    """Partition validated recording bounds for one of the two protocol folds."""
    fractions = [0, 0.25, 0.5, 1] if fold == 0 else [0, 0.5, 0.75, 1]
    order = ("val", "test", "train") if fold == 0 else ("train", "val", "test")
    edges = bounds[0] + np.asarray(fractions) * (bounds[1] - bounds[0])
    blocks = {}
    for i, split in enumerate(order):
        lo = float(edges[i] + (gap_s / 2 if i else 0))
        hi = float(edges[i + 1] - (gap_s / 2 if i < 2 else 0))
        if hi <= lo:
            raise ValueError("gap_s removes an entire temporal block")
        blocks[split] = (lo, hi)
    return blocks


def temporal_metadata(domain, *, context_width_s=1.0, gap_s=0.0):
    """Describe the shared twofold blocks and support settings for saved selections.

    Args:
        domain: Two finite increasing recording boundaries in seconds.
        context_width_s: Total centered support width, at least one second.
        gap_s: Additional nonnegative blank interval between neighboring blocks.

    Returns:
        Protocol identifier, support width, gap, and a ``blocks`` mapping keyed
        by string fold IDs (``"0"``, ``"1"``), then train/val/test. Block bounds
        are half-open pairs in seconds, ready for H5 JSON metadata serialization.
    """
    if (
        isinstance(context_width_s, bool)
        or not isinstance(context_width_s, Real)
        or not np.isfinite(context_width_s)
        or context_width_s < 1
    ):
        raise ValueError("context_width_s must be finite and at least 1 second")
    bounds = np.asarray(domain, dtype=float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[1] <= bounds[0]:
        raise ValueError("domain must contain two finite increasing boundaries")
    if (
        isinstance(gap_s, bool)
        or not isinstance(gap_s, Real)
        or not np.isfinite(gap_s)
        or gap_s < 0
    ):
        raise ValueError("gap_s must be a finite nonnegative number")
    return {
        "protocol": PROTOCOL_VERSION,
        "context_width_s": float(context_width_s),
        "gap_s": float(gap_s),
        "blocks": {
            str(fold): _temporal_blocks(bounds, fold, gap_s=gap_s) for fold in (0, 1)
        },
    }


def temporal_balance_rng(
    *,
    dataset_id: str,
    balance_seed: int,
    recording_id: str,
    task_name: str,
    label_mode: str,
    fold_idx: int,
    split_name: str,
) -> np.random.Generator:
    """Create a task/split-local RNG without depending on preparation iteration order.

    Dataset and recording identities namespace the seed. Keep these identities
    and ``balance_seed`` fixed when varying model training seeds.
    """
    key = "|".join(
        [
            dataset_id,
            str(int(balance_seed)),
            recording_id,
            task_name,
            label_mode,
            str(int(fold_idx)),
            split_name,
        ]
    )
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    seed = int.from_bytes(digest, byteorder="little", signed=False)
    return np.random.default_rng(seed)


def _balanced_temporal_indices(labels, rng):
    classes = np.unique(labels)
    if len(classes) < 2:
        return np.array([], dtype=int)
    # Preserve the historical positive-first binary RNG draws and output order.
    if np.all(np.isin(classes, [0, 1])):
        classes = (1, 0)
    by_class = [np.flatnonzero(labels == label) for label in classes]
    n = min(map(len, by_class))
    selected = np.concatenate(
        [rng.choice(indices, size=n, replace=False) for indices in by_class]
    )
    rng.shuffle(selected)
    return selected


def generate_temporal_folds(
    start: np.ndarray,
    end: np.ndarray,
    labels: np.ndarray,
    *,
    domain,
    context_width_s: float = 1.0,
    gap_s: float = 0.0,
    balance: bool = False,
    rng: np.random.Generator | None = None,
    rng_factory: Callable[[int, str], np.random.Generator] | None = None,
) -> dict[int, dict[str, Interval]]:
    """Generate two recording-time folds with support exclusion before balancing.

    Fold 0 uses the first quarter for validation, second quarter for test, and
    last half for training. Fold 1 uses the first half for training, third
    quarter for validation, and last quarter for test. All bounds are half-open.

    ``context_width_s`` reserves centered preprocessing support around each
    target; it does not expand returned windows. ``gap_s`` is an additional
    blank interval between neighboring blocks. Boundaries depend only on the
    recording domain, so different tasks share the same partition.

    If ``balance`` is true, downsample each split to its smallest class. A
    missing recording-level class produces an empty split; folds remain present.
    Supply ``rng`` or ``rng_factory(fold_idx, split_name)`` for balancing. The
    factory takes precedence and allows task/split-local deterministic seeds.
    No sampling caps or dataset-specific label transforms are applied here.
    """
    metadata = temporal_metadata(domain, context_width_s=context_width_s, gap_s=gap_s)
    start, end, labels = np.asarray(start), np.asarray(end), np.asarray(labels)
    # Validate once before assigning candidates to any temporal block.
    if (
        start.ndim != 1
        or start.shape != end.shape
        or start.shape != labels.shape
        or not np.isfinite(start).all()
        or not np.isfinite(end).all()
        or np.any(end <= start)
    ):
        raise ValueError("start/end/labels must match finite positive-duration windows")
    # NaN would enter the class universe but match no candidates during balancing.
    if (np.issubdtype(labels.dtype, np.inexact) and not np.isfinite(labels).all()) or (
        labels.dtype == object
        and any(
            isinstance(label, (float, complex, np.inexact)) and not np.isfinite(label)
            for label in labels
        )
    ):
        raise ValueError("labels must not contain nonfinite numeric values")
    if np.any(end - start > context_width_s + 1e-9):
        raise ValueError("context_width_s cannot be shorter than target windows")
    if balance and rng is None and rng_factory is None:
        raise ValueError("balance=True requires an RNG or rng_factory")
    extra = np.maximum(0, (context_width_s - (end - start)) / 2)
    n_classes = len(np.unique(labels))
    folds = {}
    for fold_idx in (0, 1):
        fold = {}
        for split in ("train", "val", "test"):
            lo, hi = metadata["blocks"][str(fold_idx)][split]
            keep = (start - extra >= lo) & (end + extra <= hi)
            selected = np.flatnonzero(keep)
            selected = selected[np.argsort(start[selected], kind="stable")]
            if balance:
                split_rng = (
                    rng_factory(fold_idx, split) if rng_factory is not None else rng
                )
                if split_rng is None:
                    raise ValueError("balance=True requires an RNG or rng_factory")
                # Never move events across time boundaries to restore a class.
                if len(np.unique(labels[selected])) < n_classes:
                    selected = selected[:0]
                selected = selected[
                    _balanced_temporal_indices(labels[selected], split_rng)
                ]
                # Sorting after all RNG draws preserves the balanced selection.
                selected = selected[np.argsort(start[selected], kind="stable")]
            fold[split] = Interval(
                start=start[selected], end=end[selected], label=labels[selected]
            )
        folds[fold_idx] = fold
    return folds
