"""Pipeline contracts for processed neuroprobe evaluation paths."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf


# =========================
# Dataset
# =========================


def _require_brainsets_neuroprobe2025():
    # Import lazily so local tooling/tests that do not instantiate datasets can
    # still import this module without optional data dependencies installed.
    try:
        from brainsets.datasets import Neuroprobe2025
    except ImportError as exc:
        raise ImportError(
            "Processed provider 'neuroprobe2025' requires brainsets with Neuroprobe2025."
        ) from exc
    return Neuroprobe2025


# =========================
# Routing
# =========================

# Keep all provider facts in one place so public/private variants can differ by
# a small localized patch instead of scattering provider checks across helpers.
_PROVIDER_SPECS: dict[str, dict[str, Any]] = {
    "neuroprobe2025": {
        "dataset_class_loader": _require_brainsets_neuroprobe2025,
        "regime_is_multi_subject": {
            "SS-SM": False,
            "SS-DM": False,
            "DS-DM": True,
        },
    },
}


def _get_provider_spec(provider: str) -> dict[str, Any]:
    return _PROVIDER_SPECS[provider]


def is_multi_subject(provider: str, regime: str) -> bool:
    """Whether (provider, regime) involves multiple subjects with potentially
    different channel sets.

    Call only after validate_eval_config has passed.
    """
    return _get_provider_spec(provider)["regime_is_multi_subject"][regime]


def needs_region_intersection_pool(
    provider: str, regime: str, requires_aligned_channels: bool
) -> bool:
    """Whether the (provider, regime, model) combo needs region-intersection pooling.

    Region-intersection pooling is needed when the regime is multi-subject AND
    the model requires aligned channels.

    Call only after validate_eval_config has passed.
    """
    return is_multi_subject(provider, regime) and requires_aligned_channels


def get_dataset_class(provider: str):
    """Return dataset class for a validated provider key.

    Call only after validate_eval_config has passed.
    """
    return _get_provider_spec(provider)["dataset_class_loader"]()


def resolve_provider_n_folds(*, dataset_provider: str, regime: str) -> int:
    """Resolve fold count from dataset class API.

    Call only after validate_eval_config has passed.
    """
    dataset_cls = get_dataset_class(dataset_provider)
    # Dataset classes are the source of truth for fold cardinality per regime.
    resolver = getattr(dataset_cls, "num_folds_for_regime", None)
    if resolver is None or not callable(resolver):
        raise RuntimeError(
            "Selected dataset class does not define num_folds_for_regime(...). "
            "Please add this API on the dataset class to support processed eval routing."
        )
    n_folds = resolver(regime)
    if not isinstance(n_folds, int) or isinstance(n_folds, bool):
        raise TypeError(
            "Dataset class num_folds_for_regime(...) must return int, got "
            f"{type(n_folds).__name__}."
        )
    return n_folds


def _validate_dataset_provider(provider: Any) -> str:
    """Validate dataset.provider and return canonical provider key."""
    if not isinstance(provider, str):
        raise TypeError(
            f"dataset.provider must be a str, got {type(provider).__name__}."
        )
    if provider.strip() != provider:
        raise ValueError(
            "dataset.provider must not include leading/trailing whitespace. "
            f"Got '{provider}'."
        )
    valid_dataset_providers = sorted(_PROVIDER_SPECS.keys())
    if provider not in _PROVIDER_SPECS:
        raise ValueError(
            "dataset.provider must be one of "
            f"{valid_dataset_providers}, got '{provider}'."
        )
    return provider


def _validate_provider_regime(
    *,
    dataset_provider: Any,
    dataset_regime: Any,
) -> None:
    """Validate dataset.provider + dataset.regime pair.

    Used only inside validate_eval_config for boundary validation.
    """
    provider = _validate_dataset_provider(dataset_provider)
    if not isinstance(dataset_regime, str):
        raise TypeError(
            f"dataset.regime must be a str, got {type(dataset_regime).__name__}."
        )
    if dataset_regime.strip() != dataset_regime:
        raise ValueError(
            "dataset.regime must not include leading/trailing whitespace. "
            f"Got '{dataset_regime}'."
        )
    if dataset_regime == "":
        raise ValueError("dataset.regime must be non-empty.")
    provider_routes = _get_provider_spec(provider)["regime_is_multi_subject"]
    if dataset_regime not in provider_routes:
        raise ValueError(
            "Unsupported dataset provider/regime tuple: "
            f"(dataset.provider='{provider}', dataset.regime='{dataset_regime}'). "
            "Allowed regimes for provider are "
            f"{sorted(provider_routes.keys())}."
        )


def build_processed_split_provider(
    *,
    dataset_provider: Any,
    dataset_cfg: Any,
    split: str,
    fold_idx: int,
    regime: str,
):
    """Instantiate one split provider object via dataset.provider."""
    dataset_cls = get_dataset_class(dataset_provider)

    # Constructor kwargs are intentionally mirrored from dataset_cfg so this
    # helper stays a thin adapter over dataset-class APIs.
    return dataset_cls(
        root=dataset_cfg.root,
        dirname=dataset_cfg.dirname,
        subset_tier=dataset_cfg.subset_tier,
        test_subject=dataset_cfg.test_subject,
        test_session=dataset_cfg.test_session,
        split=split,
        label_mode=dataset_cfg.label_mode,
        task=dataset_cfg.task,
        regime=regime,
        fold=fold_idx,
        uniquify_channel_ids_with_subject=dataset_cfg.uniquify_channel_ids_with_subject,
        uniquify_channel_ids_with_session=dataset_cfg.uniquify_channel_ids_with_session,
    )


# =========================
# Config
# =========================

VALID_LABEL_MODES = {"binary", "multiclass"}
VALID_SUBSET_TIERS = {"full", "lite", "nano"}


def _validate_label_mode(label_mode: str) -> None:
    if label_mode not in VALID_LABEL_MODES:
        raise ValueError(
            f"label_mode must be one of {sorted(VALID_LABEL_MODES)}, got '{label_mode}'."
        )


def _validate_subset_tier(subset_tier: str) -> None:
    if subset_tier not in VALID_SUBSET_TIERS:
        raise ValueError(
            "dataset.subset_tier must be one of "
            f"{sorted(VALID_SUBSET_TIERS)}, got '{subset_tier}'."
        )


def _parse_required_bool(value: Any, key: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"dataset.{key} must be a bool, got {type(value).__name__}.")


def _require_non_empty_dataset_str(dataset_cfg: dict[str, Any], key: str) -> str:
    if key not in dataset_cfg:
        raise ValueError(
            f"Missing required dataset.{key}. Set dataset.{key} explicitly."
        )
    value = dataset_cfg[key]
    if not isinstance(value, str):
        raise TypeError(f"dataset.{key} must be a str, got {type(value).__name__}.")
    if value == "":
        raise ValueError(f"dataset.{key} must be non-empty.")
    if value.strip() != value:
        raise ValueError(
            f"dataset.{key} must not include leading/trailing whitespace. "
            f"Got '{value}'."
        )
    return value


def _require_dataset_int(dataset_cfg: dict[str, Any], key: str) -> int:
    if key not in dataset_cfg:
        raise ValueError(
            f"Missing required dataset.{key}. Set dataset.{key} explicitly."
        )
    value = dataset_cfg[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"dataset.{key} must be an int, got {type(value).__name__}.")
    return value


def _parse_optional_dataset_brain_area_key(dataset_cfg: dict[str, Any]) -> str | None:
    value = dataset_cfg.get("brain_area_key")
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(
            f"dataset.brain_area_key must be a str when set, got {type(value).__name__}."
        )
    if value == "":
        raise ValueError("dataset.brain_area_key must be non-empty when set.")
    if value.strip() != value:
        raise ValueError(
            "dataset.brain_area_key must not include leading/trailing whitespace. "
            f"Got '{value}'."
        )
    return value


def _require_cfg_mapping(cfg: DictConfig, section: str):
    section_cfg = cfg.get(section)
    if section_cfg is None:
        raise ValueError(f"cfg.{section} is required and must be a mapping.")
    if not hasattr(section_cfg, "get"):
        raise TypeError(f"cfg.{section} must be a mapping/dict-like object.")
    return section_cfg


def _require_non_empty_cfg_str(section_cfg, *, section: str, key: str) -> str:
    value = section_cfg.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{section}.{key} must be a str, got {type(value).__name__}.")
    if value == "":
        raise ValueError(f"{section}.{key} must be non-empty.")
    if value.strip() != value:
        raise ValueError(
            f"{section}.{key} must not include leading/trailing whitespace. "
            f"Got '{value}'."
        )
    return value


def validate_eval_config(cfg: DictConfig) -> None:
    """Validate full eval config: dataset, model, runtime, submitter, runner.

    Call this once before the evaluation loop. After it returns, callers can
    trust that all cfg sections are well-formed and access them directly.
    """
    # -- dataset --
    # Resolve once into a plain mapping to avoid accidental OmegaConf mutation
    # while applying strict validation checks.
    dataset_cfg = OmegaConf.to_container(cfg.get("dataset", {}), resolve=True) or {}
    if not isinstance(dataset_cfg, dict):
        raise TypeError("cfg.dataset must resolve to a mapping/dict.")

    _require_non_empty_dataset_str(dataset_cfg, "root")
    _require_non_empty_dataset_str(dataset_cfg, "dirname")
    _require_non_empty_dataset_str(dataset_cfg, "task")
    _validate_label_mode(_require_non_empty_dataset_str(dataset_cfg, "label_mode"))
    _validate_subset_tier(_require_non_empty_dataset_str(dataset_cfg, "subset_tier"))
    _require_dataset_int(dataset_cfg, "test_subject")
    _require_dataset_int(dataset_cfg, "test_session")
    provider = _require_non_empty_dataset_str(dataset_cfg, "provider")
    regime = _require_non_empty_dataset_str(dataset_cfg, "regime")
    _validate_provider_regime(dataset_provider=provider, dataset_regime=regime)
    _parse_required_bool(
        dataset_cfg.get("uniquify_channel_ids_with_subject"),
        "uniquify_channel_ids_with_subject",
    )
    _parse_required_bool(
        dataset_cfg.get("uniquify_channel_ids_with_session"),
        "uniquify_channel_ids_with_session",
    )
    _parse_required_bool(
        dataset_cfg.get("merge_val_into_test"),
        "merge_val_into_test",
    )
    dataset_brain_area_key = _parse_optional_dataset_brain_area_key(dataset_cfg)

    # -- model / channel compatibility --
    model_cfg = _require_cfg_mapping(cfg, "model")
    _require_non_empty_cfg_str(model_cfg, section="model", key="name")
    requires_aligned = model_cfg.get("requires_aligned_channels")
    if not isinstance(requires_aligned, bool):
        raise TypeError(
            "model.requires_aligned_channels must be a bool, got "
            f"{type(requires_aligned).__name__}."
        )
    requires_coords = model_cfg.get("requires_coords")
    if not isinstance(requires_coords, bool):
        raise TypeError(
            "model.requires_coords must be a bool, got "
            f"{type(requires_coords).__name__}."
        )

    # Derive region-intersection pooling need from (dataset, regime, model).
    pool = needs_region_intersection_pool(provider, regime, requires_aligned)
    if pool and dataset_brain_area_key is None:
        raise ValueError(
            "dataset.brain_area_key is required when the regime is multi-subject "
            f"and model.requires_aligned_channels is true "
            f"(dataset.provider='{provider}', dataset.regime='{regime}')."
        )

    # -- preprocessor --
    preprocessor_cfg = _require_cfg_mapping(cfg, "preprocessor")
    _require_non_empty_cfg_str(preprocessor_cfg, section="preprocessor", key="name")
    if pool:
        chain = preprocessor_cfg.get("chain") or []
        stage_names = {s.get("name") for s in chain if hasattr(s, "get")}
        if "region_intersection_pool" not in stage_names:
            raise ValueError(
                "Multi-subject regime with model.requires_aligned_channels=true "
                "requires a preprocessor chain that includes the "
                "'region_intersection_pool' stage "
                f"(dataset.provider='{provider}', dataset.regime='{regime}')."
            )

    # -- runtime --
    runtime_cfg = _require_cfg_mapping(cfg, "runtime")
    if "seed" not in runtime_cfg:
        raise ValueError("runtime.seed is required.")
    seed = runtime_cfg.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"runtime.seed must be an int, got {type(seed).__name__}.")
    overwrite = runtime_cfg.get("overwrite")
    if not isinstance(overwrite, bool):
        raise TypeError(
            f"runtime.overwrite must be a bool, got {type(overwrite).__name__}."
        )

    # -- submitter --
    submitter_cfg = _require_cfg_mapping(cfg, "submitter")
    _require_non_empty_cfg_str(
        submitter_cfg,
        section="submitter",
        key="author",
    )
    _require_non_empty_cfg_str(submitter_cfg, section="submitter", key="organization")
    _require_non_empty_cfg_str(
        submitter_cfg, section="submitter", key="organization_url"
    )
