import os

import pytest
from omegaconf import OmegaConf

os.environ.setdefault("ROOT_DIR_BRAINTREEBANK", "/tmp")

from neuroprobe_eval.utils.pipeline_contracts import validate_eval_config


def _cfg(preprocessor):
    return OmegaConf.create(
        {
            "dataset": {
                "root": "/tmp",
                "dirname": "neuroprobe_2025",
                "provider": "neuroprobe2025",
                "subset_tier": "lite",
                "label_mode": "binary",
                "task": "onset",
                "test_subject": 1,
                "test_session": 1,
                "regime": "DS-DM",
                "brain_area_key": "localization_DesikanKilliany",
                "uniquify_channel_ids_with_subject": True,
                "uniquify_channel_ids_with_session": False,
                "merge_val_into_test": False,
            },
            "model": {
                "name": "cnn",
                "requires_aligned_channels": True,
                "requires_coords": False,
            },
            "preprocessor": preprocessor,
            "runtime": {"seed": 0, "overwrite": True},
            "submitter": {
                "author": "tester",
                "organization": "org",
                "organization_url": "https://example.com",
            },
        }
    )


def test_validate_eval_config_accepts_top_level_region_intersection_pool():
    cfg = _cfg({"name": "region_intersection_pool"})
    validate_eval_config(cfg)


def test_validate_eval_config_rejects_missing_region_intersection_pool():
    cfg = _cfg({"name": "raw"})
    with pytest.raises(ValueError, match="requires either a top-level"):
        validate_eval_config(cfg)
