"""Compatibility exports for the lightweight PIPPI subset contract."""

from torch_brain.data.pippi import (
    PIPPI_HIGH_COV_SUBJECT_NUMBERS,
    PIPPI_LOW_COV_SUBJECT_NUMBERS,
    PIPPI_SUBSET_TIERS,
    pippi_subject_numbers_for_subset_tier,
    pippi_subset_tiers_for_subject,
)

__all__ = [
    "PIPPI_HIGH_COV_SUBJECT_NUMBERS",
    "PIPPI_LOW_COV_SUBJECT_NUMBERS",
    "PIPPI_SUBSET_TIERS",
    "pippi_subject_numbers_for_subset_tier",
    "pippi_subset_tiers_for_subject",
]
