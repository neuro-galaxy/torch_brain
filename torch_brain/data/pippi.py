"""PIPPI subset-tier definitions shared by loading and preparation."""

from typing import Final

PIPPI_SUBSET_TIERS: Final[tuple[str, ...]] = ("full", "high-cov", "low-cov")

PIPPI_HIGH_COV_SUBJECT_NUMBERS: Final[frozenset[int]] = frozenset({1, 9, 21, 23, 42})
PIPPI_LOW_COV_SUBJECT_NUMBERS: Final[frozenset[int]] = frozenset(
    {2, 3, 10, 22, 25, 28, 33, 39, 50, 59, 63}
)


def pippi_subject_numbers_for_subset_tier(subset_tier: str) -> frozenset[int] | None:
    """Return eligible PIPPI subjects for a subset tier, or all subjects."""
    if subset_tier == "full":
        return None
    if subset_tier == "high-cov":
        return PIPPI_HIGH_COV_SUBJECT_NUMBERS
    if subset_tier == "low-cov":
        return PIPPI_LOW_COV_SUBJECT_NUMBERS
    raise ValueError(
        f"Invalid Pippi subset_tier '{subset_tier}'. Must be one of "
        f"{PIPPI_SUBSET_TIERS}."
    )


def pippi_subset_tiers_for_subject(subject_number: int) -> tuple[str, ...]:
    """Return every PIPPI subset tier containing a subject."""
    tiers = ["full"]
    if subject_number in PIPPI_HIGH_COV_SUBJECT_NUMBERS:
        tiers.append("high-cov")
    if subject_number in PIPPI_LOW_COV_SUBJECT_NUMBERS:
        tiers.append("low-cov")
    return tuple(tiers)


__all__ = [
    "PIPPI_HIGH_COV_SUBJECT_NUMBERS",
    "PIPPI_LOW_COV_SUBJECT_NUMBERS",
    "PIPPI_SUBSET_TIERS",
    "pippi_subject_numbers_for_subset_tier",
    "pippi_subset_tiers_for_subject",
]
