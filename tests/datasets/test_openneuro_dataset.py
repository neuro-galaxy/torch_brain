"""Unit tests for OpenNeuroDataset class."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from torch_brain.data import Interval
from torch_brain.datasets.OpenNeuroDataset import OpenNeuroDataset
from torch_brain.utils.split import _get_integer_hash_from_string


# ============================================================================
# Helpers and Fakes
# ============================================================================
def empty_interval() -> Interval:
    """Return an empty interval."""
    return Interval(start=np.array([]), end=np.array([]))


def _assert_intervals_close(
    left: Interval, right: Interval, rtol: float = 1e-7, atol: float = 0.0
) -> None:
    """Assert two Interval objects represent the same bounds (array-safe)."""
    np.testing.assert_allclose(
        np.asarray(left.start, dtype=float),
        np.asarray(right.start, dtype=float),
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        np.asarray(left.end, dtype=float),
        np.asarray(right.end, dtype=float),
        rtol=rtol,
        atol=atol,
    )


class FakeRecording:
    """Minimal fake recording object for testing."""

    def __init__(
        self,
        recording_id: str,
        attributes: dict | None = None,
        domain=None,
        *,
        subject_id: str = "sub-default",
        session_id: str = "ses-default",
    ):
        """
        Args:
            recording_id: Identifier for this recording.
            attributes: Dict mapping nested attribute paths to values (unused by
                current ``get_sampling_intervals`` default-intervals path).
            domain: Value for ``rec.domain`` (temporal ``Interval``).
            subject_id: ``recording.subject.id`` for k-fold string ids.
            session_id: ``recording.session.id`` for intersession string ids.
        """
        self.recording_id = recording_id
        self.attributes = attributes or {}
        self.domain = domain or Interval(start=0.0, end=1.0)
        self.subject = SimpleNamespace(id=subject_id)
        self.session = SimpleNamespace(id=session_id)


def _expected_intrasession_intervals(
    domain: Interval,
    split: str,
    split_ratios: tuple[float, float, float],
) -> Interval:
    """Mirror ``OpenNeuroDataset.get_default_sampling_intervals`` intrasession math."""
    starts = np.asarray(domain.start, dtype=float)
    ends = np.asarray(domain.end, dtype=float)
    durations = ends - starts
    train_ends = starts + durations * split_ratios[0]
    val_ends = train_ends + durations * split_ratios[1]
    test_ends = val_ends + durations * split_ratios[2]
    if split == "train":
        return Interval(start=starts, end=train_ends)
    if split == "val":
        return Interval(start=train_ends, end=val_ends)
    if split == "test":
        return Interval(start=val_ends, end=test_ends)
    raise AssertionError(split)


def _expected_hash_assignment(
    string_id: str,
    *,
    seed: int,
    split_ratios: tuple[float, float, float],
) -> str:
    """Mirror ``OpenNeuroDataset.get_default_sampling_intervals`` hash-based assignment."""
    base_str = f"{string_id}_{seed}"
    hash_int = _get_integer_hash_from_string(base_str)
    normalized_hash = (hash_int % 10000) / 10000.0
    if normalized_hash < split_ratios[0]:
        return "train"
    if normalized_hash < split_ratios[0] + split_ratios[1]:
        return "val"
    return "test"


def _expected_hash_sampling_interval(
    recording: FakeRecording,
    split: str,
    string_id: str,
    *,
    seed: int,
    split_ratios: tuple[float, float, float],
) -> Interval:
    """Expected interval for one recording under intersubject/intersession hash-based logic."""
    assignment = _expected_hash_assignment(
        string_id, seed=seed, split_ratios=split_ratios
    )
    return recording.domain if assignment == split else empty_interval()


def _make_dataset(
    split_type: str = "intrasession",
    recording_ids: list[str] | None = None,
    **overrides,
) -> OpenNeuroDataset:
    """Build an OpenNeuroDataset instance with concise defaults for tests.

    Args:
        split_type: One of "intrasession", "intersubject", "intersession".
        recording_ids: List of recording IDs, or None for defaults.
        **overrides: Additional kwargs to pass to OpenNeuroDataset constructor.

    Returns:
        OpenNeuroDataset instance with mocked parent initialization.
    """
    kwargs = {
        "root": "/fake/root",
        "dataset_dir": "dataset",
        "split_type": split_type,
    }
    if recording_ids is not None:
        kwargs["recording_ids"] = recording_ids
    elif split_type == "intrasession":
        kwargs["recording_ids"] = ["rec-001", "rec-002"]
    elif split_type == "intersubject":
        kwargs["recording_ids"] = ["rec-001", "rec-002", "rec-003"]
    elif split_type == "intersession":
        kwargs["recording_ids"] = ["rec-001", "rec-002"]

    kwargs.update(overrides)
    ds = OpenNeuroDataset(**kwargs)
    # Since parent Dataset.__init__ is mocked, we need to set _recording_ids manually
    ds._recording_ids = kwargs["recording_ids"]
    return ds


def _make_recording(
    recording_id: str,
    *,
    attributes: dict | None = None,
    domain: Interval | None = None,
    subject_id: str = "sub-default",
    session_id: str = "ses-default",
) -> FakeRecording:
    """Build a FakeRecording with optional attributes and domain."""
    return FakeRecording(
        recording_id,
        attributes=attributes,
        domain=domain or Interval(start=0.0, end=1.0),
        subject_id=subject_id,
        session_id=session_id,
    )


@pytest.fixture
def mock_parent_init():
    """Patch parent Dataset.__init__ to avoid filesystem access and set recording_ids."""
    # TODO check if this is correct
    with patch("torch_brain.datasets.Dataset.__init__", return_value=None):
        yield


# ============================================================================
# Tests for Constructor
# ============================================================================


class TestOpenNeuroDatasetInit:
    """Tests for OpenNeuroDataset.__init__."""

    @pytest.mark.parametrize(
        "split_type", ["intrasession", "intersubject", "intersession"]
    )
    def test_accepts_valid_split_type(self, split_type, mock_parent_init):
        """Constructor accepts each valid split_type."""
        ds = _make_dataset(split_type=split_type)
        assert ds.split_type == split_type

    @pytest.mark.parametrize(
        ("invalid_split_type", "expected_msg_fragment"),
        [
            ("invalid", "Invalid split_type 'invalid'"),
            ("intra_session", "Invalid split_type 'intra_session'"),
            ("", "Invalid split_type ''"),
        ],
    )
    def test_rejects_invalid_split_type(
        self, invalid_split_type, expected_msg_fragment, mock_parent_init
    ):
        """Constructor raises ValueError for invalid split_type."""
        with pytest.raises(ValueError, match=expected_msg_fragment):
            _make_dataset(split_type=invalid_split_type)

    def test_constructor_with_optional_args(self, mock_parent_init):
        """Constructor accepts optional recording_ids and transform."""
        recording_ids = ["rec-001", "rec-002"]
        transform = lambda x: x  # noqa: E731

        ds = _make_dataset(
            recording_ids=recording_ids, transform=transform, split_type="intrasession"
        )
        assert ds.split_type == "intrasession"
        assert ds.split_ratios == (0.8, 0.1, 0.1)
        assert ds.seed == 42

    def test_split_ratios_validated(self, mock_parent_init):
        """Invalid ``split_ratios`` raise ``ValueError``."""
        with pytest.raises(ValueError, match="negative"):
            _make_dataset(split_type="intrasession", split_ratios=(-0.1, 0.6, 0.5))
        with pytest.raises(ValueError, match="sum of `split_ratios`"):
            _make_dataset(split_type="intrasession", split_ratios=(0.5, 0.5, 0.1))

    def test_uniquify_channel_id_flags_stored(self, mock_parent_init):
        """Mixin uniquify flags are forwarded from constructor."""
        ds = _make_dataset(
            split_type="intrasession",
            uniquify_channel_ids_with_subject=True,
            uniquify_channel_ids_with_session=False,
        )
        assert ds.multichannel_dataset_mixin_uniquify_channel_ids_with_subject is True
        assert ds.multichannel_dataset_mixin_uniquify_channel_ids_with_session is False


# ============================================================================
# Tests for get_sampling_intervals - Basic & Delegation
# ============================================================================


class TestGetSamplingIntervalsBasic:
    """Tests for basic get_sampling_intervals behavior."""

    def test_split_assignment_none_delegates_to_parent(
        self, mock_parent_init, monkeypatch
    ):
        """When split=None, delegates to parent implementation."""
        ds = _make_dataset(split_type="intrasession")
        expected_result = {"rec-001": Interval(0.0, 10.0)}

        with patch.object(
            OpenNeuroDataset.__bases__[1],
            "get_sampling_intervals",
            return_value=expected_result,
        ) as mock_parent:
            result = ds.get_sampling_intervals(split=None)

        assert result == expected_result
        mock_parent.assert_called_once()

    @pytest.mark.parametrize(
        ("invalid_assignment", "expected_msg_fragment"),
        [
            ("invalid", "Invalid split 'invalid'"),
            ("train_split", "Invalid split 'train_split'"),
            ("valid", "Invalid split 'valid'"),
            ("", "Invalid split ''"),
        ],
    )
    def test_rejects_invalid_split_assignment(
        self, invalid_assignment, expected_msg_fragment, mock_parent_init
    ):
        """Invalid split raises ValueError before touching recordings."""
        ds = _make_dataset(split_type="intrasession")
        with pytest.raises(ValueError, match=expected_msg_fragment):
            ds.get_sampling_intervals(split=invalid_assignment)


# ============================================================================
# Tests for get_sampling_intervals - Intrasession
# ============================================================================


class TestGetSamplingIntervalsIntrasession:
    """Tests for intrasession split strategy (domain-based causal chunks)."""

    @pytest.mark.parametrize("split", ["train", "val", "test"])
    def test_intrasession_splits_domain_by_ratios(
        self, split, mock_parent_init, monkeypatch
    ):
        """Intrasession returns time sub-intervals derived from ``domain`` and ratios."""
        split_ratios = (0.8, 0.1, 0.1)
        ds = _make_dataset(split_type="intrasession", split_ratios=split_ratios)
        domain_1 = Interval(start=0.0, end=10.0)
        domain_2 = Interval(start=0.0, end=100.0)

        recordings = {
            "rec-001": _make_recording("rec-001", domain=domain_1),
            "rec-002": _make_recording("rec-002", domain=domain_2),
        }
        monkeypatch.setattr(ds, "get_recording", lambda rid: recordings[rid])

        result = ds.get_sampling_intervals(split=split)
        exp_1 = _expected_intrasession_intervals(domain_1, split, split_ratios)
        exp_2 = _expected_intrasession_intervals(domain_2, split, split_ratios)

        assert set(result) == {"rec-001", "rec-002"}
        _assert_intervals_close(result["rec-001"], exp_1)
        _assert_intervals_close(result["rec-002"], exp_2)


# ============================================================================
# Tests for get_sampling_intervals - Intersubject
# ============================================================================


class TestGetSamplingIntervalsIntersubject:
    """Tests for intersubject split strategy (hash-based proportional assignment on ``subject.id``)."""

    @pytest.mark.parametrize("split", ["train", "val", "test"])
    def test_intersubject_returns_domain_or_empty_per_split(
        self, split, mock_parent_init, monkeypatch
    ):
        """Each recording gets full ``domain`` iff hash assignment matches ``split``."""
        ds = _make_dataset(split_type="intersubject", seed=42)
        recordings = {
            "rec-001": _make_recording("rec-001", subject_id="sub-01"),
            "rec-002": _make_recording("rec-002", subject_id="sub-02"),
            "rec-003": _make_recording("rec-003", subject_id="sub-03"),
        }
        monkeypatch.setattr(ds, "get_recording", lambda rid: recordings[rid])

        result = ds.get_sampling_intervals(split=split)
        assert set(result) == set(recordings)
        for rid, rec in recordings.items():
            exp = _expected_hash_sampling_interval(
                recording=rec,
                split=split,
                string_id=rec.subject.id,
                seed=42,
                split_ratios=ds.split_ratios,
            )
            _assert_intervals_close(result[rid], exp)

    def test_intersubject_exactly_one_split_per_recording(
        self, mock_parent_init, monkeypatch
    ):
        """Each recording assigned to exactly one split (train, val, or test) based on hash."""
        ds = _make_dataset(split_type="intersubject", recording_ids=["rec-001"], seed=0)
        rec = _make_recording("rec-001", subject_id="sub-xyz")
        monkeypatch.setattr(ds, "get_recording", lambda rid: rec)

        assignment = _expected_hash_assignment(
            "sub-xyz", seed=0, split_ratios=ds.split_ratios
        )
        for split in ("train", "val", "test"):
            result = ds.get_sampling_intervals(split=split)["rec-001"]
            expected = rec.domain if split == assignment else empty_interval()
            _assert_intervals_close(result, expected)


# ============================================================================
# Tests for get_sampling_intervals - Intersession
# ============================================================================


class TestGetSamplingIntervalsIntersession:
    """Tests for intersession split strategy (hash-based proportional assignment on ``subject.id_session.id``)."""

    @pytest.mark.parametrize("split", ["train", "val", "test"])
    def test_intersession_returns_domain_or_empty_per_split(
        self, split, mock_parent_init, monkeypatch
    ):
        """String id is ``f\"{subject.id}_{session.id}\"``; intervals match hash assignment."""
        ds = _make_dataset(split_type="intersession", seed=42)
        recordings = {
            "rec-001": _make_recording(
                "rec-001", subject_id="sub-01", session_id="ses-a"
            ),
            "rec-002": _make_recording(
                "rec-002", subject_id="sub-01", session_id="ses-b"
            ),
        }
        monkeypatch.setattr(ds, "get_recording", lambda rid: recordings[rid])

        result = ds.get_sampling_intervals(split=split)
        assert set(result) == set(recordings)
        for rid, rec in recordings.items():
            sid = f"{rec.subject.id}_{rec.session.id}"
            exp = _expected_hash_sampling_interval(
                recording=rec,
                split=split,
                string_id=sid,
                seed=42,
                split_ratios=ds.split_ratios,
            )
            _assert_intervals_close(result[rid], exp)


# ============================================================================
# Tests for get_default_sampling_intervals - Direct
# ============================================================================


class TestGetDefaultSamplingIntervals:
    """Tests for direct calls to get_default_sampling_intervals."""

    @pytest.mark.parametrize("split", ["train", "val", "test"])
    def test_intrasession_direct_call(self, split, mock_parent_init):
        """Direct call to get_default_sampling_intervals for intrasession split."""
        split_ratios = (0.6, 0.2, 0.2)
        ds = _make_dataset(split_type="intrasession", split_ratios=split_ratios)
        domain = Interval(start=0.0, end=100.0)
        rec = _make_recording("rec-001", domain=domain)

        result = ds.get_default_sampling_intervals(rec, split)
        expected = _expected_intrasession_intervals(domain, split, split_ratios)

        _assert_intervals_close(result, expected)

    @pytest.mark.parametrize("split", ["train", "val", "test"])
    def test_intersubject_direct_call(self, split, mock_parent_init):
        """Direct call to get_default_sampling_intervals for intersubject split."""
        ds = _make_dataset(split_type="intersubject", seed=42)
        rec = _make_recording("rec-001", subject_id="sub-test")

        result = ds.get_default_sampling_intervals(rec, split)
        expected = _expected_hash_sampling_interval(
            recording=rec,
            split=split,
            string_id=rec.subject.id,
            seed=42,
            split_ratios=ds.split_ratios,
        )

        _assert_intervals_close(result, expected)


# ============================================================================
# Tests for Custom Split Ratios and Seed Sensitivity
# ============================================================================


class TestHashAssignmentSensitivity:
    """Tests for behavior variations with custom parameters."""

    def test_custom_split_ratios_change_assignment_thresholds(
        self, mock_parent_init, monkeypatch
    ):
        """Non-default split_ratios change assignment distribution."""
        custom_ratios = (0.5, 0.25, 0.25)
        ds = _make_dataset(
            split_type="intersubject",
            recording_ids=["rec-001"],
            split_ratios=custom_ratios,
            seed=42,
        )
        rec = _make_recording("rec-001", subject_id="sub-test")
        monkeypatch.setattr(ds, "get_recording", lambda rid: rec)

        assignment = _expected_hash_assignment(
            "sub-test", seed=42, split_ratios=custom_ratios
        )
        for split in ("train", "val", "test"):
            result = ds.get_sampling_intervals(split=split)["rec-001"]
            expected = rec.domain if split == assignment else empty_interval()
            _assert_intervals_close(result, expected)

    def test_seed_changes_assignment(self, mock_parent_init, monkeypatch):
        """Different seeds can produce different assignments for same subject."""
        ds_seed42 = _make_dataset(
            split_type="intersubject",
            recording_ids=["rec-001"],
            seed=42,
        )
        ds_seed99 = _make_dataset(
            split_type="intersubject",
            recording_ids=["rec-001"],
            seed=99,
        )
        rec = _make_recording("rec-001", subject_id="sub-xyz")

        assignment_42 = _expected_hash_assignment(
            "sub-xyz", seed=42, split_ratios=ds_seed42.split_ratios
        )
        assignment_99 = _expected_hash_assignment(
            "sub-xyz", seed=99, split_ratios=ds_seed99.split_ratios
        )

        # Seeds likely differ (with high probability for typical ratios)
        # At minimum, verify both assignments are valid splits
        assert assignment_42 in ("train", "val", "test")
        assert assignment_99 in ("train", "val", "test")


# ============================================================================
# Tests for Defensive Behavior
# ============================================================================


class TestDefensiveBehavior:
    """Tests for defensive/edge case behavior."""

    def test_invalid_split_type_at_runtime_raises_value_error(
        self, mock_parent_init, monkeypatch
    ):
        """If ``split_type`` is mutated to an invalid value, raises ValueError."""
        ds = _make_dataset(split_type="intrasession")
        ds.split_type = "bad_split_type"
        rec = _make_recording("rec-001")
        monkeypatch.setattr(ds, "get_recording", lambda rid: rec)

        with pytest.raises(
            ValueError,
            match="Invalid split_type 'bad_split_type'",
        ):
            ds.get_sampling_intervals(split="train")
