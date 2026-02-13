import numpy as np
import pytest
from temporaldata import Data, Interval
from brainsets.utils.split import (
    chop_intervals,
    generate_stratified_folds,
    generate_train_valid_splits_one_epoch,
)


class TestChopIntervals:
    def test_chop_intervals_exact_multiple(self):
        start = np.array([0.0, 200.0])
        end = np.array([100.0, 250.0])
        ids = np.array([1, 2])
        intervals = Interval(start=start, end=end, id=ids)

        duration = 10.0
        chopped = chop_intervals(intervals, duration=duration)

        # 10 chunks from first interval (100s), 5 from second (50s)
        assert len(chopped) == 15
        assert np.allclose(chopped.end - chopped.start, duration)

        # Verify gaps are preserved
        sorted_indices = np.argsort(chopped.start)
        sorted_starts = chopped.start[sorted_indices]
        sorted_ends = chopped.end[sorted_indices]

        # Gap between end of chunk 9 (100.0) and start of chunk 10 (200.0)
        assert np.isclose(sorted_ends[9], 100.0)
        assert np.isclose(sorted_starts[10], 200.0)

        # Verify IDs are preserved
        assert np.all(chopped.id[:10] == 1)
        assert np.all(chopped.id[10:] == 2)

    def test_chop_intervals_with_remainder(self):
        start = np.array([0.0])
        end = np.array([25.0])
        ids = np.array([1])
        intervals = Interval(start=start, end=end, id=ids)

        duration = 10.0
        chopped = chop_intervals(intervals, duration=duration)

        # 2 full chunks (0-10, 10-20) + 1 shorter chunk (20-25)
        assert len(chopped) == 3
        assert np.allclose(chopped.start, [0.0, 10.0, 20.0])
        assert np.allclose(chopped.end, [10.0, 20.0, 25.0])
        assert np.all(chopped.id == 1)

    def test_chop_intervals_shorter_than_duration(self):
        start = np.array([0.0, 100.0])
        end = np.array([5.0, 103.0])
        ids = np.array([1, 2])
        intervals = Interval(start=start, end=end, id=ids)

        duration = 10.0
        chopped = chop_intervals(intervals, duration=duration)

        # Both intervals are shorter than duration, kept as-is
        assert len(chopped) == 2
        assert np.allclose(chopped.start, [0.0, 100.0])
        assert np.allclose(chopped.end, [5.0, 103.0])
        assert np.array_equal(chopped.id, [1, 2])

    def test_chop_intervals_overlapping_raises(self):
        start = np.array([0.0, 50.0])
        end = np.array([100.0, 150.0])
        intervals = Interval(start=start, end=end)

        with pytest.raises(ValueError, match="Intervals overlap"):
            chop_intervals(intervals, duration=10.0, check_no_overlap=True)

    def test_chop_intervals_overlapping_no_check(self):
        start = np.array([0.0, 50.0])
        end = np.array([100.0, 150.0])
        intervals = Interval(start=start, end=end)

        chopped = chop_intervals(intervals, duration=10.0, check_no_overlap=False)
        assert len(chopped) == 20


class TestGenerateStratifiedFolds:
    def test_generate_stratified_folds(self):
        n_samples = 100
        start = np.arange(n_samples, dtype=float)
        end = start + 1.0
        # Imbalanced classes: 60 of class 0, 30 of class 1, 10 of class 2
        ids = np.concatenate(
            [np.zeros(60, dtype=int), np.ones(30, dtype=int), np.full(10, 2, dtype=int)]
        )
        # Shuffle to make it realistic
        rng = np.random.default_rng(42)
        perm = rng.permutation(n_samples)
        ids = ids[perm]
        start = start[perm]
        end = end[perm]

        intervals = Interval(start=start, end=end, id=ids)

        n_folds = 5
        val_ratio = 0.25
        folds = generate_stratified_folds(
            intervals, stratify_by="id", n_folds=n_folds, val_ratio=val_ratio, seed=42
        )

        assert isinstance(folds, list)
        assert len(folds) == n_folds

        test_indices_all = []

        for fold in folds:
            assert isinstance(fold, Data)
            train, valid, test = fold.train, fold.valid, fold.test

            # 1. Verify sizes
            # Test: ~1/5 of 100 = 20 samples (from n_folds)
            assert len(test) == 20
            # Valid: 0.25 of remaining 80 = 20 samples
            assert len(valid) == 20
            # Train: 80 - 20 = 60 samples
            assert len(train) == 60

            # 2. Verify Stratification in Test Set
            # Expected counts for test set (20 samples):
            # Class 0: 0.6 * 20 = 12
            # Class 1: 0.3 * 20 = 6
            # Class 2: 0.1 * 20 = 2
            test_ids = test.id
            unique, counts = np.unique(test_ids, return_counts=True)
            counts_dict = dict(zip(unique, counts))

            # Allow small deviation due to rounding/randomness
            assert counts_dict.get(0, 0) in [11, 12, 13]
            assert counts_dict.get(1, 0) in [5, 6, 7]
            assert counts_dict.get(2, 0) in [1, 2, 3]

            # 3. Verify Stratification in Valid Set
            # Expected for valid (20 samples):
            # Class 0: 0.6 * 20 = 12
            # Class 1: 0.3 * 20 = 6
            # Class 2: 0.1 * 20 = 2
            valid_ids = valid.id
            v_unique, v_counts = np.unique(valid_ids, return_counts=True)
            v_counts_dict = dict(zip(v_unique, v_counts))

            assert v_counts_dict.get(0, 0) in [11, 12, 13]
            assert v_counts_dict.get(1, 0) in [5, 6, 7]
            assert v_counts_dict.get(2, 0) in [1, 2, 3]

            # 4. Collect test indices/IDs to verify full coverage later
            test_indices_all.append(test.start)

        # 5. Verify all samples are used in test sets exactly once across folds
        all_test_starts = np.concatenate(test_indices_all)
        all_test_starts_sorted = np.sort(all_test_starts)
        original_starts_sorted = np.sort(intervals.start)

        assert np.allclose(all_test_starts_sorted, original_starts_sorted)

    def test_generate_stratified_folds_custom_attribute(self):
        n_samples = 50
        start = np.arange(n_samples, dtype=float)
        end = start + 1.0
        # Use "label" instead of "id"
        labels = np.array(["A"] * 25 + ["B"] * 25)

        rng = np.random.default_rng(42)
        perm = rng.permutation(n_samples)
        labels = labels[perm]
        start = start[perm]
        end = end[perm]

        intervals = Interval(start=start, end=end, label=labels)

        folds = generate_stratified_folds(
            intervals, stratify_by="label", n_folds=5, val_ratio=0.25, seed=42
        )

        assert isinstance(folds, list)
        assert len(folds) == 5

        for fold in folds:
            assert isinstance(fold, Data)
            test_labels = fold.test.label
            unique, counts = np.unique(test_labels, return_counts=True)
            assert len(unique) == 2
            assert all(c == 5 for c in counts)

    def test_generate_stratified_folds_missing_attribute(self):
        start = np.arange(10, dtype=float)
        end = start + 1.0
        intervals = Interval(start=start, end=end)

        with pytest.raises(ValueError, match="must have a 'label' attribute"):
            generate_stratified_folds(intervals, stratify_by="label", n_folds=5)


class TestGenerateTrainValidSplitsOneEpoch:
    def test_default_split_ratios(self):
        epoch = Interval(start=np.array([0.0]), end=np.array([100.0]))
        train, valid = generate_train_valid_splits_one_epoch(epoch)

        assert len(train) == 1
        assert len(valid) == 1
        assert np.isclose(train.start[0], 0.0)
        assert np.isclose(train.end[0], 90.0)
        assert np.isclose(valid.start[0], 90.0)
        assert np.isclose(valid.end[0], 100.0)

    def test_custom_split_ratios(self):
        epoch = Interval(start=np.array([0.0]), end=np.array([100.0]))
        train, valid = generate_train_valid_splits_one_epoch(
            epoch, split_ratios=[0.8, 0.2]
        )

        assert np.isclose(train.start[0], 0.0)
        assert np.isclose(train.end[0], 80.0)
        assert np.isclose(valid.start[0], 80.0)
        assert np.isclose(valid.end[0], 100.0)

    def test_non_zero_start(self):
        epoch = Interval(start=np.array([50.0]), end=np.array([150.0]))

        train, valid = generate_train_valid_splits_one_epoch(
            epoch, split_ratios=[0.7, 0.3]
        )

        assert np.isclose(train.start[0], 50.0)
        assert np.isclose(train.end[0], 120.0)
        assert np.isclose(valid.start[0], 120.0)
        assert np.isclose(valid.end[0], 150.0)

    def test_intervals_are_contiguous(self):
        epoch = Interval(start=np.array([10.0]), end=np.array([110.0]))
        train, valid = generate_train_valid_splits_one_epoch(epoch)
        assert np.isclose(train.end[0], valid.start[0])

    def test_split_ratios_wrong_length(self):
        epoch = Interval(start=np.array([0.0]), end=np.array([100.0]))
        with pytest.raises(ValueError, match="sequence of two numbers"):
            generate_train_valid_splits_one_epoch(epoch, split_ratios=[0.6, 0.2, 0.2])
        with pytest.raises(ValueError, match="sequence of two numbers"):
            generate_train_valid_splits_one_epoch(epoch, split_ratios=[1.0])

    def test_split_ratios_negative(self):
        epoch = Interval(start=np.array([0.0]), end=np.array([100.0]))
        with pytest.raises(ValueError, match="must be non-negative"):
            generate_train_valid_splits_one_epoch(epoch, split_ratios=[1.1, -0.1])

    def test_split_ratios_not_sum_to_one(self):
        epoch = Interval(start=np.array([0.0]), end=np.array([100.0]))
        with pytest.raises(ValueError, match="must sum to 1"):
            generate_train_valid_splits_one_epoch(epoch, split_ratios=[0.5, 0.3])

    def test_multiple_intervals_raises(self):
        epoch = Interval(start=np.array([0.0, 100.0]), end=np.array([50.0, 150.0]))
        with pytest.raises(ValueError, match="must contain a single interval"):
            generate_train_valid_splits_one_epoch(epoch)

    def test_empty_epoch_raises(self):
        epoch = Interval(start=np.array([]), end=np.array([]))
        with pytest.raises(ValueError, match="must contain a single interval"):
            generate_train_valid_splits_one_epoch(epoch)
