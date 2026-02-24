import numpy as np
import pytest
from temporaldata import Data, Interval
from brainsets.utils.split import (
    generate_stratified_folds,
    generate_string_kfold_assignment,
)


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


class TestGenerateStringKfoldAssignment:
    """Comprehensive tests for generate_string_kfold_assignment function."""

    # ==================== BASIC STRUCTURE & PARAMETERS ====================

    def test_output_structure_and_index_order(self):
        """Test output list schema and assignment values."""
        n_folds = 3
        assignments = generate_string_kfold_assignment("S001", n_folds=n_folds)

        assert isinstance(assignments, list)
        assert len(assignments) == n_folds

        for assignment in assignments:
            assert assignment in ["train", "valid", "test"]

    @pytest.mark.parametrize("n_folds", [1, 2, 3, 5, 10])
    def test_correct_number_of_folds(self, n_folds):
        """Exactly n_folds assignments should be returned."""
        assignments = generate_string_kfold_assignment("S001", n_folds=n_folds)
        assert len(assignments) == n_folds

    @pytest.mark.parametrize("n_folds", [1, 3, 5, 10])
    def test_exactly_one_test_per_subject(self, n_folds):
        """Exactly one fold should be assigned to test (leave-one-out pattern)."""
        assignments = generate_string_kfold_assignment("S001", n_folds=n_folds)

        test_count = sum(1 for v in assignments if v == "test")
        assert test_count == 1

    # ==================== DETERMINISM & REPRODUCIBILITY ====================

    @pytest.mark.parametrize("string_id", ["S001", "S001_S001_ses01"])
    def test_deterministic_assignment(self, string_id):
        """Same inputs with same seed must produce identical assignments."""
        assignments1 = generate_string_kfold_assignment(string_id, n_folds=3, seed=42)
        assignments2 = generate_string_kfold_assignment(string_id, n_folds=3, seed=42)
        assignments3 = generate_string_kfold_assignment(string_id, n_folds=3, seed=42)

        assert assignments1 == assignments2 == assignments3

    def test_different_seeds_different_assignments(self):
        """Different seeds must produce different assignments."""
        # Since there's a 1/N chance of hitting the same bucket by coincidence,
        # we'll use a very large N for tests to effectively remove collision probability.
        assignments_seed1 = generate_string_kfold_assignment(
            "S001", n_folds=1000, seed=42
        )
        assignments_seed2 = generate_string_kfold_assignment(
            "S001", n_folds=1000, seed=84
        )

        assert assignments_seed1 != assignments_seed2

    def test_different_subjects_different_test_fold(self):
        """Different subjects should be assigned to different test folds."""
        n_subjects = 10
        n_folds = 5
        test_fold_assignments = {}

        for i in range(n_subjects):
            assignments = generate_string_kfold_assignment(
                f"S{i:03d}", n_folds=n_folds, seed=42
            )
            for k in range(n_folds):
                if assignments[k] == "test":
                    test_fold_assignments[f"S{i:03d}"] = k
                    break

        unique_test_folds = set(test_fold_assignments.values())
        assert len(unique_test_folds) > 1  # Not all in same fold

    def test_different_sessions_different_assignments(self):
        """Different sessions of same subject get different assignments."""
        assignments_ses1 = generate_string_kfold_assignment(
            "S001_S001_ses01", n_folds=3, seed=42
        )
        assignments_ses2 = generate_string_kfold_assignment(
            "S001_S001_ses02", n_folds=3, seed=42
        )

        assert assignments_ses1 != assignments_ses2

    def test_subject_and_session_lengths_are_uniform(self):
        """Subject-level and session-level IDs share the same fold list schema."""
        assignments_subj = generate_string_kfold_assignment("S001", n_folds=3)
        assignments_sess = generate_string_kfold_assignment(
            "S001_S001_ses01", n_folds=3
        )

        assert len(assignments_subj) == len(assignments_sess) == 3

    # ==================== FOLD ISOLATION (NO SPILLING BETWEEN FOLDS) ====================

    def test_no_subject_spilling_single_subject(self):
        """Single subject appears in only one test fold."""
        subject_id = "S001"
        n_folds = 5

        assignments = generate_string_kfold_assignment(
            subject_id, n_folds=n_folds, seed=42
        )

        test_folds = [k for k in range(n_folds) if assignments[k] == "test"]
        assert len(test_folds) == 1
        test_fold = test_folds[0]

        for k in range(n_folds):
            if k != test_fold:
                assignment = assignments[k]
                assert assignment in ["train", "valid"]

    def test_no_fold_contamination_at_scale(self):
        """At scale, no subject appears in multiple test folds."""
        n_subjects = 500
        n_folds = 5

        subject_test_folds = {}

        for i in range(n_subjects):
            subject_id = f"S{i:04d}"
            assignments = generate_string_kfold_assignment(
                subject_id, n_folds=n_folds, seed=42
            )

            test_count = 0
            for k in range(n_folds):
                if assignments[k] == "test":
                    subject_test_folds[subject_id] = k
                    test_count += 1

            # Each subject should appear in test exactly once
            assert test_count == 1

        assert len(subject_test_folds) == n_subjects

    def test_each_fold_is_test_for_some_subjects(self):
        """Each fold is test fold for approximately n_subjects/n_folds subjects."""
        n_subjects = 1000
        n_folds = 5

        fold_test_counts = {k: 0 for k in range(n_folds)}

        for i in range(n_subjects):
            assignments = generate_string_kfold_assignment(
                f"S{i:04d}", n_folds=n_folds, seed=42
            )

            for k in range(n_folds):
                if assignments[k] == "test":
                    fold_test_counts[k] += 1
                    break

        expected_per_fold = n_subjects / n_folds
        # Each fold should be test for roughly 1/n_folds of subjects (allow 20% deviation)
        for k, count in fold_test_counts.items():
            assert abs(count - expected_per_fold) < expected_per_fold * 0.2

    def test_train_plus_valid_equals_non_test_count(self):
        """For each subject, train + valid + test = n_folds."""
        n_subjects = 200
        n_folds = 5

        for i in range(n_subjects):
            assignments = generate_string_kfold_assignment(
                f"S{i:03d}", n_folds=n_folds, seed=42
            )

            train_count = sum(1 for v in assignments if v == "train")
            valid_count = sum(1 for v in assignments if v == "valid")
            test_count = sum(1 for v in assignments if v == "test")

            assert train_count + valid_count + test_count == n_folds

    # ==================== RATIO CORRECTNESS ====================

    @pytest.mark.parametrize("val_ratio", [0.0, 0.1, 0.2, 0.3, 0.5])
    def test_val_ratio_parameter_variations(self, val_ratio):
        """val_ratio parameter can be set without error."""
        assignments = generate_string_kfold_assignment(
            "S001", n_folds=5, val_ratio=val_ratio, seed=42
        )

        assert len(assignments) == 5
        assert sum(1 for v in assignments if v == "test") == 1

    def test_val_ratio_zero_no_valid_assignments(self):
        """With val_ratio=0, no subjects are assigned to valid."""
        n_subjects = 100
        n_folds = 5

        valid_count = 0

        for i in range(n_subjects):
            assignments = generate_string_kfold_assignment(
                f"S{i:03d}", n_folds=n_folds, val_ratio=0.0, seed=42
            )
            valid_count += sum(1 for v in assignments if v == "valid")

        assert valid_count == 0

    def test_test_fold_consistent_across_val_ratios(self):
        """Test fold assignment is independent of val_ratio."""
        subject_id = "S001"
        n_folds = 5

        test_folds = {}

        for val_ratio in [0.1, 0.2, 0.3, 0.4]:
            assignments = generate_string_kfold_assignment(
                subject_id, n_folds=n_folds, val_ratio=val_ratio, seed=42
            )
            for k in range(n_folds):
                if assignments[k] == "test":
                    test_folds[val_ratio] = k
                    break

        # All val_ratios should assign same fold to test
        assert len(set(test_folds.values())) == 1

    def test_exact_val_ratio_per_fold(self):
        """val_ratio is respected exactly per fold at scale."""
        n_subjects = 500
        n_folds = 5
        val_ratio = 0.25

        fold_valid_counts = {k: 0 for k in range(n_folds)}
        fold_non_test_counts = {k: 0 for k in range(n_folds)}

        for i in range(n_subjects):
            assignments = generate_string_kfold_assignment(
                f"S{i:04d}", n_folds=n_folds, val_ratio=val_ratio, seed=42
            )

            for k in range(n_folds):
                assignment = assignments[k]
                if assignment != "test":
                    fold_non_test_counts[k] += 1
                    if assignment == "valid":
                        fold_valid_counts[k] += 1

        # Check that val_ratio is maintained per fold
        for k in range(n_folds):
            if fold_non_test_counts[k] > 0:
                actual_ratio = fold_valid_counts[k] / fold_non_test_counts[k]
                # Should be within 5% of expected val_ratio
                assert abs(actual_ratio - val_ratio) < 0.05

    def test_global_ratio_distribution_large_scale(self):
        """At global scale, overall train/valid/test ratios are correct."""
        n_subjects = 1000
        n_folds = 5
        val_ratio = 0.2

        test_fold_counts = {k: 0 for k in range(n_folds)}
        valid_count = 0
        train_count = 0

        for i in range(n_subjects):
            assignments = generate_string_kfold_assignment(
                f"S{i:04d}", n_folds=n_folds, val_ratio=val_ratio, seed=42
            )

            for k in range(n_folds):
                assignment = assignments[k]
                if assignment == "test":
                    test_fold_counts[k] += 1
                elif assignment == "valid":
                    valid_count += 1
                else:
                    train_count += 1

        # Expected counts
        expected_test_per_fold = n_subjects / n_folds
        expected_valid_ratio = (1 - 1 / n_folds) * val_ratio
        expected_train_ratio = (1 - 1 / n_folds) * (1 - val_ratio)

        # Verify test distribution (allow 15% deviation)
        for k, count in test_fold_counts.items():
            assert abs(count - expected_test_per_fold) < expected_test_per_fold * 0.15

        # Verify valid/train ratios
        total_assignments = n_subjects * n_folds
        actual_valid_ratio = valid_count / total_assignments
        actual_train_ratio = train_count / total_assignments

        assert abs(actual_valid_ratio - expected_valid_ratio) < 0.05
        assert abs(actual_train_ratio - expected_train_ratio) < 0.05

    @pytest.mark.parametrize(
        "n_folds,val_ratio",
        [
            (3, 0.2),
            (5, 0.2),
            (10, 0.25),
            (5, 0.1),
            (5, 0.3),
        ],
    )
    def test_ratio_correctness_parametrized(self, n_folds, val_ratio):
        """Ratio correctness with various n_folds and val_ratio combinations."""
        n_subjects = 500

        fold_counts = {k: {"test": 0, "valid": 0, "train": 0} for k in range(n_folds)}

        for i in range(n_subjects):
            assignments = generate_string_kfold_assignment(
                f"S{i:04d}", n_folds=n_folds, val_ratio=val_ratio, seed=42
            )

            for k in range(n_folds):
                assignment = assignments[k]
                fold_counts[k][assignment] += 1

        # Check test fold distribution
        expected_per_fold = n_subjects / n_folds
        for k in range(n_folds):
            test_count = fold_counts[k]["test"]
            assert abs(test_count - expected_per_fold) < expected_per_fold * 0.2

        # Check val_ratio in non-test samples
        for k in range(n_folds):
            non_test_total = fold_counts[k]["valid"] + fold_counts[k]["train"]
            if non_test_total > 0:
                actual_ratio = fold_counts[k]["valid"] / non_test_total
                assert abs(actual_ratio - val_ratio) < 0.1
