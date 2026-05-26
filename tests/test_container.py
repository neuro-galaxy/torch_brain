import logging

import numpy as np
import pytest

from temporaldata import ArrayDict, Data, IrregularTimeSeries
from torch_brain.transforms.container import SkipOnFailure


@pytest.fixture
def mock_data():
    timestamps = np.arange(200)
    unit_index = [0] * 10 + [1] * 20 + [2] * 70 + [3] * 10 + [4] * 20 + [5] * 70
    unit_index = np.array(unit_index)
    data = Data(
        spikes=IrregularTimeSeries(
            timestamps=timestamps,
            unit_index=unit_index,
            domain="auto",
        ),
        units=ArrayDict(
            id=np.array(
                [
                    "sorted_a",
                    "unsorted_a",
                    "sorted_b",
                    "unsorted_b",
                    "sorted_c",
                    "unsorted_c",
                ]
            ),
        ),
        domain="auto",
    )
    return data


class TestSkipOnFailure:
    class AlwaysSucceed:
        def __call__(self, data):
            data.units.id[0] = "changed"
            return data

    class AlwaysFail:
        def __call__(self, data):
            raise ValueError("intentional failure")

    class MutateAndFail:
        def __call__(self, data):
            data.units.id[0] = "mutated"
            raise ValueError("fail after mutation")

    def test_success_returns_transformed_data(self, mock_data):
        transform = SkipOnFailure(self.AlwaysSucceed())
        result = transform(mock_data)
        assert result.units.id[0] == "changed"

    def test_failure_returns_original_data(self, mock_data):
        transform = SkipOnFailure(self.AlwaysFail())
        result = transform(mock_data)
        assert result is mock_data

    def test_failure_emits_warning_when_warn_is_true(self, mock_data, caplog):
        transform = SkipOnFailure(self.AlwaysFail(), warn=True)
        with caplog.at_level(logging.WARNING):
            transform(mock_data)
        assert len(caplog.records) == 1
        assert "intentional failure" in caplog.records[0].message

    def test_failure_no_warning_by_default(self, mock_data, caplog):
        transform = SkipOnFailure(self.AlwaysFail())
        with caplog.at_level(logging.WARNING):
            transform(mock_data)
        assert len(caplog.records) == 0

    def test_failure_with_backup_copy_restores_pre_mutation_state(self, mock_data):
        original_id = mock_data.units.id[0]
        transform = SkipOnFailure(self.MutateAndFail(), backup_copy=True)
        result = transform(mock_data)
        assert result.units.id[0] == original_id

    def test_failure_without_backup_copy_reflects_mutation(self, mock_data):
        transform = SkipOnFailure(self.MutateAndFail(), backup_copy=False)
        result = transform(mock_data)
        assert result.units.id[0] == "mutated"

    @pytest.mark.parametrize("exc_type", [ValueError, RuntimeError, KeyError])
    def test_all_exception_types_are_caught(self, mock_data, exc_type):
        class FailWith:
            def __init__(self, exc):
                self.exc = exc

            def __call__(self, data):
                raise self.exc("fail")

        transform = SkipOnFailure(FailWith(exc_type))
        result = transform(mock_data)
        assert result is mock_data
