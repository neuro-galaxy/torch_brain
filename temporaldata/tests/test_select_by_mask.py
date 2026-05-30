import os
import tempfile
from contextlib import contextmanager

import h5py
import numpy as np
import pytest

from temporaldata import (
    ArrayDict,
    Interval,
    IrregularTimeSeries,
    LazyArrayDict,
    LazyInterval,
    LazyIrregularTimeSeries,
    LazyRegularTimeSeries,
    RegularTimeSeries,
)


@pytest.fixture
def test_filepath(request):
    tmpfile = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    filepath = tmpfile.name
    tmpfile.close()

    def finalizer():
        if os.path.exists(filepath):
            os.remove(filepath)

    request.addfinalizer(finalizer)
    return filepath


def _make_array_dict():
    return ArrayDict(x=np.array([1, 2, 3]))


def _make_interval():
    return Interval(
        start=np.array([0.0, 1.0, 2.0]),
        end=np.array([1.0, 2.0, 3.0]),
    )


def _make_irregular():
    return IrregularTimeSeries(
        timestamps=np.array([0.1, 0.2, 0.3]),
        domain="auto",
    )


@contextmanager
def _make_lazy(non_lazy, lazy_cls, test_filepath):
    with h5py.File(test_filepath, "w") as f:
        non_lazy.to_hdf5(f)
    f = h5py.File(test_filepath, "r")
    yield lazy_cls.from_hdf5(f)
    f.close()


class TestInputValidation:
    @pytest.fixture(
        params=[
            "ArrayDict",
            "Interval",
            "IrregularTimeSeries",
            "LazyArrayDict",
            "LazyInterval",
            "LazyIrregularTimeSeries",
        ]
    )
    def obj(self, request, test_filepath):
        name = request.param
        if name == "ArrayDict":
            yield _make_array_dict()
        elif name == "Interval":
            yield _make_interval()
        elif name == "IrregularTimeSeries":
            yield _make_irregular()
        elif name == "LazyArrayDict":
            with _make_lazy(_make_array_dict(), LazyArrayDict, test_filepath) as data:
                yield data
        elif name == "LazyInterval":
            with _make_lazy(_make_interval(), LazyInterval, test_filepath) as data:
                yield data
        elif name == "LazyIrregularTimeSeries":
            with _make_lazy(
                _make_irregular(), LazyIrregularTimeSeries, test_filepath
            ) as data:
                yield data

    def test_select_by_mask_rejects_not_np_array(self, obj):
        with pytest.raises(ValueError, match="mask must be a numpy array"):
            obj.select_by_mask([True, False, True])

    def test_select_by_mask_rejects_2d_mask(self, obj):
        with pytest.raises(ValueError, match="mask must be 1D"):
            obj.select_by_mask(np.array([[True, False, True]]))

    def test_select_by_mask_rejects_non_bool_mask(self, obj):
        with pytest.raises(ValueError, match="mask must be boolean"):
            obj.select_by_mask(np.array([0, 1, 1]))

    def test_select_by_mask_rejects_length_mismatch(self, obj):
        with pytest.raises(ValueError, match="does not match object length"):
            obj.select_by_mask(np.array([True, False]))


class TestLazyMaskIsCopied:
    """Changing the mask numpy array should not change the mask
    cached within the lazy objects (i.e. they should store a copy).

    We check this by actually modifying the original mask object and seeing
    if that affects the lazy attribute lookup process (it should not).
    """

    def test_lazy_arraydict(self, test_filepath):
        with _make_lazy(_make_array_dict(), LazyArrayDict, test_filepath) as data:
            mask = np.array([True, False, True])
            masked = data.select_by_mask(mask)
            # modify mask. `masked` should NOT care about this
            mask[0] = False
            assert len(masked.x) == 2

    def test_lazy_arraydict_doublemask(self, test_filepath):
        with _make_lazy(_make_array_dict(), LazyArrayDict, test_filepath) as data:
            mask1 = np.array([True, False, True])
            masked = data.select_by_mask(mask1)
            mask2 = np.array([True, False])
            masked2 = masked.select_by_mask(mask2)
            mask1[0] = False
            assert len(masked2.x) == 1

    def test_lazy_irregular_ts(self, test_filepath):
        with _make_lazy(
            _make_irregular(), LazyIrregularTimeSeries, test_filepath
        ) as data:
            mask = np.array([True, False, True])
            masked = data.select_by_mask(mask)
            # modify mask. `masked` should NOT care about this
            mask[0] = False
            assert len(masked.timestamps) == 2

    def test_lazy_irregular_ts_doublemask(self, test_filepath):
        with _make_lazy(
            _make_irregular(), LazyIrregularTimeSeries, test_filepath
        ) as data:
            mask1 = np.array([True, False, True])
            masked = data.select_by_mask(mask1)
            mask2 = np.array([True, False])
            masked2 = masked.select_by_mask(mask2)
            mask1[0] = False
            assert len(masked2.timestamps) == 1

    def test_lazy_interval(self, test_filepath):
        with _make_lazy(_make_interval(), LazyInterval, test_filepath) as data:
            mask = np.array([True, False, True])
            masked = data.select_by_mask(mask)
            # modify mask. `masked` should NOT care about this
            mask[0] = False
            assert len(masked.start) == 2

    def test_lazy_interval_doublemask(self, test_filepath):
        with _make_lazy(_make_interval(), LazyInterval, test_filepath) as data:
            mask1 = np.array([True, False, True])
            masked = data.select_by_mask(mask1)
            mask2 = np.array([True, False])
            masked2 = masked.select_by_mask(mask2)
            mask1[0] = False
            assert len(masked2.start) == 1


class TestNewDomainIsNotShallowCopy:

    def test_irregular_ts(self):
        data = _make_irregular()
        masked = data.select_by_mask(np.array([True, False, True]))
        assert id(masked.domain) != id(data.domain)
        assert id(masked.domain.start) != id(data.domain.start)
        assert id(masked.domain.end) != id(data.domain.end)


class TestPrivateAttribsAreDeepCopied:

    def test_array_dict(self):
        data = _make_array_dict()
        data._private = np.array([0, 1])
        masked = data.select_by_mask(np.array([True, False, True]))
        assert id(masked._private) != id(data._private)
        assert np.array_equal(masked._private, data._private)

    def test_irregular_ts(self):
        data = _make_irregular()
        data._private = np.array([0, 1])
        masked = data.select_by_mask(np.array([True, False, True]))
        assert id(masked._private) != id(data._private)
        assert np.array_equal(masked._private, data._private)

    def test_interval(self):
        data = _make_interval()
        data._private = np.array([0, 1])
        masked = data.select_by_mask(np.array([True, False, True]))
        assert id(masked._private) != id(data._private)
        assert np.array_equal(masked._private, data._private)


class TestRegularTimeSeriesNotImplemented:

    def test_raises_not_implemented(self):
        data = RegularTimeSeries(value=np.zeros(4), sampling_rate=1.0)
        with pytest.raises(NotImplementedError):
            data.select_by_mask(np.array([True, False, True, False]))

    def test_lazy_raises_not_implemented(self, test_filepath):
        data = RegularTimeSeries(value=np.zeros(4), sampling_rate=1.0)
        with _make_lazy(data, LazyRegularTimeSeries, test_filepath) as lazy:
            with pytest.raises(NotImplementedError):
                lazy.select_by_mask(np.array([True, False, True, False]))


class TestCachedSorted:
    """_sorted is a private attrib to maintain a cache of whether
    this object is sorted or not. This should be set to None when
    it was originally False, and we do select_by_mask(), i.e. the cache
    should be invalidated. This is because masking can convert an
    unsorted timeseries to a sorted timeseries.
    If the original data is sorted, then the cached value can stay.
    """

    def test_irregular_ts(self):
        data = IrregularTimeSeries(
            timestamps=np.array([0.0, 1.0, 0.5, 2.0]),
            domain="auto",
        )
        assert data.is_sorted() == False

        # If original data is unsorted
        masked = data.select_by_mask(np.array([True, False, True, True]))
        assert masked._sorted == None
        assert masked.is_sorted() == True

        # If original data is sorted
        masked2 = masked.select_by_mask(np.array([True, False, True]))
        assert masked2._sorted == True

    def test_interval(self):
        data = Interval(
            start=np.array([0.0, 1.0, 0.5, 2.0]),
            end=np.array([0.1, 1.1, 0.6, 2.1]),
        )
        assert data.is_sorted() == False

        # If original data is unsorted
        masked = data.select_by_mask(np.array([True, False, True, True]))
        assert masked._sorted == None
        assert masked.is_sorted() == True

        # If original data is sorted
        masked2 = masked.select_by_mask(np.array([True, False, True]))
        assert masked2._sorted == True
