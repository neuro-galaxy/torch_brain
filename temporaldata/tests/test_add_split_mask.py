import pytest
import os
import copy
import h5py
import numpy as np
import pandas as pd
import tempfile
from temporaldata import (
    ArrayDict,
    LazyArrayDict,
    IrregularTimeSeries,
    LazyIrregularTimeSeries,
    RegularTimeSeries,
    LazyRegularTimeSeries,
    Interval,
    LazyInterval,
    Data,
)


def test_regulartimeseries():
    data = RegularTimeSeries(
        lfp=np.random.random((100, 48)), sampling_rate=10, domain="auto"
    )

    train_domain = Interval(0.0, 6.0)
    valid_domain = Interval(6.0, 8.0)
    test_domain = Interval(8.0, 10.0)

    data.add_split_mask("train", train_domain)
    data.add_split_mask("valid", valid_domain)
    data.add_split_mask("test", test_domain)

    assert data.train_mask.sum() == 60
    assert data.valid_mask.sum() == 20
    assert data.test_mask.sum() == 20

    # check what happens when things are skewed
    data = RegularTimeSeries(
        lfp=np.random.random((100, 48)), sampling_rate=10, domain="auto"
    )

    train_domain = Interval(0.0, 6.051)
    valid_domain = Interval(6.051, 7.999)
    test_domain = Interval(7.999, 10.0)

    data.add_split_mask("train", train_domain)
    data.add_split_mask("valid", valid_domain)
    data.add_split_mask("test", test_domain)

    assert data.train_mask.sum() == 60
    assert data.valid_mask.sum() == 18
    assert data.test_mask.sum() == 20


def test_set_split_domain():
    data = Data(
        lfp=RegularTimeSeries(
            lfp=np.random.random((100, 48)), sampling_rate=10, domain="auto"
        ),
        domain="auto",
    )

    data.set_train_domain(Interval(0.0, 6.0))
    data.set_valid_domain(Interval(6.0, 8.0))
    data.set_test_domain(Interval(8.0, 10.0))

    assert data.lfp.train_mask.sum() == 60
    assert data.lfp.valid_mask.sum() == 20
    assert data.lfp.test_mask.sum() == 20
