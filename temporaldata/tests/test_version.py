import temporaldata


def test_version_exists():
    assert hasattr(temporaldata, "__version__")
    assert len(temporaldata.__version__) > 0
