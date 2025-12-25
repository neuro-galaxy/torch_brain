import brainsets


def test_version_exists():
    assert hasattr(brainsets, "__version__")
    assert len(brainsets.__version__) > 0
