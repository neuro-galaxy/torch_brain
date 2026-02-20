import torch_brain


def test_version_exists():
    assert hasattr(torch_brain, "__version__")
    assert len(torch_brain.__version__) > 0
