import numpy as np
import pytest

from temporaldata import Data, IrregularTimeSeries


# -------------------------------------------------------------------
# Helper: build a sample IrregularTimeSeries with EEG-like signals
# -------------------------------------------------------------------
def make_eeg(T=100, C=8):
    timestamps = np.linspace(0, 1, T)
    np.random.seed(42)
    signal = np.random.randn(T, C)
    data = Data(
        eeg=IrregularTimeSeries(
            timestamps=timestamps,
            signal=signal.copy(),
            domain="auto",
        ),
        domain="auto",
    )
    return data, timestamps, signal


# -------------------------------------------------------------------
# Test: Gaussian additive noise
# -------------------------------------------------------------------
def test_random_noise_gaussian_additive():
    from torch_brain.transforms.add_noise import RandomNoise

    data, timestamps, signal = make_eeg()

    transform = RandomNoise(
        field="eeg.signal",
        loc=0.0,
        scale=0.1,
        distribution="gaussian",
        kind="additive",
        seed=42,
    )

    data_t = transform(data)
    noisy = data_t.eeg.signal

    assert noisy.shape == signal.shape
    assert np.allclose(data_t.eeg.timestamps, timestamps)
    assert not np.allclose(noisy, signal)

    transform2 = RandomNoise(
        field="eeg.signal",
        loc=0.0,
        scale=0.1,
        distribution="gaussian",
        kind="additive",
        seed=42,
    )

    data2, _, _ = make_eeg()
    noise2 = transform2(data2).eeg.signal

    assert np.allclose(noisy, noise2)


# -------------------------------------------------------------------
# Test: Laplace additive noise
# -------------------------------------------------------------------
def test_random_noise_laplace_additive():
    from torch_brain.transforms.add_noise import RandomNoise

    data, timestamps, signal = make_eeg()

    transform = RandomNoise(
        field="eeg.signal",
        loc=0.0,
        scale=0.05,
        distribution="laplace",
        kind="additive",
        seed=7,
    )

    noisy = transform(data).eeg.signal

    assert noisy.shape == signal.shape
    assert not np.allclose(noisy, signal)
    assert np.allclose(data.eeg.timestamps, timestamps)

    transform2 = RandomNoise(
        field="eeg.signal",
        loc=0.0,
        scale=0.05,
        distribution="laplace",
        kind="additive",
        seed=7,
    )

    data2, _, _ = make_eeg()
    noisy2 = transform2(data2).eeg.signal

    assert np.allclose(noisy, noisy2)


# -------------------------------------------------------------------
# Test: Uniform multiplicative noise
# -------------------------------------------------------------------
def test_random_noise_uniform_multiplicative():
    from torch_brain.transforms.add_noise import RandomNoise

    data, timestamps, signal = make_eeg()

    transform = RandomNoise(
        field="eeg.signal",
        loc=0.0,
        scale=0.2,
        distribution="uniform",
        kind="multiplicative",
        seed=123,
    )

    noisy = transform(data).eeg.signal

    assert noisy.shape == signal.shape
    assert np.allclose(data.eeg.timestamps, timestamps)

    factor = noisy / signal

    a = transform.scale * np.sqrt(3)
    expected_min = 1 - a
    expected_max = 1 + a

    assert factor.min() >= expected_min - 1e-6
    assert factor.max() <= expected_max + 1e-6


# -------------------------------------------------------------------
# Test: Clipping behavior
# -------------------------------------------------------------------
def test_random_noise_clip():
    from torch_brain.transforms.add_noise import RandomNoise

    data, _, _ = make_eeg()

    transform = RandomNoise(
        field="eeg.signal",
        loc=-1.0,
        scale=1.0,
        distribution="gaussian",
        kind="additive",
        clip=True,
        seed=0,
    )

    noisy = transform(data).eeg.signal

    assert np.all(noisy >= 0.0)


# -------------------------------------------------------------------
# Test: Invalid distribution raises error
# -------------------------------------------------------------------
def test_random_noise_invalid_distribution():
    from torch_brain.transforms.add_noise import RandomNoise

    data, _, _ = make_eeg()

    transform = RandomNoise(
        field="eeg.signal",
        loc=0.0,
        scale=0.1,
        distribution="invalid_dist",
    )

    with pytest.raises(ValueError):
        transform(data)


# -------------------------------------------------------------------
# Test: Invalid field type raises error
# -------------------------------------------------------------------
def test_random_noise_invalid_field_type():
    from torch_brain.transforms.add_noise import RandomNoise

    data = Data(cursor=np.array([1, 2, 3]))  # not IrregularTimeSeries

    transform = RandomNoise(field="cursor.signal")

    with pytest.raises(ValueError):
        transform(data)


# -------------------------------------------------------------------
# Test: Deterministic seed behavior
# -------------------------------------------------------------------
def test_random_noise_seed_reproducibility():
    from torch_brain.transforms.add_noise import RandomNoise

    # fresh data inputs
    data1, _, _ = make_eeg()
    data2, _, _ = make_eeg()

    transform1 = RandomNoise(field="eeg.signal", scale=0.1, seed=999)
    transform2 = RandomNoise(field="eeg.signal", scale=0.1, seed=999)

    out1 = transform1(data1).eeg.signal
    out2 = transform2(data2).eeg.signal

    assert np.allclose(out1, out2)


def test_random_noise_missing_nested_field():
    from torch_brain.transforms.add_noise import RandomNoise

    data, _, _ = make_eeg()

    # field="eeg" has no nested attribute → should raise ValueError
    transform = RandomNoise(field="eeg", scale=0.1, seed=0)

    with pytest.raises(ValueError):
        transform(data)


# -------------------------------------------------------------------
# Test: Shape preservation for all distributions and modes
# -------------------------------------------------------------------
@pytest.mark.parametrize("distribution", ["gaussian", "laplace", "uniform"])
@pytest.mark.parametrize("kind", ["additive", "multiplicative"])
def test_random_noise_shape_preserved(distribution, kind):
    from torch_brain.transforms.add_noise import RandomNoise

    data, timestamps, signal = make_eeg(T=20, C=3)

    transform = RandomNoise(
        field="eeg.signal",
        loc=0.0,
        scale=0.1,
        distribution=distribution,
        kind=kind,
        seed=1,
    )

    noisy = transform(data).eeg.signal

    assert noisy.shape == signal.shape
    assert np.allclose(data.eeg.timestamps, timestamps)
