import numpy as np
from scipy.ndimage import gaussian_filter1d


def compute_rates(
    binned_spikes: np.ndarray,
    method: str = "gaussian",
    sigma: float = 1.0,
    alpha: float = 0.1,
    normalize: bool = False,
    eps: float = 1e-8,
):
    """
    Convert binned spike counts into continuous firing-rate features.

    Args:
        binned_spikes: Array of shape (num_units, num_bins).
        method: One of {"gaussian", "exponential", "none"}.
        sigma: Std (in bins) for Gaussian smoothing.
        alpha: Exponential smoothing factor (0 < alpha < 1).
        normalize: If True, z-score each neuron's rate vector.
        eps: Small constant to avoid divide-by-zero.

    Returns:
        rates: Array of shape (num_units, num_bins) of firing rates.
    """

    binned_spikes = np.asarray(binned_spikes).astype(np.float32)
    num_units, num_bins = binned_spikes.shape

    # 1. No smoothing – raw counts become your "rates"
    if method == "none":
        rates = binned_spikes.copy()

    # 2. Gaussian smoothing (typical in BCIs, encoder–decoder models)
    elif method == "gaussian":
        rates = gaussian_filter1d(binned_spikes, sigma=sigma, axis=1, mode="nearest")

    # 3. Exponential smoothing ("online" firing rate estimate)
    elif method == "exponential":
        rates = np.zeros_like(binned_spikes)
        rates[:, 0] = binned_spikes[:, 0]
        for t in range(1, num_bins):
            rates[:, t] = (1 - alpha) * rates[:, t - 1] + alpha * binned_spikes[:, t]

    else:
        raise ValueError(f"Unknown method '{method}'")

    # 4. Optional normalization (per-unit z-score)
    if normalize:
        mean = rates.mean(axis=1, keepdims=True)
        std = rates.std(axis=1, keepdims=True)
        rates = (rates - mean) / (std + eps)

    return rates
