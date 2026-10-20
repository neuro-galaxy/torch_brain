import numpy as np

from torch_brain.transforms import (
    NeuroleanNHFTransform,
    anharmonic_neural_hamiltonian,
    chiral_chern_orientation,
    dehn_surgery_resolve_doublets,
    nhf_action_angle_coordinates,
    ranicki_milgram_dyad_signature,
    se2_frenet_spike_jet,
    stormer_verlet_symplectic_step,
)


def _synthetic_action_potentials(num_spikes: int = 32, n_samples: int = 64) -> np.ndarray:
    rng = np.random.default_rng(42)
    t = np.arange(n_samples, dtype=np.float64)
    t0 = 24.0
    waveforms = []
    for _ in range(num_spikes):
        amp = rng.uniform(5.0, 9.0)
        width_depol = rng.uniform(2.0, 2.8)
        width_repol = rng.uniform(5.5, 7.5)
        depol = -amp * np.exp(-0.5 * ((t - t0) / width_depol) ** 2)
        repol = 0.38 * amp * np.exp(-0.5 * ((t - (t0 + 6.0)) / width_repol) ** 2)
        wf = depol + repol + rng.normal(0.0, 0.05, size=n_samples)
        waveforms.append(wf)
    return np.stack(waveforms, axis=0)


def test_anharmonic_neural_hamiltonian_and_liouville_invariance():
    q = np.array([0.5, 1.0, 2.0], dtype=np.float64)
    p = np.array([-0.4, 0.8, -1.2], dtype=np.float64)
    res = anharmonic_neural_hamiltonian(q, p)

    assert res["globally_confined"] is True
    assert res["confinement_discriminant_lhs"] > 2.0 * res["confinement_discriminant_rhs"]
    assert np.all(res["u_ion"] >= 0.0)
    expected_asym = -(2.0 / 3.0) * 0.22981 * (q ** 3)
    np.testing.assert_allclose(res["cubic_asymmetry"], expected_asym, rtol=1e-12)

    rng = np.random.default_rng(1337)
    q0 = rng.uniform(-2.0, 2.0, size=256)
    p0 = rng.uniform(-2.0, 2.0, size=256)
    _, _, det_j = stormer_verlet_symplectic_step(q0, p0, dt=0.05)
    np.testing.assert_allclose(det_j, np.ones_like(det_j), atol=1e-14)


def test_nhf_action_angle_coordinates_and_compaction():
    waveforms = _synthetic_action_potentials(num_spikes=40, n_samples=64)
    nhf = nhf_action_angle_coordinates(waveforms, k_modes=1)

    assert nhf["action_I"].shape == (40,)
    assert np.all(nhf["action_I"] > 0.0)
    assert nhf["theta_NH"].shape == (64,)
    assert np.all(np.diff(nhf["theta_NH"]) > 0.0)
    assert 0.0 <= nhf["theta_NH"][0] and nhf["theta_NH"][-1] <= 2.0 * np.pi
    assert nhf["parseval_max_relative_error"] < 1e-13
    assert nhf["nhf_k_mode_capture_pct"] > nhf["fft_k_mode_capture_pct"]
    assert np.isclose(nhf["n1_reference_k1_nhf_pct"], 70.29)
    assert np.isclose(nhf["n1_reference_k1_fft_pct"], 35.58)


def test_dehn_surgery_resolve_doublets():
    fs = 19531.25
    n_samples = 2000
    rng = np.random.default_rng(7)
    signal = rng.normal(0.0, 0.25, size=n_samples)

    grid = np.arange(-20, 21, dtype=np.float64)
    tpl = -8.0 * np.exp(-0.5 * (grid / 2.2) ** 2) + 3.0 * np.exp(-0.5 * ((grid - 5.0) / 5.0) ** 2)

    primary_times = np.array([200, 600, 1000, 1400], dtype=np.int64)
    secondary_lags = [10, 14, 18, 12]  # 0.51 ms .. 0.92 ms (< 2.05 ms collision zone)

    for t0, lag in zip(primary_times, secondary_lags):
        signal[t0 - 20 : t0 + 21] += tpl
        t_sec = t0 + lag
        signal[t_sec - 20 : t_sec + 21] += 0.85 * tpl

    res = dehn_surgery_resolve_doublets(
        signal=signal,
        spike_times=primary_times,
        template=tpl / np.abs(tpl[20]),
        fs=fs,
        max_doublet_dt_ms=2.05,
        secondary_threshold_mad=3.5,
    )

    assert res["colliding_doublets_detected"] == 4
    assert res["surgically_resolved_count"] == 4
    assert np.isclose(res["resolution_rate"], 1.0)
    assert len(res["resolved_secondary_spike_times"]) == 4
    assert np.all(res["sylvester_resultants_ms2"] > 0.05)
    assert np.all(res["residual_energy_ratios"] < 1.0)


def test_chiral_chern_orientation():
    wfs = _synthetic_action_potentials(num_spikes=10, n_samples=41)
    # First 6 somatic (negative trough at center 20), last 4 inverted axonal (positive peak at center 20)
    t = np.arange(41, dtype=np.float64)
    for i in range(6):
        wfs[i] = -7.0 * np.exp(-0.5 * ((t - 20.0) / 2.3) ** 2) + 2.5 * np.exp(-0.5 * ((t - 26.0) / 5.0) ** 2)
    for i in range(6, 10):
        wfs[i] = -wfs[i - 6]

    out = chiral_chern_orientation(wfs, center_index=20)
    assert out["somatic_c1_neg1_count"] == 6
    assert out["axonal_c1_pos1_count"] == 4
    assert set(np.unique(out["c1"])).issubset({-1, 1})
    assert set(np.unique(out["w1"])).issubset({0, 1})
    assert np.all(out["symplectic_area"] > 0.0)


def test_ranicki_milgram_dyad_signature_and_transform():
    rng = np.random.default_rng(99)
    n = 4096
    common_artifact = rng.normal(0.0, 1.5, size=n)
    spk_a = common_artifact + rng.normal(0.0, 0.25, size=n)
    spk_b = common_artifact + rng.normal(0.0, 0.25, size=n)

    # Inject 8 differential single-unit spikes on electrode A only (emergent in V_diff)
    grid = np.arange(-12, 13, dtype=np.float64)
    ap = -4.5 * np.exp(-0.5 * (grid / 2.0) ** 2)
    for loc in [300, 650, 1000, 1350, 1700, 2050, 2400, 2750]:
        spk_a[loc - 12 : loc + 13] += ap

    dyad = ranicki_milgram_dyad_signature(spk_a, spk_b)
    assert dyad["arf_invariant"] == 1
    assert np.isclose(dyad["dyad_weight"], 6.75)
    assert dyad["emergent_hidden_spikes_in_vdiff"] >= 5

    # Concordant dyad with no hidden differential spikes or axonal twist
    spk_c = common_artifact.copy()
    spk_d = common_artifact.copy()
    dyad_concordant = ranicki_milgram_dyad_signature(spk_c, spk_d)
    assert dyad_concordant["arf_invariant"] == 0
    assert np.isclose(dyad_concordant["dyad_weight"], 1.0)

    # Verify composite NeuroleanNHFTransform on dictionary sample
    sample = {
        "waveforms": _synthetic_action_potentials(num_spikes=8, n_samples=64),
        "dyad_pair": (spk_a, spk_b),
    }
    transform = NeuroleanNHFTransform()
    out = transform(sample)
    assert "nhf" in out
    assert "se2_jet" in out
    assert "milgram_chiral" in out
    assert "ranicki_dyad" in out
    assert out["ranicki_dyad"]["arf_invariant"] == 1
    assert out["se2_jet"]["curvature"].shape == (8, 64)
