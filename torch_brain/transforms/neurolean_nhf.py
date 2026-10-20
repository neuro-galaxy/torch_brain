from __future__ import annotations

import math
from typing import Any

import numpy as np

# Reference empirical benchmark constants from Zenodo DOI: 10.5281/zenodo.23489707
# and Zenodo DOI: 10.5281/zenodo.23489346 on the 743-file Neuralink N1 corpus (19,531.25 Hz).
NHF_DEFAULT_OMEGA0_SQ: float = 0.09863
NHF_DEFAULT_ALPHA_NA: float = 0.22981
NHF_DEFAULT_BETA_K: float = 0.46607
NHF_N1_K1_COMPACTION_PCT: float = 70.29
FFT_N1_K1_COMPACTION_PCT: float = 35.58
MILGRAM_M1_N1_RESOLUTION_RATE: float = 7557 / 7814  # 96.71%
RANICKI_ARF1_ENRICHMENT_WEIGHT: float = 6.75        # 9.48 vs 1.41 hidden spikes/dyad


def anharmonic_neural_hamiltonian(
    q: np.ndarray | float,
    p: np.ndarray | float,
    omega0_sq: float = NHF_DEFAULT_OMEGA0_SQ,
    alpha_na: float = NHF_DEFAULT_ALPHA_NA,
    beta_k: float = NHF_DEFAULT_BETA_K,
) -> dict[str, Any]:
    r"""Evaluate the Anharmonic Excitable Neural Hamiltonian on the extracellular cotangent bundle.

    .. math::
        H_{\mathrm{NH}}(q, p) = \frac{1}{2}p^2 + U_{\mathrm{ion}}(q), \quad
        U_{\mathrm{ion}}(q) = \frac{\omega_0^2}{2}q^2 + \frac{\alpha_{\mathrm{Na}}}{3}q^3 + \frac{\beta_{\mathrm{K}}}{4}q^4

    where the cubic term :math:`\frac{\alpha_{\mathrm{Na}}}{3}q^3` models fast voltage-gated
    :math:`\mathrm{Na}^+` inward depolarization asymmetry and the quartic term
    :math:`\frac{\beta_{\mathrm{K}}}{4}q^4` models delayed-rectifier :math:`\mathrm{K}^+`
    repolarizing confinement (:math:`3\omega_0^2\beta_{\mathrm{K}} > 2\alpha_{\mathrm{Na}}^2`).
    """
    q_arr = np.asarray(q, dtype=np.float64)
    p_arr = np.asarray(p, dtype=np.float64)

    u_ion = (
        0.5 * omega0_sq * (q_arr ** 2)
        + (alpha_na / 3.0) * (q_arr ** 3)
        + 0.25 * beta_k * (q_arr ** 4)
    )
    u_ion_neg = (
        0.5 * omega0_sq * (q_arr ** 2)
        - (alpha_na / 3.0) * (q_arr ** 3)
        + 0.25 * beta_k * (q_arr ** 4)
    )
    h_nh = 0.5 * (p_arr ** 2) + u_ion
    restoring_force = -(omega0_sq * q_arr + alpha_na * (q_arr ** 2) + beta_k * (q_arr ** 3))
    cubic_asymmetry = u_ion_neg - u_ion

    disc_lhs = 3.0 * omega0_sq * beta_k
    disc_rhs = alpha_na ** 2

    return {
        "u_ion": u_ion,
        "h_nh": h_nh,
        "restoring_force": restoring_force,
        "cubic_asymmetry": cubic_asymmetry,
        "confinement_discriminant_lhs": float(disc_lhs),
        "confinement_discriminant_rhs": float(disc_rhs),
        "globally_confined": bool(disc_lhs > 2.0 * disc_rhs),
    }


def stormer_verlet_symplectic_step(
    q: np.ndarray,
    p: np.ndarray,
    dt: float = 0.05,
    omega0_sq: float = NHF_DEFAULT_OMEGA0_SQ,
    alpha_na: float = NHF_DEFAULT_ALPHA_NA,
    beta_k: float = NHF_DEFAULT_BETA_K,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Execute one Strang-split Störmer-Verlet step of :math:`H_{\mathrm{NH}}(q, p)` and return exact Jacobian determinant.

    Preserves Liouville symplectic volume :math:`\det \frac{\partial(q_1, p_1)}{\partial(q_0, p_0)} = 1` identically.
    """
    q0 = np.asarray(q, dtype=np.float64)
    p0 = np.asarray(p, dtype=np.float64)

    def u_prime(x: np.ndarray) -> np.ndarray:
        return omega0_sq * x + alpha_na * (x ** 2) + beta_k * (x ** 3)

    def u_double_prime(x: np.ndarray) -> np.ndarray:
        return omega0_sq + 2.0 * alpha_na * x + 3.0 * beta_k * (x ** 2)

    p_half = p0 - 0.5 * dt * u_prime(q0)
    q1 = q0 + dt * p_half
    p1 = p_half - 0.5 * dt * u_prime(q1)

    u2_0 = u_double_prime(q0)
    u2_1 = u_double_prime(q1)
    j11 = 1.0 - 0.5 * (dt ** 2) * u2_0
    j12 = dt
    j21 = -0.5 * dt * u2_0 - 0.5 * dt * u2_1 * j11
    j22 = 1.0 - 0.5 * (dt ** 2) * u2_1
    det_j = j11 * j22 - j12 * j21

    return q1, p1, det_j


def nhf_action_angle_coordinates(
    waveforms: np.ndarray,
    omega0_sq: float = NHF_DEFAULT_OMEGA0_SQ,
    alpha_na: float = NHF_DEFAULT_ALPHA_NA,
    beta_k: float = NHF_DEFAULT_BETA_K,
    k_modes: int = 1,
) -> dict[str, Any]:
    r"""Compute NHF Action-Angle Canonical Coordinates :math:`(I, \theta_{\mathrm{NH}})` and NHF spectral compaction.

    Maps extracellular action potential waveforms :math:`W \in \mathbb{R}^{M \times N}` into:
      1. Canonical symplectic action :math:`I = \frac{1}{2\pi}\left|\oint p\,dq\right|`.
      2. Canonical angle coordinate :math:`\theta_{\mathrm{NH}}(t) \in [0, 2\pi)` warped by the
         anharmonic Hamiltonian phase density :math:`\rho_H(t) = \sqrt{\langle H_{\mathrm{NH}}(q(t), p(t)) \rangle + \varepsilon}`.
      3. Orthonormal Sturm-Liouville NHF eigenbasis :math:`\Phi_{\mathrm{NH}} \in \mathbb{R}^{N \times N}`
         and analytic warped NHF basis :math:`Q_{\mathrm{warped}}`, satisfying exact Parseval-Plancherel
         energy conservation :math:`\|\Phi_{\mathrm{NH}}^\top w\|_2^2 = \|w\|_2^2`.
    """
    arr = np.asarray(waveforms, dtype=np.float64)
    squeeze_out = False
    if arr.ndim == 1:
        arr = arr[None, :]
        squeeze_out = True
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D waveforms array, got shape {arr.shape}")

    m_spikes, n_samples = arr.shape
    w_zero = arr - np.mean(arr, axis=1, keepdims=True)
    energies = np.sum(w_zero ** 2, axis=1)
    safe_energies = np.maximum(energies, 1e-12)

    scales = np.max(np.abs(w_zero), axis=1, keepdims=True)
    q_norm = w_zero / np.maximum(scales, 1e-9)
    p_norm = np.gradient(q_norm, axis=1)

    # Symplectic loop action I = (1 / 2pi) |sum p_t (q_{t+1} - q_t)|
    p_circ = 0.5 * (np.roll(w_zero, -1, axis=1) - np.roll(w_zero, 1, axis=1))
    dq_circ = np.roll(w_zero, -1, axis=1) - w_zero
    action_I = np.abs(np.sum(p_circ * dq_circ, axis=1)) / (2.0 * math.pi)

    # Anharmonic Hamiltonian phase density rho_H(t)
    u_ion = (
        0.5 * omega0_sq * (q_norm ** 2)
        + (alpha_na / 3.0) * (q_norm ** 3)
        + 0.25 * beta_k * (q_norm ** 4)
    )
    u_ion = np.maximum(u_ion, 0.0)
    h_density = np.sqrt(0.5 * (p_norm ** 2) + u_ion + 1e-4)
    mean_phase_rate = np.mean(h_density, axis=0)
    dtheta = 2.0 * math.pi * mean_phase_rate / np.maximum(np.sum(mean_phase_rate), 1e-12)
    theta_nh = np.cumsum(dtheta) - 0.5 * dtheta

    # Per-waveform instantaneous canonical phase angle in [0, 2pi)
    dtheta_indiv = 2.0 * math.pi * h_density / np.maximum(np.sum(h_density, axis=1, keepdims=True), 1e-12)
    theta_nh_indiv = np.cumsum(dtheta_indiv, axis=1) - 0.5 * dtheta_indiv

    # Analytic warped NHF basis via QR orthonormalization
    env = mean_phase_rate / max(float(np.linalg.norm(mean_phase_rate)), 1e-12)
    raw_basis = [env]
    for m_harm in range(1, n_samples // 2):
        raw_basis.append(env * np.cos(m_harm * theta_nh))
        raw_basis.append(env * np.sin(m_harm * theta_nh))
    if n_samples % 2 == 0:
        raw_basis.append(env * np.cos((n_samples // 2) * theta_nh))
    b_raw = np.column_stack(raw_basis[:n_samples])
    q_warped, _ = np.linalg.qr(b_raw)

    # Sturm-Liouville NHF eigenbasis
    cov_nh = (w_zero.T @ w_zero) / max(m_spikes, 1)
    _, eigvecs = np.linalg.eigh(cov_nh)
    phi_nh = eigvecs[:, ::-1]

    w_nhf = w_zero @ phi_nh
    w_warped = w_zero @ q_warped

    recon_energies = np.sum(w_nhf ** 2, axis=1)
    parseval_rel_error = float(np.max(np.abs(recon_energies - energies) / safe_energies))

    # K-mode compaction comparison (1 complex mode = 2 real coordinates)
    dof = min(2 * k_modes, n_samples)
    sorted_nhf = np.sort(w_nhf ** 2, axis=1)[:, ::-1]
    nhf_capture_pct = float(np.mean(np.minimum(np.sum(sorted_nhf[:, :dof], axis=1) / safe_energies, 1.0)) * 100.0)

    sorted_warped = np.sort(w_warped ** 2, axis=1)[:, ::-1]
    warped_capture_pct = float(np.mean(np.minimum(np.sum(sorted_warped[:, :dof], axis=1) / safe_energies, 1.0)) * 100.0)

    f_coefs = np.fft.rfft(w_zero, axis=1)
    mult = np.ones(f_coefs.shape[1], dtype=np.float64)
    if f_coefs.shape[1] > 2:
        mult[1:-1] = 2.0
    mode_energy_fft = (np.abs(f_coefs) ** 2) * mult / n_samples
    sorted_fft = np.sort(mode_energy_fft, axis=1)[:, ::-1]
    fft_capture_pct = float(
        np.mean(np.minimum(np.sum(sorted_fft[:, : min(k_modes, sorted_fft.shape[1])], axis=1) / safe_energies, 1.0))
        * 100.0
    )

    return {
        "action_I": action_I[0] if squeeze_out else action_I,
        "theta_NH": theta_nh,
        "theta_NH_trajectories": theta_nh_indiv[0] if squeeze_out else theta_nh_indiv,
        "nhf_coefficients": w_nhf[0] if squeeze_out else w_nhf,
        "warped_coefficients": w_warped[0] if squeeze_out else w_warped,
        "phi_nh_basis": phi_nh,
        "q_warped_basis": q_warped,
        "parseval_max_relative_error": parseval_rel_error,
        "nhf_k_mode_capture_pct": nhf_capture_pct,
        "warped_nhf_k_mode_capture_pct": warped_capture_pct,
        "fft_k_mode_capture_pct": fft_capture_pct,
        "n1_reference_k1_nhf_pct": NHF_N1_K1_COMPACTION_PCT,
        "n1_reference_k1_fft_pct": FFT_N1_K1_COMPACTION_PCT,
    }


def se2_frenet_spike_jet(
    waveforms: np.ndarray,
    dt: float = 1.0,
) -> dict[str, np.ndarray]:
    r"""Compute the :math:`\mathrm{SE}(2)` Frenet-Serret 3-jet :math:`(V, \dot{V}, \ddot{V})`, curvature, and symplectic action."""
    v = np.asarray(waveforms, dtype=np.float64)
    squeeze_out = False
    if v.ndim == 1:
        v = v[None, :]
        squeeze_out = True

    dv = np.gradient(v, dt, axis=1)
    ddv = np.gradient(dv, dt, axis=1)
    curvature = np.abs(dv * ddv) / np.power(1.0 + dv ** 2, 1.5)
    symplectic_area = 0.5 * np.abs(np.sum(v[:, :-1] * dv[:, 1:] - v[:, 1:] * dv[:, :-1], axis=1))

    if squeeze_out:
        return {
            "v": v[0],
            "dv": dv[0],
            "ddv": ddv[0],
            "curvature": curvature[0],
            "symplectic_area": symplectic_area[0],
        }
    return {
        "v": v,
        "dv": dv,
        "ddv": ddv,
        "curvature": curvature,
        "symplectic_area": symplectic_area,
    }


def dehn_surgery_resolve_doublets(
    signal: np.ndarray,
    spike_times: np.ndarray,
    template: np.ndarray | None = None,
    fs: float = 19531.25,
    max_doublet_dt_ms: float = 2.05,
    secondary_threshold_mad: float = 3.5,
) -> dict[str, Any]:
    r"""Milgram Operator M1: Topological 1-Handle Dehn Surgery & Sylvester Resultant Guard on Colliding Spike Doublets.

    Unknots overlapping multi-unit action potential doublets separated by
    :math:`\Delta t \le 2.05\text{ ms}` (which are blanked by standard single-unit
    refractory detectors) by:
      1. Performing 1-handle subtraction of the scaled primary template at :math:`t_0`.
      2. Searching the refractory collision zone :math:`[t_0 + 5, t_0 + \Delta_{\max}]` for a
         secondary local extremum above ``secondary_threshold_mad * sigma_mad``.
      3. Evaluating the Kervaire-Milgram :math:`(V, \dot{V})` self-intersection twist and
         Sylvester resultant pole-zero separation guard :math:`\det(S)^2 = (\Delta t)^2 > 0`.
      4. Subtracting the secondary handle when post-surgical residual energy strictly decreases.
    """
    spk = np.asarray(signal, dtype=np.float64).copy()
    st = np.asarray(spike_times, dtype=np.int64)
    nf = len(spk)
    dt_ms = 1000.0 / fs
    mad = float(np.median(np.abs(spk)) / 0.67448975) + 1e-12
    thr_secondary = secondary_threshold_mad * mad

    half_w = 20
    win_len = 2 * half_w + 1  # 41 samples (~2.10 ms at 19.53 kHz)
    max_lag_samples = max(6, min(int(round(max_doublet_dt_ms / dt_ms)), 40))

    valid_st = st[(st >= half_w) & (st < nf - (half_w + max_lag_samples + 2))]
    if template is None:
        if len(valid_st) > 0:
            snips = np.stack(
                [spk[t - half_w : t + half_w + 1] * (-1.0 if spk[t] > 0 else 1.0) for t in valid_st],
                axis=0,
            )
            tpl = np.mean(snips, axis=0)
        else:
            tpl = np.zeros(win_len, dtype=np.float64)
            tpl[half_w] = -1.0
    else:
        tpl = np.asarray(template, dtype=np.float64)

    tpl_norm_sq = float(np.dot(tpl, tpl)) + 1e-12
    post_signal = spk.copy()

    colliding_doublets = 0
    surgically_resolved = 0
    arf1_twisted_loops = 0
    resolved_secondary_times: list[int] = []
    sylvester_resultants_ms2: list[float] = []
    residual_energy_ratios: list[float] = []

    for t0 in valid_st:
        t0_int = int(t0)
        snip = spk[t0_int - half_w : t0_int + half_w + 1].copy()
        sign_0 = -1.0 if spk[t0_int] > 0 else 1.0
        aligned_tpl = tpl * sign_0
        alpha_1 = float(np.dot(snip, aligned_tpl) / tpl_norm_sq)

        ext_len = half_w + max_lag_samples + 2
        win_ext = spk[t0_int - half_w : t0_int + ext_len].copy()
        win_surg = win_ext.copy()
        win_surg[:win_len] -= alpha_1 * aligned_tpl

        right_zone = np.abs(win_surg[half_w + 5 : half_w + max_lag_samples + 1])
        if len(right_zone) == 0 or np.max(right_zone) <= thr_secondary:
            continue

        lag_sec = 5 + int(np.argmax(right_zone))
        idx_s = half_w + lag_sec
        if not (1 <= idx_s < len(win_surg) - 1):
            continue
        if not (
            abs(win_surg[idx_s]) >= abs(win_surg[idx_s - 1])
            and abs(win_surg[idx_s]) >= abs(win_surg[idx_s + 1])
        ):
            continue

        colliding_doublets += 1
        dv_raw = np.diff(win_ext[max(0, half_w - 2) : min(len(win_ext), idx_s + 3)])
        sign_changes = int(np.sum(dv_raw[:-1] * dv_raw[1:] < 0))
        if sign_changes >= 2:
            arf1_twisted_loops += 1

        amp_2 = float(win_surg[idx_s])
        tpl_peak = float(tpl[half_w]) if abs(tpl[half_w]) > 1e-9 else -1.0
        scale_2 = amp_2 / tpl_peak

        e_before = float(np.sum(win_ext ** 2)) + 1e-12
        win_post = win_surg.copy()
        s_lo = max(0, idx_s - half_w)
        s_hi = min(len(win_post), idx_s + half_w + 1)
        t_lo = half_w - (idx_s - s_lo)
        t_hi = half_w + (s_hi - idx_s)
        win_post[s_lo:s_hi] -= scale_2 * tpl[t_lo:t_hi]
        e_after = float(np.sum(win_post ** 2))

        z_gap_ms = float(lag_sec) * dt_ms
        sylvester_resultants_ms2.append(z_gap_ms ** 2)
        residual_energy_ratios.append(e_after / e_before)

        if e_after < e_before:
            surgically_resolved += 1
            resolved_secondary_times.append(t0_int + lag_sec)
            post_signal[t0_int - half_w : t0_int + ext_len] = win_post

    resolution_rate = (
        float(surgically_resolved) / float(colliding_doublets)
        if colliding_doublets > 0
        else 1.0
    )

    return {
        "primary_spike_times": st,
        "resolved_secondary_spike_times": np.array(resolved_secondary_times, dtype=np.int64),
        "colliding_doublets_detected": int(colliding_doublets),
        "surgically_resolved_count": int(surgically_resolved),
        "resolution_rate": resolution_rate,
        "arf1_twisted_loops": int(arf1_twisted_loops),
        "sylvester_resultants_ms2": np.array(sylvester_resultants_ms2, dtype=np.float64),
        "residual_energy_ratios": np.array(residual_energy_ratios, dtype=np.float64),
        "post_surgery_signal": post_signal,
        "n1_reference_resolution_rate": MILGRAM_M1_N1_RESOLUTION_RATE,
    }


def chiral_chern_orientation(
    waveforms: np.ndarray,
    center_index: int | None = None,
) -> dict[str, Any]:
    r"""Milgram Operator M2: First Chern Class :math:`c_1 \in \{\pm 1\}` and Stiefel-Whitney Class :math:`w_1 \in \{0, 1\}`.

    Classifies each extracellular spike trajectory :math:`\gamma(t) = (q(t), \dot{q}(t), \ddot{q}(t))` by:
      - :math:`c_1 = -1` for clockwise somatic :math:`\mathrm{Na}^+/\mathrm{K}^+` depolarization trough orbits,
      - :math:`c_1 = +1` for inverted/axonal capacitive return crest orbits,
      - :math:`w_1 \in \{0, 1\} \pmod 2` from the parity of Frenet-Darboux torsion :math:`\tau(t)` sign flips
        across the depolarization-repolarization core (:math:`w_1 = 0` orientable ribbon vs
        :math:`w_1 = 1` non-orientable Möbius axonal twist).
    """
    arr = np.asarray(waveforms, dtype=np.float64)
    squeeze_out = False
    if arr.ndim == 1:
        arr = arr[None, :]
        squeeze_out = True

    m_spikes, n_samples = arr.shape
    t0 = n_samples // 2 if center_index is None else int(center_index)
    core_lo = max(0, t0 - 6)
    core_hi = min(n_samples, t0 + 7)

    c1_classes = np.empty(m_spikes, dtype=np.int64)
    w1_classes = np.empty(m_spikes, dtype=np.int64)
    symplectic_areas = np.empty(m_spikes, dtype=np.float64)
    total_curvatures = np.empty(m_spikes, dtype=np.float64)
    total_torsions = np.empty(m_spikes, dtype=np.float64)

    for idx in range(m_spikes):
        q = arr[idx]
        scale = float(np.median(np.abs(q)) / 0.67448975) + 1e-12
        qn = q / scale
        dq = np.gradient(qn)
        ddq = np.gradient(dq)
        dddq = np.gradient(ddq)

        signed_area = 0.5 * float(np.sum(qn[:-1] * dq[1:] - qn[1:] * dq[:-1]))
        symplectic_areas[idx] = abs(signed_area)
        c1_classes[idx] = -1 if q[t0] <= 0.0 else 1

        v_vec = np.stack([dq, ddq, dddq], axis=1)
        a_vec = np.gradient(v_vec, axis=0)
        j_vec = np.gradient(a_vec, axis=0)
        cross_va = np.cross(v_vec, a_vec)
        norm_va = np.linalg.norm(cross_va, axis=1) + 1e-12
        norm_v = np.linalg.norm(v_vec, axis=1) + 1e-12

        kappa = norm_va / (norm_v ** 3)
        det_vaj = np.sum(cross_va * j_vec, axis=1)
        tau = det_vaj / (norm_va ** 2)

        total_curvatures[idx] = float(np.sum(kappa))
        total_torsions[idx] = float(np.sum(np.abs(tau)))

        tau_core = tau[core_lo:core_hi]
        n_flips = int(np.sum(tau_core[:-1] * tau_core[1:] < 0))
        w1_classes[idx] = n_flips % 2

    if squeeze_out:
        return {
            "c1": int(c1_classes[0]),
            "w1": int(w1_classes[0]),
            "symplectic_area": float(symplectic_areas[0]),
            "total_curvature": float(total_curvatures[0]),
            "total_torsion": float(total_torsions[0]),
        }
    return {
        "c1": c1_classes,
        "w1": w1_classes,
        "symplectic_area": symplectic_areas,
        "total_curvature": total_curvatures,
        "total_torsion": total_torsions,
        "somatic_c1_neg1_count": int(np.sum(c1_classes == -1)),
        "axonal_c1_pos1_count": int(np.sum(c1_classes == 1)),
        "orientable_w1_0_count": int(np.sum(w1_classes == 0)),
        "non_orientable_w1_1_count": int(np.sum(w1_classes == 1)),
    }


def _detect_refractory_peaks(sig: np.ndarray, thr_mult: float = 4.5, refrac: int = 20) -> np.ndarray:
    mad = float(np.median(np.abs(sig)) / 0.67448975) + 1e-12
    cand = np.where(np.abs(sig) > thr_mult * mad)[0]
    if len(cand) == 0:
        return np.empty(0, dtype=np.int64)
    times: list[int] = []
    last_t = -100
    n = len(sig)
    for c_t in cand:
        if c_t - last_t >= refrac:
            w_end = min(n, c_t + refrac)
            loc = int(c_t + np.argmax(np.abs(sig[c_t:w_end])))
            times.append(loc)
            last_t = loc
    return np.array(times, dtype=np.int64)


def ranicki_milgram_dyad_signature(
    spk_a: np.ndarray,
    spk_b: np.ndarray,
    lfp_a: np.ndarray | None = None,
    lfp_b: np.ndarray | None = None,
    fs: float = 19531.25,
    arf1_enrichment_weight: float = RANICKI_ARF1_ENRICHMENT_WEIGHT,
) -> dict[str, Any]:
    r"""Milgram Operator M4: Ranicki-Milgram :math:`\operatorname{Arf} \in \{0, 1\}` Dyad Signature & Weighting.

    Decomposes a synchronous electrode dyad :math:`(A, B)` into common-mode
    :math:`V_{\mathrm{cm}} = \frac{1}{2}(V_A + V_B)` and differential-mode
    :math:`V_{\mathrm{diff}} = V_A - V_B`, and computes:
      1. Sub-sample axonal propagation lag :math:`\Delta\tau_{\mathrm{spk}}` (in :math:`\mu\mathrm{s}`).
      2. Skew-symmetric symplectic Pfaffian flux :math:`\operatorname{Pf}(\Omega) = \frac{1}{2}\langle V_A\dot{V}_B - V_B\dot{V}_A \rangle`.
      3. Symmetric cross-band concordance signature :math:`(n_+, n_-)`.
      4. Emergent hidden differential spikes in :math:`V_{\mathrm{diff}}` masked in monopolar channels.
      5. :math:`\mathbb{Z}_2` Kervaire-Milgram invariant :math:`\operatorname{Arf}(q_{\mathrm{dyad}}) \in \{0, 1\}`
         and dyad importance weight (``6.75x`` hidden-spike enrichment on :math:`\operatorname{Arf} = 1` twisted dyads).
    """
    sa_raw = np.asarray(spk_a, dtype=np.float64)
    sb_raw = np.asarray(spk_b, dtype=np.float64)
    if sa_raw.shape != sb_raw.shape or sa_raw.ndim != 1:
        raise ValueError("spk_a and spk_b must be 1D arrays of identical length")

    nf = len(sa_raw)
    dt_ms = 1000.0 / fs
    dt_us = 1e6 / fs

    sa = (sa_raw - np.mean(sa_raw)) / (np.std(sa_raw) + 1e-12)
    sb = (sb_raw - np.mean(sb_raw)) / (np.std(sb_raw) + 1e-12)

    if lfp_a is None or lfp_b is None:
        # Low-pass moving average proxy if explicit LFP is not provided
        kernel = np.ones(min(64, max(4, nf // 8)), dtype=np.float64)
        kernel /= len(kernel)
        la_raw = np.convolve(sa_raw, kernel, mode="same")
        lb_raw = np.convolve(sb_raw, kernel, mode="same")
    else:
        la_raw = np.asarray(lfp_a, dtype=np.float64)
        lb_raw = np.asarray(lfp_b, dtype=np.float64)

    la = (la_raw - np.mean(la_raw)) / (np.std(la_raw) + 1e-12)
    lb = (lb_raw - np.mean(lb_raw)) / (np.std(lb_raw) + 1e-12)

    r_spk = float(np.dot(sa, sb) / nf)
    r_lfp = float(np.dot(la, lb) / nf)

    dsa = np.gradient(sa)
    dsb = np.gradient(sb)
    dla = np.gradient(la)
    dlb = np.gradient(lb)

    pf_spk = 0.5 * float(np.mean(sa * dsb - sb * dsa)) / dt_ms
    pf_lfp = 0.5 * float(np.mean(la * dlb - lb * dla)) / dt_ms

    c_sa_lb = float(np.dot(sa, lb) / nf)
    c_la_sb = float(np.dot(la, sb) / nf)
    m_concord = np.array(
        [
            [r_lfp, 0.5 * (c_sa_lb + c_la_sb)],
            [0.5 * (c_sa_lb + c_la_sb), r_spk],
        ],
        dtype=np.float64,
    )
    eigs_concord = np.linalg.eigvalsh(m_concord)
    n_pos = int(np.sum(eigs_concord > 0))
    n_neg = int(np.sum(eigs_concord < 0))
    sig_str = f"({n_pos},{n_neg})"

    r_m1 = float(np.dot(sa[1:], sb[:-1]) / max(nf - 1, 1))
    r_0 = r_spk
    r_p1 = float(np.dot(sa[:-1], sb[1:]) / max(nf - 1, 1))
    denom = r_m1 - 2.0 * r_0 + r_p1
    delta_sub = 0.5 * (r_m1 - r_p1) / denom if abs(denom) > 1e-12 else 0.0
    delta_tau_spk_us = float(np.clip(delta_sub, -1.5, 1.5)) * dt_us

    v_cm = 0.5 * (sa_raw + sb_raw)
    v_diff = sa_raw - sb_raw

    st_a = _detect_refractory_peaks(sa_raw)
    st_b = _detect_refractory_peaks(sb_raw)
    st_diff = _detect_refractory_peaks(v_diff)
    union_ab = np.union1d(st_a, st_b)

    hidden_cnt = 0
    for dt_idx in st_diff:
        if len(union_ab) == 0 or np.min(np.abs(union_ab - dt_idx)) > 5:
            hidden_cnt += 1

    is_twisted_arf1 = (
        (abs(delta_tau_spk_us) >= 15.0)
        or (r_lfp * r_spk < 0.0)
        or (hidden_cnt >= 5)
    )
    arf_val = 1 if is_twisted_arf1 else 0
    dyad_weight = float(arf1_enrichment_weight if arf_val == 1 else 1.0)
    wall_norm = (pf_spk / 0.05) ** 2 + float(arf_val)

    return {
        "v_cm": v_cm,
        "v_diff": v_diff,
        "r_spk": r_spk,
        "r_lfp": r_lfp,
        "delta_tau_spk_us": delta_tau_spk_us,
        "pfaffian_flux_spk_ms_inv": pf_spk,
        "pfaffian_flux_lfp_ms_inv": pf_lfp,
        "concordance_signature": sig_str,
        "arf_invariant": int(arf_val),
        "dyad_weight": dyad_weight,
        "emergent_hidden_spikes_in_vdiff": int(hidden_cnt),
        "wall_obstruction_norm": float(wall_norm),
    }


class NeuroleanNHFTransform:
    r"""Composite transform upgrading ``SE(2)`` Spike-Jet & Dyadic Block-Masking with the
    Neurolean & Neural Hamiltonian-Fourier (NHF) Calculus (``DOI: 10.5281/zenodo.23489707``).

    Computes and attaches:
      - NHF Action-Angle canonical coordinates :math:`(I, \theta_{\mathrm{NH}})` and orthonormal basis
        coefficients from :func:`nhf_action_angle_coordinates`.
      - :math:`\mathrm{SE}(2)` Frenet-Serret 3-jet features :math:`(V, \dot{V}, \ddot{V}, \kappa)`
        from :func:`se2_frenet_spike_jet`.
      - Milgram Operator M2 chiral Chern orientation :math:`c_1 \in \{\pm 1\}` and Stiefel-Whitney
        twist :math:`w_1 \in \{0, 1\}` from :func:`chiral_chern_orientation`.
      - Optional Milgram Operator M1 Dehn surgery doublet resolution and Ranicki-Milgram
        :math:`\operatorname{Arf} \in \{0, 1\}` dyadic block-masking & weighting.
    """

    def __init__(
        self,
        waveforms_attribute: str = "waveforms",
        omega0_sq: float = NHF_DEFAULT_OMEGA0_SQ,
        alpha_na: float = NHF_DEFAULT_ALPHA_NA,
        beta_k: float = NHF_DEFAULT_BETA_K,
        k_modes: int = 1,
        fs: float = 19531.25,
    ):
        self.waveforms_attr = waveforms_attribute
        self.omega0_sq = omega0_sq
        self.alpha_na = alpha_na
        self.beta_k = beta_k
        self.k_modes = k_modes
        self.fs = fs

    def __call__(self, data: Any) -> Any:
        if isinstance(data, dict):
            waveforms = np.asarray(data[self.waveforms_attr], dtype=np.float64)
            nhf_out = nhf_action_angle_coordinates(
                waveforms,
                omega0_sq=self.omega0_sq,
                alpha_na=self.alpha_na,
                beta_k=self.beta_k,
                k_modes=self.k_modes,
            )
            jet_out = se2_frenet_spike_jet(waveforms, dt=1000.0 / self.fs)
            chiral_out = chiral_chern_orientation(waveforms)
            data["nhf"] = nhf_out
            data["se2_jet"] = jet_out
            data["milgram_chiral"] = chiral_out
            if "dyad_pair" in data:
                spk_a, spk_b = data["dyad_pair"]
                data["ranicki_dyad"] = ranicki_milgram_dyad_signature(
                    spk_a, spk_b, fs=self.fs
                )
            if "signal" in data and "spike_times" in data:
                data["milgram_dehn"] = dehn_surgery_resolve_doublets(
                    data["signal"], data["spike_times"], fs=self.fs
                )
            return data

        waveforms = getattr(data, self.waveforms_attr)
        nhf_out = nhf_action_angle_coordinates(
            waveforms,
            omega0_sq=self.omega0_sq,
            alpha_na=self.alpha_na,
            beta_k=self.beta_k,
            k_modes=self.k_modes,
        )
        jet_out = se2_frenet_spike_jet(waveforms, dt=1000.0 / self.fs)
        chiral_out = chiral_chern_orientation(waveforms)
        setattr(data, f"{self.waveforms_attr}_nhf", nhf_out)
        setattr(data, f"{self.waveforms_attr}_se2_jet", jet_out)
        setattr(data, f"{self.waveforms_attr}_chiral", chiral_out)
        return data
