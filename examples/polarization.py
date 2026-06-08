from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

import matplotlib
import numpy as np

# Allow running this file directly from examples/ or project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Save figures to files without opening GUI windows.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from radar_learning_sim import (
    RadarConfig,
    Target,
    simulate_raw_datacube,
    matched_filter_range,
    doppler_process,
    range_doppler_power,
    db10_power,
)


EPS = 1e-12
OUTPUT_DIR = Path("outputs/polarization")


# ============================================================
# Jones vector utilities
# ============================================================


def normalize_jones(v: Sequence[complex]) -> np.ndarray:
    """Return a unit-norm Jones vector [E_H, E_V]."""
    v = np.asarray(v, dtype=np.complex128)
    n = np.linalg.norm(v)
    if n <= EPS:
        raise ValueError("Cannot normalize a zero Jones vector.")
    return v / n


def jones_linear(angle_deg: float) -> np.ndarray:
    """Linear polarization: 0 deg = H, 90 deg = V, 45 deg = equal H/V."""
    angle_rad = np.deg2rad(angle_deg)
    return np.array([np.cos(angle_rad), np.sin(angle_rad)], dtype=np.complex128)


def jones_circular(hand: str = "R") -> np.ndarray:
    """Circular polarization using R=[1,-j]/sqrt(2), L=[1,+j]/sqrt(2)."""
    hand = hand.upper()
    if hand == "R":
        return normalize_jones([1.0, -1j])
    if hand == "L":
        return normalize_jones([1.0, 1j])
    raise ValueError("hand must be 'R' or 'L'.")


def jones_elliptical(
    amp_ratio_v_over_h: float = 0.5,
    phase_v_minus_h_deg: float = 60,
) -> np.ndarray:
    """Elliptical polarization with E_H=1 and complex E_V set by ratio/phase."""
    phase_rad = np.deg2rad(phase_v_minus_h_deg)
    return normalize_jones([1.0, amp_ratio_v_over_h * np.exp(1j * phase_rad)])


def polarization_gain(tx_pol: np.ndarray, rx_pol: np.ndarray, S: np.ndarray) -> complex:
    """
    Compute y = p_rx^H S p_tx.

    This complex scalar is the external polarization layer. The original radar
    simulator still handles range delay, Doppler phase, LFM matched filtering,
    array steering, and noise.
    """
    tx_pol = normalize_jones(tx_pol)
    rx_pol = normalize_jones(rx_pol)
    S = np.asarray(S, dtype=np.complex128)
    if S.shape != (2, 2):
        raise ValueError("S must be a 2x2 polarimetric scattering matrix.")
    E_scattered = S @ tx_pol
    return np.vdot(rx_pol, E_scattered)


def stokes_parameters(E: np.ndarray) -> np.ndarray:
    """Return [S0, S1, S2, S3] from Jones vector E=[E_H, E_V]."""
    E = np.asarray(E, dtype=np.complex128)
    EH, EV = E[0], E[1]
    S0 = np.abs(EH) ** 2 + np.abs(EV) ** 2
    S1 = np.abs(EH) ** 2 - np.abs(EV) ** 2
    S2 = 2.0 * np.real(EH * np.conj(EV))
    S3 = -2.0 * np.imag(EH * np.conj(EV))
    return np.array([S0, S1, S2, S3], dtype=float)


def estimate_polarization_from_dual_rx(E: np.ndarray) -> dict:
    """
    Estimate polarization properties from dual-pol H/V complex receiver values.

    A single-pol radar gives only one complex scalar. With both H and V channels,
    we can estimate amplitude ratio, relative phase, normalized Stokes vector,
    orientation angle psi, and ellipticity angle chi.
    """
    E = np.asarray(E, dtype=np.complex128)
    EH, EV = E[0], E[1]
    amp_H = np.abs(EH)
    amp_V = np.abs(EV)
    amp_ratio = amp_V / max(amp_H, EPS)

    phase_diff_rad = np.angle(np.exp(1j * (np.angle(EV) - np.angle(EH))))
    S0, S1, S2, S3 = stokes_parameters(E)
    if S0 <= EPS:
        normalized_stokes = np.array([0.0, 0.0, 0.0, 0.0])
        psi = 0.0
        chi = 0.0
    else:
        normalized_stokes = np.array([1.0, S1 / S0, S2 / S0, S3 / S0])
        psi = 0.5 * np.arctan2(S2, S1)
        chi = 0.5 * np.arcsin(np.clip(S3 / S0, -1.0, 1.0))

    return {
        "amp_H": amp_H,
        "amp_V": amp_V,
        "amp_ratio_V_over_H": amp_ratio,
        "phase_diff_V_minus_H_deg": np.rad2deg(phase_diff_rad),
        "orientation_psi_deg": np.rad2deg(psi),
        "ellipticity_chi_deg": np.rad2deg(chi),
        "normalized_stokes": normalized_stokes,
    }


# ============================================================
# Polarized target wrapper
# ============================================================


@dataclass
class PolarizedTarget:
    base_target: Target
    scattering_matrix: np.ndarray
    name: str = ""


def make_effective_target(
    pol_tgt: PolarizedTarget,
    tx_pol: np.ndarray,
    rx_pol: np.ndarray,
) -> Target:
    """
    Convert a polarimetric target into a normal simulator Target.

    The sealed simulator does not know about polarization, so this creates a
    fresh Target with amplitude A * (p_rx^H S p_tx). No base target is mutated.
    """
    base = pol_tgt.base_target
    g = polarization_gain(tx_pol, rx_pol, pol_tgt.scattering_matrix)
    return Target(
        range_m=base.range_m,
        velocity_mps=base.velocity_mps,
        angle_deg=base.angle_deg,
        amplitude=base.amplitude * g,
        acceleration_mps2=base.acceleration_mps2,
        micro_range_amp_m=base.micro_range_amp_m,
        micro_freq_hz=base.micro_freq_hz,
        micro_phase_rad=base.micro_phase_rad,
        name=pol_tgt.name or base.name,
    )


def make_effective_targets(
    pol_targets: Sequence[PolarizedTarget],
    tx_pol: np.ndarray,
    rx_pol: np.ndarray,
) -> list[Target]:
    return [make_effective_target(t, tx_pol, rx_pol) for t in pol_targets]


# ============================================================
# Original radar pipeline and plotting
# ============================================================


def run_pipeline(cfg: RadarConfig, targets: Sequence[Target]):
    raw, meta = simulate_raw_datacube(cfg, targets)
    rc, range_axis = matched_filter_range(raw, cfg)
    rd, fd_axis, vel_axis = doppler_process(rc, cfg)
    power = range_doppler_power(rd, integrate_rx=True)
    return raw, rc, rd, power, range_axis, fd_axis, vel_axis, meta


def plot_range_doppler(
    power_2d: np.ndarray,
    range_axis: np.ndarray,
    vel_axis: np.ndarray,
    title: str,
    range_lim: tuple[float, float] | None = None,
):
    """Plot a [range, doppler] power map in dB."""
    plt.figure()
    extent = [vel_axis[0], vel_axis[-1], range_axis[0], range_axis[-1]]
    im = plt.imshow(
        db10_power(power_2d + EPS),
        extent=extent,
        aspect="auto",
        origin="lower",
    )
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Range (m)")
    plt.title(title)
    plt.colorbar(im, label="Power (dB)")
    if range_lim is not None:
        plt.ylim(range_lim)


def _safe_filename(text: str) -> str:
    """Turn a plot title into a simple PNG filename."""
    text = text.lower().replace("+", "plus")
    chars = []
    for ch in text:
        if ch.isalnum():
            chars.append(ch)
        elif ch in {" ", "-", "_", "/"}:
            chars.append("_")
    name = "".join(chars).strip("_")
    while "__" in name:
        name = name.replace("__", "_")
    return name or "figure"


def save_all_figures(outdir: Path = OUTPUT_DIR):
    """Save every open Matplotlib figure and close it."""
    outdir.mkdir(parents=True, exist_ok=True)
    for index, fig_num in enumerate(plt.get_fignums(), start=1):
        fig = plt.figure(fig_num)
        title = ""
        if fig.axes:
            title = fig.axes[0].get_title()
        filename = f"{index:02d}_{_safe_filename(title)}.png"
        fig.savefig(outdir / filename, dpi=160, bbox_inches="tight")
    plt.close("all")
    print(f"\nDone. Figures are in: {outdir.resolve()}")


def _default_demo_config(num_cpi: int = 2, noise_power: float = 1e-4) -> RadarConfig:
    return RadarConfig(
        fc_hz=10e9,
        fs_hz=20e6,
        prf_hz=5e3,
        pulse_width_s=5e-6,
        bandwidth_hz=10e6,
        num_fast_time=1024,
        num_pulses=64,
        num_cpi=num_cpi,
        num_rx=8,
        noise_power=noise_power,
        seed=7,
    )


# ============================================================
# Demo A: single-pol projection only
# ============================================================


def demo_single_pol_projection():
    """A single-pol H receiver only sees the H projection of the field."""
    rx_H = jones_linear(0)
    angles = np.linspace(0.0, 180.0, 361)
    power = []

    for angle in angles:
        E = jones_linear(angle)
        y = np.vdot(rx_H, E)
        power.append(np.abs(y) ** 2)

    power = np.asarray(power)

    plt.figure()
    plt.plot(angles, power)
    plt.xlabel("Incoming linear polarization angle (deg)")
    plt.ylabel("Received H-channel power")
    plt.title("Single-pol H receiver sees only projection")
    plt.grid(True)

    plt.figure()
    plt.plot(angles, db10_power(power + EPS))
    plt.xlabel("Incoming linear polarization angle (deg)")
    plt.ylabel("Received H-channel power (dB)")
    plt.title("Single-pol H receiver sees only projection")
    plt.grid(True)


# ============================================================
# Demo B: single-pol ambiguity
# ============================================================


def demo_single_pol_ambiguity():
    """
    A single H receiver only sees yH, so it loses information about yV.

    The printed dual-Rx estimates are shown to emphasize what extra information
    becomes available only when both H and V complex channels are measured.
    """
    candidates = [
        np.array([1.0, 0.0], dtype=np.complex128),
        normalize_jones([1.0, 0.5]),
        normalize_jones([1.0, 0.5j]),
        normalize_jones([1.0, np.exp(1j * np.deg2rad(70))]),
    ]

    print("\nDemo B: single-pol ambiguity")
    for i, E in enumerate(candidates, start=1):
        yH = np.vdot(jones_linear(0), E)
        yV = np.vdot(jones_linear(90), E)
        est = estimate_polarization_from_dual_rx(np.array([yH, yV]))
        print(f"\nCandidate {i}")
        print(f"  E = {E}")
        print(f"  H receiver yH = {yH:.4g}")
        print(f"  V receiver yV = {yV:.4g}")
        print(f"  phase V-H = {est['phase_diff_V_minus_H_deg']:.2f} deg")
        print(f"  orientation psi = {est['orientation_psi_deg']:.2f} deg")
        print(f"  ellipticity chi = {est['ellipticity_chi_deg']:.2f} deg")


# ============================================================
# Demo C: same radar target, different Tx/Rx polarization
# ============================================================


def demo_same_target_different_polarization():
    """Same range/velocity/angle target, different amplitude from p_rx^H S p_tx."""
    cfg = _default_demo_config(num_cpi=2, noise_power=1e-4)
    base = Target(
        range_m=2800.0,
        velocity_mps=18.0,
        angle_deg=12.0,
        amplitude=1.0 + 0j,
        name="same physical target",
    )
    S = np.array([[1.0, 0.0], [0.0, 0.2]], dtype=np.complex128)
    pol_tgt = PolarizedTarget(base, S)

    cases = [
        ("Tx H / Rx H", jones_linear(0), jones_linear(0)),
        ("Tx H / Rx V", jones_linear(0), jones_linear(90)),
        ("Tx V / Rx H", jones_linear(90), jones_linear(0)),
        ("Tx V / Rx V", jones_linear(90), jones_linear(90)),
    ]

    print("\nDemo C: same target, different Tx/Rx polarization")
    for title, tx_pol, rx_pol in cases:
        g = polarization_gain(tx_pol, rx_pol, S)
        print(f"  {title}: polarization gain = {g:.4g}, |gain|^2 = {np.abs(g) ** 2:.4g}")
        effective = make_effective_target(pol_tgt, tx_pol, rx_pol)
        _, _, _, power, range_axis, _, vel_axis, _ = run_pipeline(cfg, [effective])
        plot_range_doppler(power[0], range_axis, vel_axis, title, range_lim=(2400, 3600))


# ============================================================
# Demo D: H and V receive channels simulated by two separate runs
# ============================================================


def _pack_pipeline_result(result_tuple: tuple) -> dict:
    raw, rc, rd, power, range_axis, fd_axis, vel_axis, meta = result_tuple
    return {
        "raw": raw,
        "rc": rc,
        "rd": rd,
        "power": power,
        "range_axis": range_axis,
        "fd_axis": fd_axis,
        "vel_axis": vel_axis,
        "meta": meta,
    }


def simulate_dual_rx_externally(
    cfg: RadarConfig,
    pol_targets: Sequence[PolarizedTarget],
    tx_pol: np.ndarray,
) -> tuple[dict, dict]:
    """
    Approximate dual-pol receive by running the sealed single-pol simulator twice.

    One run uses Rx H and one run uses Rx V. This produces two scalar receive
    channels, while the original simulator continues to generate the radar
    datacube physics for each effective target list.
    """
    effective_targets_H = make_effective_targets(pol_targets, tx_pol, jones_linear(0))
    effective_targets_V = make_effective_targets(pol_targets, tx_pol, jones_linear(90))
    result_H = _pack_pipeline_result(run_pipeline(cfg, effective_targets_H))
    result_V = _pack_pipeline_result(run_pipeline(cfg, effective_targets_V))
    return result_H, result_V


def demo_external_dual_pol_range_doppler():
    """Most important demo: compare H-Rx, V-Rx, total power, and V/H ratio."""
    cfg = _default_demo_config(num_cpi=4, noise_power=1e-5)
    tx_pol = jones_linear(0)

    pol_targets = [
        PolarizedTarget(
            Target(2800.0, 18.0, 12.0, 1.0 + 0j, name="co-pol strong target"),
            np.array([[1.0, 0.0], [0.0, 0.1]], dtype=np.complex128),
        ),
        PolarizedTarget(
            Target(3300.0, -12.0, -20.0, 0.8 + 0j, name="cross-pol strong target"),
            np.array([[0.1, 0.0], [0.9, 0.1]], dtype=np.complex128),
        ),
        PolarizedTarget(
            Target(3100.0, 3.5, 28.0, 0.6 + 0j, name="mixed target"),
            np.array(
                [
                    [0.7, 0.2j],
                    [0.4 * np.exp(1j * np.deg2rad(40)), 0.2],
                ],
                dtype=np.complex128,
            ),
        ),
    ]

    print("\nDemo D: external dual-pol range-Doppler")
    print("For Tx H, expected H gain is S_HH and expected V gain is S_VH.")
    for t in pol_targets:
        base = t.base_target
        S = t.scattering_matrix
        print(f"\n  {base.name}")
        print(f"    range={base.range_m:.1f} m, velocity={base.velocity_mps:.1f} m/s, angle={base.angle_deg:.1f} deg")
        print(f"    S=\n{S}")
        print(f"    expected H gain S_HH = {S[0, 0]:.4g}")
        print(f"    expected V gain S_VH = {S[1, 0]:.4g}")

    result_H, result_V = simulate_dual_rx_externally(cfg, pol_targets, tx_pol)
    power_H = result_H["power"][0]
    power_V = result_V["power"][0]
    power_total = power_H + power_V
    ratio_db = 10.0 * np.log10((power_V + EPS) / (power_H + EPS))
    range_axis = result_H["range_axis"]
    vel_axis = result_H["vel_axis"]

    plot_range_doppler(power_H, range_axis, vel_axis, "H-Rx range-Doppler power", range_lim=(2400, 3800))
    plot_range_doppler(power_V, range_axis, vel_axis, "V-Rx range-Doppler power", range_lim=(2400, 3800))
    plot_range_doppler(power_total, range_axis, vel_axis, "Total H+V range-Doppler power", range_lim=(2400, 3800))

    plt.figure()
    extent = [vel_axis[0], vel_axis[-1], range_axis[0], range_axis[-1]]
    im = plt.imshow(ratio_db, extent=extent, aspect="auto", origin="lower", vmin=-40, vmax=40)
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Range (m)")
    plt.title("Cross-pol ratio V/H in dB")
    plt.ylim((2400, 3800))
    plt.colorbar(im, label="V/H ratio (dB)")


# ============================================================
# Demo E: rotating radar polarization
# ============================================================


def demo_rotating_tx_rx_polarization():
    """
    Rotate Tx and Rx linear polarization together.

    For monostatic radar with Tx/Rx rotated together, y(theta)=p(theta)^H S p(theta).
    This can be more complicated than a simple cos^2 projection curve.
    """
    cfg = _default_demo_config(num_cpi=1, noise_power=0.0)
    base = Target(
        range_m=2800.0,
        velocity_mps=18.0,
        angle_deg=0.0,
        amplitude=1.0 + 0j,
        name="rotating-pol target",
    )
    S = np.array(
        [
            [1.0, 0.3 * np.exp(1j * np.deg2rad(30))],
            [0.1 * np.exp(-1j * np.deg2rad(20)), 0.4],
        ],
        dtype=np.complex128,
    )
    pol_tgt = PolarizedTarget(base, S)

    theta = np.linspace(0.0, 180.0, 91)
    sim_max_power = []
    theoretical_gain_power = []

    for angle in theta:
        p = jones_linear(angle)
        effective = make_effective_target(pol_tgt, p, p)
        _, _, _, power, _, _, _, _ = run_pipeline(cfg, [effective])
        sim_max_power.append(np.max(power[0]))
        theoretical_gain_power.append(np.abs(polarization_gain(p, p, S)) ** 2)

    sim_max_power = np.asarray(sim_max_power)
    theoretical_gain_power = np.asarray(theoretical_gain_power)

    plt.figure()
    plt.plot(theta, db10_power(sim_max_power + EPS), label="Simulated max RD power")
    plt.plot(theta, db10_power(theoretical_gain_power + EPS), label="Theoretical |p^H S p|^2")
    plt.xlabel("Rotated Tx/Rx linear polarization angle (deg)")
    plt.ylabel("Power (dB, arbitrary offsets)")
    plt.title("Rotating Tx/Rx polarization")
    plt.grid(True)
    plt.legend()


# ============================================================
# Demo F: drone-like time-varying polarization toy model
# ============================================================


def demo_drone_like_time_varying_cross_pol():
    """
    Toy drone-like example.

    This is not a physically exact drone model. It uses the existing simulator's
    micro-range field to make a rotor-like component, illustrating how a
    time-varying scattering matrix S(t) could create different H/V signatures.
    """
    cfg = _default_demo_config(num_cpi=4, noise_power=1e-5)
    tx_pol = jones_linear(0)

    body = PolarizedTarget(
        Target(
            range_m=2800.0,
            velocity_mps=0.0,
            angle_deg=10.0,
            amplitude=1.0 + 0j,
            name="drone body",
        ),
        np.array([[1.0, 0.0], [0.0, 0.1]], dtype=np.complex128),
    )
    rotor = PolarizedTarget(
        Target(
            range_m=2800.0,
            velocity_mps=0.0,
            angle_deg=10.0,
            amplitude=0.3 + 0j,
            micro_range_amp_m=0.01,
            micro_freq_hz=80.0,
            name="rotor-like component",
        ),
        np.array([[0.2, 0.0], [0.8, 0.0]], dtype=np.complex128),
    )

    result_H, result_V = simulate_dual_rx_externally(cfg, [body, rotor], tx_pol)
    range_axis = result_H["range_axis"]
    vel_axis = result_H["vel_axis"]

    plot_range_doppler(
        result_H["power"][0],
        range_axis,
        vel_axis,
        "Toy drone-like target: H-Rx range-Doppler",
        range_lim=(2400, 3200),
    )
    plot_range_doppler(
        result_V["power"][0],
        range_axis,
        vel_axis,
        "Toy drone-like target: V-Rx range-Doppler",
        range_lim=(2400, 3200),
    )


def main():
    demo_single_pol_projection()
    demo_single_pol_ambiguity()
    demo_same_target_different_polarization()
    demo_external_dual_pol_range_doppler()
    demo_rotating_tx_rx_polarization()
    demo_drone_like_time_varying_cross_pol()
    save_all_figures()


if __name__ == "__main__":
    main()
