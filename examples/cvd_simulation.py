"""
Micro-Doppler CVD experiment for a drone-like target.

Run from the project root:

    python examples/cvd_simulation.py

The script writes STFT/CVD comparison figures to ./outputs/cvd_simulation.

Idea
----
1. Simulate raw pulsed-LFM radar data for a drone body plus blade scatterers.
2. Range-compress and extract the target range gate as one slow-time signal.
3. Make a time-velocity map with STFT.
4. Make a CVD map by taking an FFT along the STFT time axis at each velocity bin.
5. Compare the CVD with and without known body-velocity compensation.

This is a teaching experiment, not a high-fidelity rotorcraft EM model. The
blade scatterers are modeled as small sinusoidal range motions riding on the
same body translation as the drone.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

# Allow running this file directly from examples/ or project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from radar_learning_sim import (  # noqa: E402
    RadarConfig,
    Target,
    db10_power,
    matched_filter_range,
    simulate_raw_datacube,
)


@dataclass(frozen=True)
class Scenario:
    name: str
    body_velocity_mps: float
    aspect_deg: float
    noise_power: float = 2e-4
    description: str = ""


def make_drone_targets(
    body_velocity_mps: float,
    aspect_deg: float,
    blade_freq_hz: float,
) -> list[Target]:
    """Return one body scatterer plus three blade scatterers.

    aspect_deg is a simple observation-aspect knob. At broadside-like aspects,
    the rotor radial motion is weak, so the micro-Doppler range modulation is
    scaled down by cos(aspect). This lets us test the angle-sensitivity concern.
    """
    base_range_m = 2200.0
    micro_scale = abs(np.cos(np.deg2rad(aspect_deg)))
    micro_amp_m = 0.030 * micro_scale

    targets = [
        Target(
            range_m=base_range_m,
            velocity_mps=body_velocity_mps,
            angle_deg=0.0,
            amplitude=1.0,
            name="drone body",
        )
    ]

    phases = [0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0]
    for i, phase in enumerate(phases, start=1):
        targets.append(
            Target(
                range_m=base_range_m,
                velocity_mps=body_velocity_mps,
                angle_deg=0.0,
                amplitude=0.24,
                micro_range_amp_m=micro_amp_m,
                micro_freq_hz=blade_freq_hz,
                micro_phase_rad=phase,
                name=f"blade scatterer {i}",
            )
        )
    return targets


def extract_range_gate_signal(
    range_cube: np.ndarray,
    range_axis_m: np.ndarray,
    expected_range_m: float,
    gate_half_width: int = 2,
) -> tuple[np.ndarray, int]:
    """Average a few range bins and receive channels into one slow-time signal."""
    center = int(np.argmin(np.abs(range_axis_m - expected_range_m)))
    lo = max(0, center - gate_half_width)
    hi = min(range_cube.shape[2], center + gate_half_width + 1)

    # [CPI, pulse, range, rx] -> [CPI*pulse]
    slow = np.mean(range_cube[:, :, lo:hi, :], axis=(2, 3)).reshape(-1)
    return slow, center


def compensate_body_velocity(
    slow_signal: np.ndarray,
    cfg: RadarConfig,
    body_velocity_mps: float,
) -> np.ndarray:
    """Shift the bulk Doppler of a known body radial velocity to 0 m/s."""
    t = np.arange(len(slow_signal)) * cfg.pri_s
    fd_body_hz = 2.0 * body_velocity_mps / cfg.wavelength_m
    return slow_signal * np.exp(-1j * 2.0 * np.pi * fd_body_hz * t)


def stft_time_velocity(
    slow_signal: np.ndarray,
    cfg: RadarConfig,
    window_len: int = 128,
    hop: int = 8,
    nfft: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """STFT over slow time, returned as power[time, velocity]."""
    if len(slow_signal) < window_len:
        raise ValueError("slow_signal is shorter than one STFT window")

    window = np.hanning(window_len)
    starts = np.arange(0, len(slow_signal) - window_len + 1, hop)
    spec = np.empty((len(starts), nfft), dtype=np.complex128)
    for i, start in enumerate(starts):
        frame = slow_signal[start : start + window_len] * window
        spec[i] = np.fft.fftshift(np.fft.fft(frame, n=nfft))

    fd_axis_hz = np.fft.fftshift(np.fft.fftfreq(nfft, d=cfg.pri_s))
    vel_axis_mps = fd_axis_hz * cfg.wavelength_m / 2.0
    time_axis_s = (starts + window_len / 2.0) * cfg.pri_s
    return np.abs(spec) ** 2, time_axis_s, vel_axis_mps


def cvd_from_stft_power(
    stft_power: np.ndarray,
    stft_time_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """CVD: FFT along STFT time for every velocity bin.

    The time mean is removed per velocity bin so a perfectly constant body
    ridge does not dominate the cyclic-frequency map at 0 Hz.
    """
    centered = stft_power - np.mean(stft_power, axis=0, keepdims=True)
    window = np.hanning(centered.shape[0])[:, None]
    cvd = np.fft.rfft(centered * window, axis=0)
    dt = float(np.mean(np.diff(stft_time_s)))
    cyclic_freq_hz = np.fft.rfftfreq(centered.shape[0], d=dt)
    return np.abs(cvd) ** 2, cyclic_freq_hz


def local_cvd_peak(
    cvd_power: np.ndarray,
    cyclic_freq_hz: np.ndarray,
    vel_axis_mps: np.ndarray,
    target_freq_hz: float,
    freq_tolerance_hz: float = 8.0,
) -> tuple[float, float, float]:
    """Return peak frequency, velocity, and dB value near the expected blade rate."""
    freq_mask = np.abs(cyclic_freq_hz - target_freq_hz) <= freq_tolerance_hz
    if not np.any(freq_mask):
        raise ValueError("No CVD bins inside the requested frequency tolerance")
    sub = cvd_power[freq_mask]
    local_freq = cyclic_freq_hz[freq_mask]
    fi, vi = np.unravel_index(np.nanargmax(sub), sub.shape)
    peak_value_db = float(db10_power(np.array([sub[fi, vi]]))[0])
    return float(local_freq[fi]), float(vel_axis_mps[vi]), peak_value_db


def plot_stft_cvd_comparison(
    outpath: Path,
    scenario: Scenario,
    blade_freq_hz: float,
    raw_stft: np.ndarray,
    comp_stft: np.ndarray,
    stft_time_s: np.ndarray,
    vel_axis_mps: np.ndarray,
    raw_cvd: np.ndarray,
    comp_cvd: np.ndarray,
    cyclic_freq_hz: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex="col")
    fig.suptitle(
        (
            f"{scenario.name}: v_body={scenario.body_velocity_mps:+.1f} m/s, "
            f"aspect={scenario.aspect_deg:.0f} deg"
        ),
        fontsize=13,
    )

    stft_vmax = max(np.nanmax(db10_power(raw_stft)), np.nanmax(db10_power(comp_stft)))
    cvd_vmax = max(np.nanmax(db10_power(raw_cvd[1:])), np.nanmax(db10_power(comp_cvd[1:])))

    for ax, data, title in [
        (axes[0, 0], raw_stft, "STFT before body compensation"),
        (axes[0, 1], comp_stft, "STFT after body compensation"),
    ]:
        image = ax.imshow(
            db10_power(data).T,
            origin="lower",
            aspect="auto",
            extent=[stft_time_s[0], stft_time_s[-1], vel_axis_mps[0], vel_axis_mps[-1]],
            vmin=stft_vmax - 55,
            vmax=stft_vmax,
        )
        ax.set_title(title)
        ax.set_ylabel("Velocity bin (m/s)")
        ax.grid(False)
        fig.colorbar(image, ax=ax, label="Power (dB)")

    axes[0, 0].axhline(scenario.body_velocity_mps, color="white", linestyle="--", linewidth=1.0)
    axes[0, 1].axhline(0.0, color="white", linestyle="--", linewidth=1.0)

    freq_mask = cyclic_freq_hz <= 150.0
    for ax, data, title in [
        (axes[1, 0], raw_cvd, "CVD before body compensation"),
        (axes[1, 1], comp_cvd, "CVD after body compensation"),
    ]:
        image = ax.imshow(
            db10_power(data[freq_mask, :]),
            origin="lower",
            aspect="auto",
            extent=[
                vel_axis_mps[0],
                vel_axis_mps[-1],
                cyclic_freq_hz[freq_mask][0],
                cyclic_freq_hz[freq_mask][-1],
            ],
            vmin=cvd_vmax - 50,
            vmax=cvd_vmax,
        )
        ax.axhline(blade_freq_hz, color="white", linestyle="--", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("Velocity bin (m/s)")
        ax.set_ylabel("Cyclic frequency (Hz)")
        fig.colorbar(image, ax=ax, label="CVD power (dB)")

    for ax in axes[0, :]:
        ax.set_xlabel("Time (s)")

    if scenario.description:
        fig.text(0.5, 0.01, scenario.description, ha="center", fontsize=9)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def run_scenario(
    cfg: RadarConfig,
    scenario: Scenario,
    blade_freq_hz: float,
    outdir: Path,
) -> dict[str, float]:
    cfg = RadarConfig(**{**cfg.__dict__, "noise_power": scenario.noise_power})
    targets = make_drone_targets(
        body_velocity_mps=scenario.body_velocity_mps,
        aspect_deg=scenario.aspect_deg,
        blade_freq_hz=blade_freq_hz,
    )

    raw, _ = simulate_raw_datacube(cfg, targets)
    range_cube, range_axis_m = matched_filter_range(raw, cfg)
    slow_signal, range_index = extract_range_gate_signal(
        range_cube,
        range_axis_m,
        expected_range_m=targets[0].range_m,
    )
    comp_signal = compensate_body_velocity(slow_signal, cfg, scenario.body_velocity_mps)

    raw_stft, stft_time_s, vel_axis_mps = stft_time_velocity(slow_signal, cfg)
    comp_stft, _, _ = stft_time_velocity(comp_signal, cfg)
    raw_cvd, cyclic_freq_hz = cvd_from_stft_power(raw_stft, stft_time_s)
    comp_cvd, _ = cvd_from_stft_power(comp_stft, stft_time_s)

    raw_peak_f, raw_peak_v, raw_peak_db = local_cvd_peak(
        raw_cvd,
        cyclic_freq_hz,
        vel_axis_mps,
        blade_freq_hz,
    )
    comp_peak_f, comp_peak_v, comp_peak_db = local_cvd_peak(
        comp_cvd,
        cyclic_freq_hz,
        vel_axis_mps,
        blade_freq_hz,
    )

    safe_name = scenario.name.lower().replace(" ", "_").replace("/", "_")
    plot_stft_cvd_comparison(
        outdir / f"{safe_name}.png",
        scenario,
        blade_freq_hz,
        raw_stft,
        comp_stft,
        stft_time_s,
        vel_axis_mps,
        raw_cvd,
        comp_cvd,
        cyclic_freq_hz,
    )

    return {
        "range_gate_m": float(range_axis_m[range_index]),
        "raw_peak_freq_hz": raw_peak_f,
        "raw_peak_velocity_mps": raw_peak_v,
        "raw_peak_db": raw_peak_db,
        "comp_peak_freq_hz": comp_peak_f,
        "comp_peak_velocity_mps": comp_peak_v,
        "comp_peak_db": comp_peak_db,
    }


def plot_summary(outpath: Path, rows: list[tuple[Scenario, dict[str, float]]]) -> None:
    labels = [scenario.name for scenario, _ in rows]
    raw_vel = [result["raw_peak_velocity_mps"] for _, result in rows]
    comp_vel = [result["comp_peak_velocity_mps"] for _, result in rows]
    raw_db = [result["raw_peak_db"] for _, result in rows]
    comp_db = [result["comp_peak_db"] for _, result in rows]

    x = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].bar(x - width / 2, raw_vel, width, label="before compensation")
    axes[0].bar(x + width / 2, comp_vel, width, label="after compensation")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Velocity of CVD peak (m/s)")
    axes[0].set_title("Where the blade-frequency CVD peak appears")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x - width / 2, raw_db, width, label="before compensation")
    axes[1].bar(x + width / 2, comp_db, width, label="after compensation")
    axes[1].set_ylabel("CVD peak power (dB)")
    axes[1].set_title("Blade-frequency peak strength")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")

    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main() -> None:
    outdir = Path("outputs") / "cvd_simulation"
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = RadarConfig(
        fc_hz=10e9,
        fs_hz=20e6,
        prf_hz=5e3,
        pulse_width_s=5e-6,
        bandwidth_hz=10e6,
        num_fast_time=384,
        num_pulses=128,
        num_cpi=16,
        num_rx=1,
        noise_power=2e-4,
        seed=11,
    )
    blade_freq_hz = 55.0

    scenarios = [
        Scenario(
            name="hover",
            body_velocity_mps=0.0,
            aspect_deg=0.0,
            description="Hover is the easy case: there is almost no bulk Doppler to remove.",
        ),
        Scenario(
            name="approaching",
            body_velocity_mps=12.0,
            aspect_deg=0.0,
            description="Approaching motion shifts the STFT ridge; compensation should recenter it.",
        ),
        Scenario(
            name="receding",
            body_velocity_mps=-12.0,
            aspect_deg=0.0,
            description="Receding motion is the same test with the opposite Doppler sign.",
        ),
        Scenario(
            name="weak aspect",
            body_velocity_mps=12.0,
            aspect_deg=70.0,
            description="Large aspect angle weakens radial blade motion, so the CVD peak should fade.",
        ),
        Scenario(
            name="low snr",
            body_velocity_mps=12.0,
            aspect_deg=0.0,
            noise_power=2e-3,
            description="Higher noise tests whether the cyclic blade signature remains visible.",
        ),
    ]

    print("CVD micro-Doppler simulation")
    print(f"  output directory: {outdir.resolve()}")
    print(f"  blade cyclic frequency: {blade_freq_hz:.1f} Hz")
    print(f"  unambiguous velocity: +/-{cfg.velocity_axis_unambiguous_mps:.1f} m/s")
    print()

    rows: list[tuple[Scenario, dict[str, float]]] = []
    for scenario in scenarios:
        result = run_scenario(cfg, scenario, blade_freq_hz, outdir)
        rows.append((scenario, result))
        print(f"{scenario.name}")
        print(f"  range gate: {result['range_gate_m']:.1f} m")
        print(
            "  raw CVD peak near blade freq: "
            f"f={result['raw_peak_freq_hz']:.1f} Hz, "
            f"v={result['raw_peak_velocity_mps']:+.2f} m/s, "
            f"power={result['raw_peak_db']:.1f} dB"
        )
        print(
            "  compensated CVD peak:        "
            f"f={result['comp_peak_freq_hz']:.1f} Hz, "
            f"v={result['comp_peak_velocity_mps']:+.2f} m/s, "
            f"power={result['comp_peak_db']:.1f} dB"
        )
        print()

    plot_summary(outdir / "summary.png", rows)
    print("Done. Open the PNG files in outputs/cvd_simulation.")


if __name__ == "__main__":
    main()
