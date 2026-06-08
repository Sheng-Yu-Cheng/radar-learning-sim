#!/usr/bin/env python3
"""
CFS simulation: hover vs approaching UAV

This script simulates a simplified monostatic radar slow-time return from one UAV under
 two states:
1. Hovering: bulk radial velocity = 0 m/s
2. Approaching radar: bulk radial velocity = +12 m/s

Pipeline:
    synthetic complex slow-time radar signal
    -> STFT time-Doppler map
    -> CVD: FFT along time axis of STFT power for each velocity bin
    -> CFS: sum CVD over selected velocity bins

Output:
    cfs_hover_vs_approach.png

Install dependencies:
    pip install numpy scipy matplotlib

Run:
    python cfs_hover_vs_approach.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, get_window


# -----------------------------
# Radar / simulation parameters
# -----------------------------
C = 3.0e8                    # speed of light, m/s
FC = 77.0e9                  # carrier frequency, Hz
LAMBDA = C / FC              # wavelength, m
PRF = 20_000.0               # slow-time sampling frequency, Hz; high enough for 12 m/s at 77 GHz
DURATION = 1.5               # seconds
N = int(PRF * DURATION)
T = np.arange(N) / PRF

# STFT parameters. Small hop size gives enough cadence sampling rate for 55 Hz.
N_PER_SEG = 1024
N_OVERLAP = 992
N_FFT = 4096
WINDOW = get_window("hann", N_PER_SEG)

MIN_CADENCE_HZ = 20.0
MAX_CADENCE_HZ = 160.0
USE_TARGET_VELOCITY_MASK = True

RNG = np.random.default_rng(7)


def doppler_hz_from_velocity(v_mps: float) -> float:
    """Monostatic radar Doppler frequency from radial velocity."""
    return 2.0 * v_mps / LAMBDA


def velocity_from_doppler_hz(fd_hz: np.ndarray) -> np.ndarray:
    """Convert Doppler frequency to radial velocity."""
    return fd_hz * LAMBDA / 2.0


def make_uav_slow_time_signal(
    bulk_velocity_mps: float,
    blade_cadence_hz: float = 55.0,
    snr_db: float = 8.0,
    body_strength: float = 0.75,
    rotor_strength: float = 0.65,
) -> np.ndarray:
    """
    Build a simplified complex slow-time radar signal.

    Body return:
        a strong component at the bulk Doppler frequency.
    Rotor return:
        several weaker nearby Doppler components with amplitude modulation at the
        blade cadence. The amplitude modulation is what CVD/CFS is designed to reveal.

    This is a teaching simulation, not a full electromagnetic propeller model.
    """
    fd_body = doppler_hz_from_velocity(bulk_velocity_mps)
    x_body = body_strength * np.exp(1j * 2.0 * np.pi * fd_body * T)

    x_rotor = np.zeros_like(x_body)
    # symmetric velocity spread around the body, representing blade-induced micro-Doppler
    for k, side_velocity in enumerate([0.45, 0.9, 1.35, 1.8], start=1):
        for sign, scale in [(+1, 1.0), (-1, 0.85)]:
            fd = doppler_hz_from_velocity(bulk_velocity_mps + sign * side_velocity)
            phase_offset = RNG.uniform(0, 2.0 * np.pi)
            amp_mod = 1.0 + 0.9 * np.cos(2.0 * np.pi * blade_cadence_hz * T + phase_offset)
            x_rotor += scale * (rotor_strength / k) * amp_mod * np.exp(1j * 2.0 * np.pi * fd * T)

    x_clean = x_body + x_rotor

    signal_power = np.mean(np.abs(x_clean) ** 2)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power / 2.0) * (RNG.standard_normal(N) + 1j * RNG.standard_normal(N))
    return x_clean + noise


def compute_cfs(x: np.ndarray):
    """Compute STFT -> CVD -> CFS."""
    f_doppler, t_stft, zxx = stft(
        x,
        fs=PRF,
        window=WINDOW,
        nperseg=N_PER_SEG,
        noverlap=N_OVERLAP,
        nfft=N_FFT,
        return_onesided=False,
        boundary=None,
        padded=False,
    )

    f_doppler = np.fft.fftshift(f_doppler)
    zxx = np.fft.fftshift(zxx, axes=0)
    velocity_axis = velocity_from_doppler_hz(f_doppler)

    stft_power = np.abs(zxx) ** 2
    stft_power_centered = stft_power - np.mean(stft_power, axis=1, keepdims=True)

    dt_frame = np.median(np.diff(t_stft))
    cvd_freq = np.fft.rfftfreq(stft_power_centered.shape[1], d=dt_frame)
    cvd = np.abs(np.fft.rfft(stft_power_centered, axis=1))

    if USE_TARGET_VELOCITY_MASK:
        # In real processing, CFS is usually computed from a detected target region,
        # not from completely empty velocity bins. This mask keeps velocity bins with
        # relatively large average target energy and reduces pure-noise accumulation.
        mean_power = np.mean(stft_power, axis=1)
        velocity_mask = mean_power >= np.percentile(mean_power, 85)
    else:
        velocity_mask = np.ones(stft_power.shape[0], dtype=bool)

    cfs = np.sum(cvd[velocity_mask, :], axis=0)

    keep = (cvd_freq >= MIN_CADENCE_HZ) & (cvd_freq <= MAX_CADENCE_HZ)
    cfs_out = cfs[keep]
    cfs_out = cfs_out / (np.max(cfs_out) + 1e-12)
    return cvd_freq[keep], cfs_out, velocity_axis, stft_power, cvd[:, keep]


def main() -> None:
    output_dir = Path(__file__).resolve().parent

    x_hover = make_uav_slow_time_signal(bulk_velocity_mps=0.0)
    x_approach = make_uav_slow_time_signal(bulk_velocity_mps=12.0)

    f_c_hover, cfs_hover, *_ = compute_cfs(x_hover)
    f_c_approach, cfs_approach, *_ = compute_cfs(x_approach)

    plt.figure(figsize=(10, 5.5))
    plt.plot(f_c_hover, cfs_hover, label="Hovering UAV, body velocity = 0 m/s")
    plt.plot(f_c_approach, cfs_approach, label="Approaching UAV, body velocity = +12 m/s")
    plt.axvline(55.0, linestyle="--", linewidth=1.2, label="Simulated blade cadence = 55 Hz")
    plt.xlabel("Cadence frequency (Hz)")
    plt.ylabel("Normalized CFS magnitude")
    plt.title("CFS comparison: hovering vs approaching UAV")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()

    out_path = "outputs/cfs_simulation/cfs_hover_vs_approach.png"
    plt.savefig(out_path, dpi=180)
    print(f"Saved figure to: {out_path}")

    for name, f_axis, cfs in [("hover", f_c_hover, cfs_hover), ("approach", f_c_approach, cfs_approach)]:
        idx = np.argsort(cfs)[-5:][::-1]
        peaks = ", ".join([f"{f_axis[i]:.1f} Hz ({cfs[i]:.2f})" for i in idx])
        print(f"Top CFS peaks for {name}: {peaks}")


if __name__ == "__main__":
    main()
