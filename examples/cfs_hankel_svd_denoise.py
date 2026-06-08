#!/usr/bin/env python3
"""
CFS simulation with Hankel SVD denoising on the raw slow-time signal.

This script simulates a noisy UAV slow-time radar signal, applies Hankel SVD denoising
before STFT/CVD/CFS, and compares the resulting CFS with the non-denoised pipeline.

Pipeline without denoising:
    raw slow-time signal -> STFT -> CVD -> CFS

Pipeline with denoising:
    raw slow-time signal -> Hankel SVD denoise -> STFT -> CVD -> CFS

Output:
    cfs_hankel_svd_denoise_comparison.png

Install dependencies:
    pip install numpy scipy matplotlib

Run:
    python cfs_hankel_svd_denoise.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hankel
from scipy.signal import stft, get_window
from scipy.sparse.linalg import svds


# -----------------------------
# Radar / simulation parameters
# -----------------------------
C = 3.0e8
FC = 77.0e9
LAMBDA = C / FC
PRF = 4_000.0               # enough for the smaller velocity used in this denoising demo
DURATION = 1.5
N = int(PRF * DURATION)
T = np.arange(N) / PRF

N_PER_SEG = 256
N_OVERLAP = 248             # small hop size -> cadence Nyquist well above 55 Hz
N_FFT = 1024
WINDOW = get_window("hann", N_PER_SEG)

MIN_CADENCE_HZ = 20.0
MAX_CADENCE_HZ = 160.0
USE_TARGET_VELOCITY_MASK = True

RNG = np.random.default_rng(11)


def doppler_hz_from_velocity(v_mps: float) -> float:
    return 2.0 * v_mps / LAMBDA


def velocity_from_doppler_hz(fd_hz: np.ndarray) -> np.ndarray:
    return fd_hz * LAMBDA / 2.0


def make_noisy_uav_signal(
    bulk_velocity_mps: float = 1.2,
    blade_cadence_hz: float = 55.0,
    snr_db: float = -2.0,
    rotor_strength: float = 0.70,
    body_strength: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return noisy raw slow-time signal and clean reference signal.

    The moderate/low SNR makes the denoising effect visible.
    """
    fd_body = doppler_hz_from_velocity(bulk_velocity_mps)
    x_body = body_strength * np.exp(1j * 2.0 * np.pi * fd_body * T)

    x_rotor = np.zeros_like(x_body)
    for k, side_velocity in enumerate([0.35, 0.7, 1.05], start=1):
        for sign, scale in [(+1, 1.0), (-1, 0.85)]:
            fd = doppler_hz_from_velocity(bulk_velocity_mps + sign * side_velocity)
            phase_offset = RNG.uniform(0, 2.0 * np.pi)
            amp_mod = 1.0 + 0.9 * np.cos(2.0 * np.pi * blade_cadence_hz * T + phase_offset)
            x_rotor += scale * (rotor_strength / k) * amp_mod * np.exp(1j * 2.0 * np.pi * fd * T)

    # Weak periodic background interference at a different cadence.
    # This imitates an environmental distribution shift such as a fan/tree/mechanical vibration.
    bg_cadence_hz = 92.0
    fd_bg = doppler_hz_from_velocity(-0.8)
    bg_amp = 0.18 * (1.0 + 0.9 * np.cos(2.0 * np.pi * bg_cadence_hz * T))
    x_background = bg_amp * np.exp(1j * 2.0 * np.pi * fd_bg * T)

    x_clean = x_body + x_rotor + x_background

    signal_power = np.mean(np.abs(x_clean) ** 2)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power / 2.0) * (RNG.standard_normal(N) + 1j * RNG.standard_normal(N))
    x_noisy = x_clean + noise
    return x_noisy, x_clean


def anti_diagonal_average(H: np.ndarray) -> np.ndarray:
    """
    Reconstruct a 1-D signal from a Hankel matrix by averaging anti-diagonals.
    """
    rows, cols = H.shape
    i = np.arange(rows)[:, None]
    j = np.arange(cols)[None, :]
    anti_diag_index = (i + j).ravel()

    y = np.zeros(rows + cols - 1, dtype=H.dtype)
    counts = np.zeros(rows + cols - 1, dtype=float)
    np.add.at(y, anti_diag_index, H.ravel())
    np.add.at(counts, anti_diag_index, 1.0)
    return y / counts


def hankel_svd_denoise(x: np.ndarray, window_length: int = 160, rank: int = 14) -> np.ndarray:
    """
    Hankel SVD denoise for a complex 1-D signal.

    Steps:
    1. Embed signal into a Hankel matrix.
    2. Keep only the largest singular components.
    3. Reconstruct a denoised 1-D signal by anti-diagonal averaging.

    Caution:
    rank is important. If rank is too small, weak micro-Doppler components may be
    removed together with noise. If rank is too large, too much noise is retained.
    """
    if window_length <= 1 or window_length >= len(x):
        raise ValueError("window_length must be between 2 and len(x)-1")
    if rank < 1:
        raise ValueError("rank must be positive")

    H = hankel(x[:window_length], x[window_length - 1:])
    rank = min(rank, min(H.shape) - 1)
    U, s, Vh = svds(H, k=rank, which="LM")

    order = np.argsort(s)[::-1]
    U, s, Vh = U[:, order], s[order], Vh[order, :]
    H_denoised = (U * s) @ Vh
    return anti_diagonal_average(H_denoised)


def compute_cfs(x: np.ndarray):
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

    raw_signal, clean_signal = make_noisy_uav_signal()
    denoised_signal = hankel_svd_denoise(raw_signal, window_length=160, rank=14)

    f_raw, cfs_raw, *_ = compute_cfs(raw_signal)
    f_den, cfs_den, *_ = compute_cfs(denoised_signal)
    f_clean, cfs_clean, *_ = compute_cfs(clean_signal)

    plt.figure(figsize=(10, 5.5))
    plt.plot(f_raw, cfs_raw, label="Without denoising: raw signal -> STFT -> CVD -> CFS")
    plt.plot(f_den, cfs_den, label="With Hankel SVD: raw signal -> denoise -> STFT -> CVD -> CFS")
    plt.plot(f_clean, cfs_clean, linestyle="--", linewidth=1.3, label="Clean reference, not available in real data")
    plt.axvline(55.0, linestyle=":", linewidth=1.2, label="Simulated blade cadence = 55 Hz")
    plt.axvline(92.0, linestyle=":", linewidth=1.2, label="Simulated background cadence = 92 Hz")
    plt.xlabel("Cadence frequency (Hz)")
    plt.ylabel("Normalized CFS magnitude")
    plt.title("Effect of Hankel SVD denoising before CFS extraction")
    plt.grid(True, alpha=0.35)
    plt.legend(fontsize=8)
    plt.tight_layout()

    out_path = "outputs/cfs_simulation/cfs_hankel_svd_denoise_comparison.png"
    plt.savefig(out_path, dpi=180)
    print(f"Saved figure to: {out_path}")

    for name, f_axis, cfs in [
        ("raw", f_raw, cfs_raw),
        ("denoised", f_den, cfs_den),
        ("clean", f_clean, cfs_clean),
    ]:
        idx = np.argsort(cfs)[-5:][::-1]
        peaks = ", ".join([f"{f_axis[i]:.1f} Hz ({cfs[i]:.2f})" for i in idx])
        print(f"Top CFS peaks for {name}: {peaks}")


if __name__ == "__main__":
    main()
