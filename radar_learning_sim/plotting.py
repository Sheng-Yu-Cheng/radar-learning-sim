"""Plot helpers for Radar Learning Simulator."""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from .core import db10_power, db20, RadarConfig, make_lfm_waveform, steering_vector_ula, C0


def _save_or_show(fig, savepath: Optional[str | Path] = None):
    fig.tight_layout()
    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=180)
        plt.close(fig)
    return fig


def plot_waveform(cfg: RadarConfig, savepath: Optional[str | Path] = None):
    x = make_lfm_waveform(cfg)
    t_us = np.arange(len(x)) / cfg.fs_hz * 1e6
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t_us, np.real(x), label="real")
    ax.plot(t_us, np.imag(x), label="imag", alpha=0.8)
    ax.set_title("Transmit LFM waveform")
    ax.set_xlabel("Time inside pulse (us)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _save_or_show(fig, savepath)


def plot_range_profile(range_cube, range_axis_m, cpi=0, pulse=0, rx=0, max_range_m=None, savepath=None):
    y = range_cube[cpi, pulse, :, rx]
    x = range_axis_m
    mask = np.ones_like(x, dtype=bool) if max_range_m is None else x <= max_range_m
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(x[mask], db20(y[mask]))
    ax.set_title(f"Matched-filter range profile, CPI={cpi}, pulse={pulse}, rx={rx}")
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, alpha=0.3)
    return _save_or_show(fig, savepath)


def plot_rdi(power_rdi, range_axis_m, vel_axis_mps, cpi=0, max_range_m=None, dynamic_range_db=55, detections=None, savepath=None):
    p = power_rdi[cpi]
    r = range_axis_m
    mask_r = np.ones_like(r, dtype=bool) if max_range_m is None else r <= max_range_m
    z = db10_power(p[mask_r, :])
    zmax = np.nanmax(z)
    fig, ax = plt.subplots(figsize=(9, 5.6))
    im = ax.imshow(
        z,
        origin="lower",
        aspect="auto",
        extent=[vel_axis_mps[0], vel_axis_mps[-1], r[mask_r][0], r[mask_r][-1]],
        vmin=zmax - dynamic_range_db,
        vmax=zmax,
    )
    ax.set_title(f"Range-Doppler Image / Range-Velocity Map, CPI={cpi}")
    ax.set_xlabel("Radial velocity (m/s)")
    ax.set_ylabel("Range (m)")
    fig.colorbar(im, ax=ax, label="Power (dB)")
    if detections is not None:
        det = detections[mask_r, :]
        rr, dd = np.where(det)
        if len(rr):
            ax.scatter(vel_axis_mps[dd], r[mask_r][rr], s=14, marker="o", facecolors="none", edgecolors="white", linewidths=0.8)
    return _save_or_show(fig, savepath)


def plot_range_time(range_cube, range_axis_m, cfg: RadarConfig, rx=0, max_range_m=None, savepath=None):
    # Average magnitude over pulses in each CPI.
    rt = np.mean(np.abs(range_cube[:, :, :, rx])**2, axis=1)  # [cpi, range]
    r = range_axis_m
    mask_r = np.ones_like(r, dtype=bool) if max_range_m is None else r <= max_range_m
    z = db10_power(rt[:, mask_r].T)
    t = np.arange(cfg.num_cpi) * cfg.num_pulses * cfg.pri_s
    fig, ax = plt.subplots(figsize=(9, 5.2))
    im = ax.imshow(z, origin="lower", aspect="auto", extent=[t[0], t[-1] if len(t)>1 else 0, r[mask_r][0], r[mask_r][-1]])
    ax.set_title("Range-Time Image / Range Spectrogram")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Range (m)")
    fig.colorbar(im, ax=ax, label="Power (dB)")
    return _save_or_show(fig, savepath)


def plot_doppler_time(power_rdi, vel_axis_mps, range_axis_m, range_gate_m: Tuple[float, float], savepath=None):
    lo, hi = range_gate_m
    mask = (range_axis_m >= lo) & (range_axis_m <= hi)
    vt = np.mean(power_rdi[:, mask, :], axis=1)  # [cpi, doppler]
    z = db10_power(vt.T)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    im = ax.imshow(z, origin="lower", aspect="auto", extent=[0, power_rdi.shape[0]-1, vel_axis_mps[0], vel_axis_mps[-1]])
    ax.set_title(f"Velocity-Time Image / Doppler Spectrogram, range gate {lo:.0f}-{hi:.0f} m")
    ax.set_xlabel("CPI index")
    ax.set_ylabel("Radial velocity (m/s)")
    fig.colorbar(im, ax=ax, label="Power (dB)")
    return _save_or_show(fig, savepath)


def plot_beam_pattern(cfg: RadarConfig, steer_angle_deg=0.0, angle_grid_deg=None, savepath=None):
    if angle_grid_deg is None:
        angle_grid_deg = np.linspace(-90, 90, 721)
    A = steering_vector_ula(angle_grid_deg, cfg)
    a0 = steering_vector_ula(steer_angle_deg, cfg)
    # normalized DAS response
    b = np.abs(A @ np.conj(a0))**2 / cfg.num_rx**2
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(angle_grid_deg, db10_power(b))
    ax.set_ylim(-50, 1)
    ax.set_title(f"ULA Beam Pattern, steer angle = {steer_angle_deg:.1f} deg")
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Normalized power (dB)")
    ax.grid(True, alpha=0.3)
    return _save_or_show(fig, savepath)


def plot_range_angle(angle_cube, range_axis_m, angle_grid_deg, cpi=0, doppler_index=None, max_range_m=None, dynamic_range_db=50, savepath=None):
    # If doppler_index is None, noncoherently sum Doppler power.
    if doppler_index is None:
        p = np.sum(np.abs(angle_cube[cpi])**2, axis=1)  # [range, angle]
        title_suffix = "Doppler-summed"
    else:
        p = np.abs(angle_cube[cpi, :, doppler_index, :])**2
        title_suffix = f"Doppler bin {doppler_index}"
    r = range_axis_m
    mask_r = np.ones_like(r, dtype=bool) if max_range_m is None else r <= max_range_m
    z = db10_power(p[mask_r, :])
    zmax = np.nanmax(z)
    fig, ax = plt.subplots(figsize=(9, 5.6))
    im = ax.imshow(
        z,
        origin="lower",
        aspect="auto",
        extent=[angle_grid_deg[0], angle_grid_deg[-1], r[mask_r][0], r[mask_r][-1]],
        vmin=zmax - dynamic_range_db,
        vmax=zmax,
    )
    ax.set_title(f"Range-Angle Image, CPI={cpi}, {title_suffix}")
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Range (m)")
    fig.colorbar(im, ax=ax, label="Power (dB)")
    return _save_or_show(fig, savepath)


def plot_cfar_slice(power_map, threshold, range_axis_m, vel_axis_mps, range_index=None, savepath=None):
    if range_index is None:
        range_index = int(np.nanargmax(np.nanmax(power_map, axis=1)))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(vel_axis_mps, db10_power(power_map[range_index, :]), label="RDI power")
    ax.plot(vel_axis_mps, db10_power(threshold[range_index, :]), label="CA-CFAR threshold")
    ax.set_title(f"CFAR threshold slice at range = {range_axis_m[range_index]:.1f} m")
    ax.set_xlabel("Radial velocity (m/s)")
    ax.set_ylabel("Power (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _save_or_show(fig, savepath)
