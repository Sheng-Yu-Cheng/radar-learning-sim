"""
Simulation 2: Simplified FMCW Post-Interferometry Radar Signal
              Kalman Filter vs Moving Average

Run:
    pip install streamlit numpy matplotlib pandas
    streamlit run sim2_fmcw_post_interferometry_kf_vs_ma.py

This is a simplified signal-level simulation inspired by interferometric FMCW radar.
It generates two receiver beat signals, extracts beat-frequency peaks, forms
post-interferometry range/angle measurements, and compares moving average with
Kalman filtering for UAV tracking.

The radar model is intentionally simplified for learning:
    RB1 = 2R + D/2 sin(theta)
    RB2 = 2R - D/2 sin(theta)
    fr_k = K * RB_k / c
    theta estimate from RB2 - RB1
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

C = 299_792_458.0


def make_trajectory(mode: str, t: np.ndarray):
    if mode == "Back-and-forth with angle drift":
        x = 7.0 + 1.4 * np.sin(0.55 * t)
        y = 1.7 + 0.55 * np.sin(0.22 * t + 0.4)
    elif mode == "Curved pass":
        x = 5.5 + 0.35 * t
        y = -1.8 + 2.2 * np.sin(0.22 * t)
    else:  # Maneuvering UAV
        x = 6.5 + 0.25 * t + 0.8 * np.sin(0.75 * t)
        y = 1.0 + 1.2 * np.sin(0.58 * t + 0.3) + 0.6 * np.where(t > t.max() * 0.55, 1 - np.exp(-(t - t.max() * 0.55)), 0)
    r = np.sqrt(x**2 + y**2)
    theta = np.unwrap(np.arctan2(y, x))
    dt = t[1] - t[0]
    vr = np.gradient(r, dt)
    omega = np.gradient(theta, dt)
    return x, y, r, theta, vr, omega


def moving_average(values, window):
    window = max(1, int(window))
    kernel = np.ones(window) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def kalman_filter(z, dt, q_level, r_diag):
    # State: [theta, range, omega, radial_velocity]^T
    F = np.array([
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    H = np.eye(4)
    Q = q_level * np.array([
        [dt**3 / 3, 0.0, dt**2 / 2, 0.0],
        [0.0, dt**3 / 3, 0.0, dt**2 / 2],
        [dt**2 / 2, 0.0, dt, 0.0],
        [0.0, dt**2 / 2, 0.0, dt],
    ])
    R = np.diag(r_diag)
    P = np.diag([0.6, 1.0, 0.6, 1.0])
    x = z[0].reshape(4, 1)
    I = np.eye(4)
    out = np.zeros_like(z)
    out[0] = z[0]
    for k in range(1, len(z)):
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        y = z[k].reshape(4, 1) - H @ x_pred
        S = H @ P_pred @ H.T + R
        K_gain = P_pred @ H.T @ np.linalg.inv(S)
        x = x_pred + K_gain @ y
        P = (I - K_gain @ H) @ P_pred
        out[k] = x[:, 0]
    return out


def generate_rx_signal(fr, fd, amp, snr_db, m_samples, n_chirps, fs, chirp_time, rng, sidelobe_strength=0.0, false_fr=None):
    """Generate a simplified complex FMCW beat signal matrix [chirp, sample]."""
    ts = np.arange(m_samples) / fs
    n = np.arange(n_chirps)
    signal = amp * np.exp(1j * 2 * np.pi * (fr * ts[None, :] + fd * chirp_time * n[:, None]))

    # Add a weaker false/sidelobe-like component at another beat frequency.
    if sidelobe_strength > 0 and false_fr is not None:
        signal += amp * sidelobe_strength * np.exp(1j * 2 * np.pi * (false_fr * ts[None, :] + 0.25 * fd * chirp_time * n[:, None]))

    sig_power = amp**2
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (rng.normal(size=signal.shape) + 1j * rng.normal(size=signal.shape))
    return signal + noise


def estimate_range_frequency(sig, fs, nfft):
    # Average range spectrum over chirps, then peak-pick.
    spec = np.fft.fft(sig, n=nfft, axis=1)
    mag = np.mean(np.abs(spec), axis=0)
    half = nfft // 2
    idx = int(np.argmax(mag[:half]))
    fr = idx * fs / nfft
    return fr, mag[:half]


def estimate_doppler_frequency(sig, fr_idx, chirp_time, ndop):
    # Extract slow-time samples at a selected fast-time FFT bin and FFT along chirps.
    range_fft = np.fft.fft(sig, n=sig.shape[1], axis=1)
    fr_idx = int(np.clip(fr_idx, 0, sig.shape[1] - 1))
    slow = range_fft[:, fr_idx]
    dop = np.fft.fftshift(np.fft.fft(slow, n=ndop))
    mag = np.abs(dop)
    bins = np.fft.fftshift(np.fft.fftfreq(ndop, d=chirp_time))
    idx = int(np.argmax(mag))
    return bins[idx]


def rmse_xy(theta, r, x_true, y_true):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    err = np.sqrt((x - x_true) ** 2 + (y - y_true) ** 2)
    return float(np.sqrt(np.mean(err**2))), err, x, y


st.set_page_config(page_title="FMCW Post-Interferometry KF vs MA", layout="wide")
st.title("Simulation 2 — FMCW Post-Interferometry: Kalman Filter vs Moving Average")
st.caption("Simplified signal-level simulation: two receiver beat signals → post-interferometry measurements → tracking.")

with st.sidebar:
    st.header("Trajectory")
    mode = st.selectbox("UAV motion", ["Back-and-forth with angle drift", "Curved pass", "Maneuvering UAV"])
    n_frames = st.slider("Frames", 20, 120, 55, 5)
    dt_frame = st.slider("Frame interval (s)", 0.05, 0.40, 0.12, 0.01)
    seed = st.number_input("Random seed", value=11, step=1)

    st.header("FMCW radar")
    carrier_ghz = st.slider("Carrier frequency (GHz)", 24.0, 79.0, 77.0, 1.0)
    bandwidth_mhz = st.slider("Bandwidth (MHz)", 100.0, 2000.0, 1000.0, 50.0)
    chirp_time_us = st.slider("Chirp time (µs)", 40.0, 400.0, 120.0, 10.0)
    sample_rate_mhz = st.slider("ADC sample rate (MHz)", 1.0, 10.0, 4.0, 0.5)
    m_samples = st.select_slider("Samples per chirp", options=[64, 96, 128, 192, 256], value=128)
    n_chirps = st.select_slider("Chirps per frame", options=[16, 24, 32, 48, 64], value=32)
    nfft = st.select_slider("Range FFT size", options=[128, 256, 512, 1024], value=512)
    ndop = st.select_slider("Doppler FFT size", options=[32, 64, 128, 256], value=128)

    st.header("Interferometer / environment")
    baseline_m = st.slider("Rx baseline D (m)", 0.02, 1.50, 0.08, 0.01)
    snr_db = st.slider("SNR (dB)", -10.0, 40.0, 12.0, 1.0)
    sidelobe_prob = st.slider("False sidelobe peak probability", 0.0, 0.50, 0.08, 0.01)
    sidelobe_strength = st.slider("Sidelobe component strength", 0.0, 0.90, 0.25, 0.05)

    st.header("Tracking")
    ma_window = st.slider("Moving average window", 1, 21, 5, 1)
    q_level = st.slider("Kalman Q: maneuver allowance", 0.0001, 2.0, 0.10, 0.0001, format="%.4f")
    r_scale = st.slider("Kalman R scale", 0.1, 50.0, 4.0, 0.1)

rng = np.random.default_rng(int(seed))
t = np.arange(n_frames) * dt_frame
x_true, y_true, r_true, theta_true, vr_true, omega_true = make_trajectory(mode, t)

f0 = carrier_ghz * 1e9
lam = C / f0
B = bandwidth_mhz * 1e6
T = chirp_time_us * 1e-6
K = B / T
fs = sample_rate_mhz * 1e6

fr1_est = np.zeros(n_frames)
fr2_est = np.zeros(n_frames)
fd1_est = np.zeros(n_frames)
fd2_est = np.zeros(n_frames)

range_spectrum_last = None
false_mask = np.zeros(n_frames, dtype=bool)

for k in range(n_frames):
    R = r_true[k]
    th = theta_true[k]
    vr = vr_true[k]
    w = omega_true[k]

    # Simplified transmitter-target-receiver beat path lengths.
    RB1 = 2 * R + 0.5 * baseline_m * np.sin(th)
    RB2 = 2 * R - 0.5 * baseline_m * np.sin(th)
    fr1 = K * RB1 / C
    fr2 = K * RB2 / C

    # Simplified Doppler terms. Difference contains angular velocity information.
    fd_common = 2 * vr / lam
    fd_delta = baseline_m * w * np.cos(th) / lam
    fd1 = fd_common + 0.5 * fd_delta
    fd2 = fd_common - 0.5 * fd_delta

    # Occasional false range component can win under low SNR/high sidelobe strength.
    do_false = rng.random() < sidelobe_prob
    false_mask[k] = do_false
    false_offset_hz = rng.choice([-1, 1]) * rng.uniform(2.0, 8.0) * fs / nfft
    false_fr1 = max(0.0, fr1 + false_offset_hz) if do_false else None
    false_fr2 = max(0.0, fr2 - false_offset_hz) if do_false else None

    s1 = generate_rx_signal(fr1, fd1, 1.0, snr_db, m_samples, n_chirps, fs, T, rng, sidelobe_strength if do_false else 0.0, false_fr1)
    s2 = generate_rx_signal(fr2, fd2, 1.0, snr_db, m_samples, n_chirps, fs, T, rng, sidelobe_strength if do_false else 0.0, false_fr2)

    fr1_est[k], spec1 = estimate_range_frequency(s1, fs, nfft)
    fr2_est[k], spec2 = estimate_range_frequency(s2, fs, nfft)
    range_spectrum_last = (spec1, spec2)

    # Estimate Doppler at the selected fast-time bin. Map nfft peak to raw sample FFT bin approximately.
    fr1_idx_raw = int(round(fr1_est[k] / fs * m_samples))
    fr2_idx_raw = int(round(fr2_est[k] / fs * m_samples))
    fd1_est[k] = estimate_doppler_frequency(s1, fr1_idx_raw, T, ndop)
    fd2_est[k] = estimate_doppler_frequency(s2, fr2_idx_raw, T, ndop)

# Post-interferometry estimates.
RB1_est = fr1_est * C / K
RB2_est = fr2_est * C / K
r_meas = (RB1_est + RB2_est) / 4.0
sin_arg = np.clip((RB2_est - RB1_est) / baseline_m, -1.0, 1.0)
theta_meas = np.unwrap(np.arcsin(sin_arg))

vr_meas = 0.25 * (fd1_est + fd2_est) * lam  # since common two-way fd is 2vr/lambda
omega_meas = (fd1_est - fd2_est) * lam / np.maximum(baseline_m * np.cos(theta_meas), 1e-6)

# Stabilize rare impossible values from bad peak picks.
omega_meas = np.clip(omega_meas, -5.0, 5.0)
vr_meas = np.clip(vr_meas, -10.0, 10.0)

z = np.column_stack([theta_meas, r_meas, omega_meas, vr_meas])
ma_theta = moving_average(theta_meas, ma_window)
ma_r = moving_average(r_meas, ma_window)

# Measurement variance derived roughly from observed jitter and scaled by user.
angle_var = max(np.var(theta_meas - theta_true), 1e-5) * r_scale
range_var = max(np.var(r_meas - r_true), 1e-4) * r_scale
omega_var = max(np.var(omega_meas - omega_true), 1e-4) * r_scale
vr_var = max(np.var(vr_meas - vr_true), 1e-4) * r_scale
kf = kalman_filter(z, dt_frame, q_level=q_level, r_diag=[angle_var, range_var, omega_var, vr_var])

rmse_raw, err_raw, x_raw, y_raw = rmse_xy(theta_meas, r_meas, x_true, y_true)
rmse_ma, err_ma, x_ma, y_ma = rmse_xy(ma_theta, ma_r, x_true, y_true)
rmse_kf, err_kf, x_kf, y_kf = rmse_xy(kf[:, 0], kf[:, 1], x_true, y_true)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Raw post-interferometry RMSE", f"{rmse_raw:.3f} m")
m2.metric("Moving average RMSE", f"{rmse_ma:.3f} m")
m3.metric("Kalman filter RMSE", f"{rmse_kf:.3f} m")
m4.metric("False sidelobe frames", f"{false_mask.sum()} / {n_frames}")

col1, col2 = st.columns([1.1, 1.0])
with col1:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(x_true, y_true, linewidth=2, label="True trajectory")
    ax.scatter(x_raw, y_raw, s=15, alpha=0.35, label="Post-interferometry measurement")
    ax.plot(x_ma, y_ma, linewidth=2, label="Moving average")
    ax.plot(x_kf, y_kf, linewidth=2, label="Kalman filter")
    if false_mask.any():
        ax.scatter(x_raw[false_mask], y_raw[false_mask], marker="x", s=50, label="Frames with false sidelobe component")
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.set_title("2D tracking from simulated FMCW post-interferometry measurements")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    st.pyplot(fig)

with col2:
    fig2, ax2 = plt.subplots(figsize=(6.4, 3.1))
    ax2.plot(t, np.rad2deg(theta_true), label="True angle")
    ax2.plot(t, np.rad2deg(theta_meas), alpha=0.35, label="Measured angle")
    ax2.plot(t, np.rad2deg(ma_theta), label="Moving average")
    ax2.plot(t, np.rad2deg(kf[:, 0]), label="Kalman filter")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Angle (deg)")
    ax2.set_title("Angle estimate")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(6.4, 3.1))
    ax3.plot(t, err_raw, alpha=0.35, label="Raw")
    ax3.plot(t, err_ma, label="Moving average")
    ax3.plot(t, err_kf, label="Kalman filter")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("2D error (m)")
    ax3.set_title("Tracking error")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)
    st.pyplot(fig3)

st.subheader("Range spectrum from the last simulated frame")
if range_spectrum_last is not None:
    spec1, spec2 = range_spectrum_last
    f_axis = np.arange(len(spec1)) * fs / nfft
    range_axis = (f_axis * C / K) / 2.0
    fig4, ax4 = plt.subplots(figsize=(9, 3.2))
    ax4.plot(range_axis, 20 * np.log10(spec1 / (np.max(spec1) + 1e-12) + 1e-12), label="Receiver 1")
    ax4.plot(range_axis, 20 * np.log10(spec2 / (np.max(spec2) + 1e-12) + 1e-12), label="Receiver 2")
    ax4.set_xlabel("Approximate one-way range (m)")
    ax4.set_ylabel("Normalized magnitude (dB)")
    ax4.set_title("Fast-time FFT range spectrum")
    ax4.set_ylim(-60, 3)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    st.pyplot(fig4)

st.subheader("What to observe")
st.write(
    "Increase SNR, baseline, or FFT size to improve the measurement before tracking. "
    "Increase sidelobe probability/strength to see false peaks. Kalman filtering helps when peak picks are noisy but still related to the true target; "
    "it cannot reliably recover when the selected radar peak frequently comes from a false sidelobe component."
)

with st.expander("Show measurement table"):
    df = pd.DataFrame({
        "time_s": t,
        "true_range_m": r_true,
        "measured_range_m": r_meas,
        "true_angle_deg": np.rad2deg(theta_true),
        "measured_angle_deg": np.rad2deg(theta_meas),
        "true_omega_rad_s": omega_true,
        "measured_omega_rad_s": omega_meas,
        "true_vr_m_s": vr_true,
        "measured_vr_m_s": vr_meas,
        "false_sidelobe_frame": false_mask,
    })
    st.dataframe(df, use_container_width=True)
