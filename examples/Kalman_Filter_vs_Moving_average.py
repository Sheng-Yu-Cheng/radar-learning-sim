"""
Simulation 1: Kalman Filter vs Moving Average for UAV Tracking

Run:
    pip install streamlit numpy matplotlib pandas
    streamlit run sim1_kf_vs_moving_average.py

This is a measurement-level radar simulation. It assumes the radar has already
estimated range and azimuth angle for each frame, then compares moving average
and Kalman filtering under different noise, maneuver, and false-detection settings.
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# -----------------------------
# Simulation utilities
# -----------------------------

def make_trajectory(mode: str, t: np.ndarray):
    """Return true x, y, range, angle, radial velocity, angular velocity."""
    if mode == "Smooth crossing":
        x = 4.0 + 0.45 * t
        y = -2.2 + 0.22 * t
    elif mode == "Circling / curved":
        x = 9.0 + 2.2 * np.sin(0.55 * t)
        y = 2.8 * np.cos(0.55 * t)
    elif mode == "Aggressive maneuver":
        x = 5.0 + 0.45 * t + 0.9 * np.sin(1.15 * t)
        y = -1.8 + 0.18 * t + 1.0 * np.sin(0.85 * t + 0.8)
        # abrupt turn-like perturbation
        y += np.where(t > t.max() * 0.55, 1.2 * (1.0 - np.exp(-(t - t.max() * 0.55))), 0.0)
    else:  # Hover with drift
        x = 8.0 + 0.35 * np.sin(0.9 * t)
        y = 1.5 + 0.22 * np.sin(1.4 * t + 0.5)

    r = np.sqrt(x**2 + y**2)
    theta = np.unwrap(np.arctan2(y, x))
    dt = t[1] - t[0]
    vr = np.gradient(r, dt)
    omega = np.gradient(theta, dt)
    return x, y, r, theta, vr, omega


def apply_moving_average(values: np.ndarray, window: int):
    window = max(1, int(window))
    kernel = np.ones(window) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def kalman_filter(z, dt, q_level, r_diag):
    """
    State: [theta, range, omega, radial_velocity]^T
    Measurement: [theta, range, omega, radial_velocity]^T
    Constant-velocity model similar to the textbook chapter.
    """
    n = len(z)
    x = np.zeros((4, 1))
    x[:, 0] = z[0]

    F = np.array([
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    H = np.eye(4)

    # Continuous white-noise acceleration style, adapted to [theta, R, omega, vr].
    Q = q_level * np.array([
        [dt**3 / 3, 0.0, dt**2 / 2, 0.0],
        [0.0, dt**3 / 3, 0.0, dt**2 / 2],
        [dt**2 / 2, 0.0, dt, 0.0],
        [0.0, dt**2 / 2, 0.0, dt],
    ])
    R = np.diag(r_diag)
    P = np.diag([0.5, 1.0, 0.5, 1.0])
    I = np.eye(4)

    out = np.zeros((n, 4))
    out[0] = x[:, 0]

    for k in range(1, n):
        # predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # update
        zk = z[k].reshape(4, 1)
        innovation = zk - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x = x_pred + K @ innovation
        P = (I - K @ H) @ P_pred
        out[k] = x[:, 0]

    return out


def rmse_xy(theta, r, x_true, y_true):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    err = np.sqrt((x - x_true) ** 2 + (y - y_true) ** 2)
    return float(np.sqrt(np.mean(err**2))), err, x, y


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="KF vs Moving Average UAV Tracking", layout="wide")
st.title("Simulation 1 — Kalman Filter vs Moving Average for UAV Tracking")
st.caption("Measurement-level radar simulation: noisy range/angle measurements are tracked by moving average and Kalman filtering.")

with st.sidebar:
    st.header("Scenario")
    mode = st.selectbox("UAV motion", ["Smooth crossing", "Circling / curved", "Aggressive maneuver", "Hover with drift"])
    duration = st.slider("Duration (s)", 5.0, 30.0, 16.0, 1.0)
    fps = st.slider("Frame rate (Hz)", 5, 60, 20, 1)
    seed = st.number_input("Random seed", value=7, step=1)

    st.header("Radar measurement quality")
    virtual_rx = st.slider("Effective Rx / virtual antennas", 2, 16, 3, 1)
    angle_noise_deg_base = st.slider("Angle noise at 2 Rx (deg)", 1.0, 40.0, 12.0, 0.5)
    range_noise_m = st.slider("Range noise (m)", 0.01, 2.0, 0.25, 0.01)
    false_prob = st.slider("Sidelobe / false detection probability", 0.0, 0.40, 0.05, 0.01)
    false_angle_jump_deg = st.slider("False angle jump (deg)", 5.0, 80.0, 30.0, 1.0)

    st.header("Moving average")
    ma_window = st.slider("Window size (frames)", 1, 25, 7, 1)

    st.header("Kalman filter")
    q_level = st.slider("Q: maneuver allowance", 0.0001, 2.0, 0.08, 0.0001, format="%.4f")
    angle_r_scale = st.slider("R scale for angle measurement", 0.1, 30.0, 1.0, 0.1)
    range_r_scale = st.slider("R scale for range measurement", 0.1, 30.0, 1.0, 0.1)

rng = np.random.default_rng(int(seed))
t = np.arange(0.0, duration, 1.0 / fps)
dt = 1.0 / fps
x_true, y_true, r_true, theta_true, vr_true, omega_true = make_trajectory(mode, t)

# More virtual antennas reduce angle noise, but do not create perfect measurements.
angle_sigma = np.deg2rad(angle_noise_deg_base) * np.sqrt(2.0 / virtual_rx)
range_sigma = range_noise_m

theta_meas = theta_true + rng.normal(0.0, angle_sigma, size=len(t))
r_meas = r_true + rng.normal(0.0, range_sigma, size=len(t))

# False detections: imitate sidelobe peak selection or bad angle peak tracking.
false_mask = rng.random(len(t)) < false_prob
theta_meas[false_mask] += rng.choice([-1, 1], size=false_mask.sum()) * np.deg2rad(false_angle_jump_deg) * (0.7 + 0.6 * rng.random(false_mask.sum()))
r_meas[false_mask] += rng.normal(0.0, 0.8 + range_sigma, size=false_mask.sum())

# unwrap angle before filtering/smoothing
theta_meas_unwrap = np.unwrap(theta_meas)
vr_meas = np.gradient(r_meas, dt)
omega_meas = np.gradient(theta_meas_unwrap, dt)
z = np.column_stack([theta_meas_unwrap, r_meas, omega_meas, vr_meas])

ma_theta = apply_moving_average(theta_meas_unwrap, ma_window)
ma_r = apply_moving_average(r_meas, ma_window)

r_diag = [
    max((angle_sigma**2) * angle_r_scale, 1e-8),
    max((range_sigma**2) * range_r_scale, 1e-8),
    max((angle_sigma / dt) ** 2 * angle_r_scale, 1e-8),
    max((range_sigma / dt) ** 2 * range_r_scale, 1e-8),
]
kf = kalman_filter(z, dt, q_level=q_level, r_diag=r_diag)

rmse_meas, err_meas, x_meas, y_meas = rmse_xy(theta_meas_unwrap, r_meas, x_true, y_true)
rmse_ma, err_ma, x_ma, y_ma = rmse_xy(ma_theta, ma_r, x_true, y_true)
rmse_kf, err_kf, x_kf, y_kf = rmse_xy(kf[:, 0], kf[:, 1], x_true, y_true)

m1, m2, m3 = st.columns(3)
m1.metric("Raw measurement RMSE", f"{rmse_meas:.3f} m")
m2.metric("Moving average RMSE", f"{rmse_ma:.3f} m")
m3.metric("Kalman filter RMSE", f"{rmse_kf:.3f} m")

col1, col2 = st.columns([1.15, 1])

with col1:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(x_true, y_true, label="True trajectory", linewidth=2)
    ax.scatter(x_meas, y_meas, s=12, alpha=0.35, label="Noisy radar measurement")
    ax.plot(x_ma, y_ma, label="Moving average", linewidth=2)
    ax.plot(x_kf, y_kf, label="Kalman filter", linewidth=2)
    if false_mask.any():
        ax.scatter(x_meas[false_mask], y_meas[false_mask], s=45, marker="x", label="False detections")
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.set_title("2D tracking result")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

with col2:
    fig2, ax2 = plt.subplots(figsize=(6.5, 3.2))
    ax2.plot(t, np.rad2deg(theta_true), label="True angle")
    ax2.plot(t, np.rad2deg(theta_meas_unwrap), alpha=0.35, label="Measured angle")
    ax2.plot(t, np.rad2deg(ma_theta), label="Moving average")
    ax2.plot(t, np.rad2deg(kf[:, 0]), label="Kalman filter")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Angle (deg)")
    ax2.set_title("Angle estimate")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(6.5, 3.2))
    ax3.plot(t, err_meas, alpha=0.35, label="Raw")
    ax3.plot(t, err_ma, label="Moving average")
    ax3.plot(t, err_kf, label="Kalman filter")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("2D error (m)")
    ax3.set_title("Tracking error")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)
    st.pyplot(fig3)

st.subheader("What to observe")
st.write(
    "Moving average often looks close to Kalman filtering when the target is slow and smooth. "
    "Kalman filtering becomes more useful when the target is moving continuously and the measurements are noisy but still informative. "
    "If false detections dominate or the angle measurement has almost no real information, Kalman filtering can only smooth the wrong track."
)

with st.expander("Show simulated data table"):
    df = pd.DataFrame({
        "time_s": t,
        "true_x_m": x_true,
        "true_y_m": y_true,
        "measured_x_m": x_meas,
        "measured_y_m": y_meas,
        "ma_x_m": x_ma,
        "ma_y_m": y_ma,
        "kf_x_m": x_kf,
        "kf_y_m": y_kf,
        "false_detection": false_mask,
    })
    st.dataframe(df, use_container_width=True)
