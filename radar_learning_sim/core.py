"""
Radar Learning Simulator
========================
A small, stable Python framework for learning radar datacube concepts:
fast-time matched filtering, slow-time Doppler processing, spatial beamforming,
and 2D CA-CFAR detection.

This is intentionally written for readability rather than maximum speed.
It models a monostatic pulsed LFM radar with a ULA receive array.

Main datacube shapes
--------------------
raw:              [num_cpi, num_pulses, num_fast_time, num_rx]
range_cube:       [num_cpi, num_pulses, num_range_bins, num_rx]
range_doppler:    [num_cpi, num_range_bins, num_doppler_bins, num_rx]
angle_cube:       [num_cpi, num_range_bins, num_doppler_bins, num_angles]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Tuple
import numpy as np

C0 = 299_792_458.0


def db20(x: np.ndarray, floor_db: float = -120.0) -> np.ndarray:
    """20 log10(abs(x)) with numerical floor."""
    mag = np.maximum(np.abs(x), 10 ** (floor_db / 20))
    return 20 * np.log10(mag)


def db10_power(x: np.ndarray, floor_db: float = -120.0) -> np.ndarray:
    """10 log10(power) with numerical floor."""
    p = np.maximum(np.real(x), 10 ** (floor_db / 10))
    return 10 * np.log10(p)


@dataclass
class RadarConfig:
    """Basic pulsed LFM phased-array radar settings."""

    fc_hz: float = 10e9                 # carrier frequency
    fs_hz: float = 20e6                 # fast-time sampling rate
    prf_hz: float = 5e3                 # pulse repetition frequency
    pulse_width_s: float = 5e-6         # LFM pulse duration
    bandwidth_hz: float = 10e6          # LFM swept bandwidth
    num_fast_time: int = 1024           # samples in one receive swath
    num_pulses: int = 64                # pulses per CPI
    num_cpi: int = 8                    # number of CPIs in one simulated video
    num_rx: int = 8                     # receive antennas in ULA
    rx_spacing_m: Optional[float] = None # default = lambda/2
    noise_power: float = 1e-3           # complex noise variance per sample
    range_attenuation: bool = False     # apply 1/R^2 voltage attenuation if True
    seed: int = 7

    @property
    def wavelength_m(self) -> float:
        return C0 / self.fc_hz

    @property
    def pri_s(self) -> float:
        return 1.0 / self.prf_hz

    @property
    def pulse_samples(self) -> int:
        return int(round(self.pulse_width_s * self.fs_hz))

    @property
    def max_unambiguous_range_m(self) -> float:
        return C0 / (2.0 * self.prf_hz)

    @property
    def swath_range_m(self) -> float:
        return self.num_fast_time * C0 / (2.0 * self.fs_hz)

    @property
    def velocity_axis_unambiguous_mps(self) -> float:
        return self.wavelength_m * self.prf_hz / 4.0

    def rx_positions_m(self) -> np.ndarray:
        """ULA element x positions centered at zero."""
        d = self.rx_spacing_m if self.rx_spacing_m is not None else self.wavelength_m / 2
        q = np.arange(self.num_rx) - (self.num_rx - 1) / 2
        return q * d


@dataclass
class Target:
    """Point target model.

    velocity_mps convention:
        positive means approaching the radar, so range decreases with time and
        Doppler frequency is positive by the lecture convention FD = 2v/lambda.
    """

    range_m: float
    velocity_mps: float
    angle_deg: float = 0.0
    amplitude: complex = 1.0 + 0.0j
    acceleration_mps2: float = 0.0
    name: str = "target"

    # Optional sinusoidal range micro-motion, useful for later micro-Doppler learning.
    micro_range_amp_m: float = 0.0
    micro_freq_hz: float = 0.0
    micro_phase_rad: float = 0.0

    def range_at(self, t_s: np.ndarray | float) -> np.ndarray | float:
        # Positive radial velocity means approaching => range decreases.
        base = self.range_m - self.velocity_mps * t_s - 0.5 * self.acceleration_mps2 * t_s**2
        if self.micro_range_amp_m != 0 and self.micro_freq_hz != 0:
            base = base + self.micro_range_amp_m * np.sin(2*np.pi*self.micro_freq_hz*t_s + self.micro_phase_rad)
        return base


def make_lfm_waveform(cfg: RadarConfig, normalize: bool = True) -> np.ndarray:
    """Complex baseband up-chirp from -B/2 to +B/2 over pulse_width."""
    n = np.arange(cfg.pulse_samples)
    t = n / cfg.fs_hz
    # centered in time to reduce irrelevant absolute phase
    tc = t - cfg.pulse_width_s / 2
    k = cfg.bandwidth_hz / cfg.pulse_width_s
    x = np.exp(1j * np.pi * k * tc**2)
    if normalize:
        x = x / np.sqrt(np.sum(np.abs(x) ** 2))
    return x.astype(np.complex128)


def eval_lfm_continuous(t_echo_s: np.ndarray, cfg: RadarConfig) -> np.ndarray:
    """Evaluate the same LFM waveform at arbitrary echo-local time.

    Returns zero outside [0, pulse_width_s].
    """
    out = np.zeros_like(t_echo_s, dtype=np.complex128)
    mask = (t_echo_s >= 0.0) & (t_echo_s < cfg.pulse_width_s)
    if np.any(mask):
        tc = t_echo_s[mask] - cfg.pulse_width_s / 2
        k = cfg.bandwidth_hz / cfg.pulse_width_s
        out[mask] = np.exp(1j * np.pi * k * tc**2)
        # match discrete waveform energy normalization approximately
        out[mask] /= np.sqrt(cfg.pulse_samples)
    return out


def steering_vector_ula(angle_deg: float | np.ndarray, cfg: RadarConfig) -> np.ndarray:
    """ULA steering vector a(theta), shape [num_rx] or [num_angles, num_rx]."""
    angles = np.atleast_1d(angle_deg).astype(float)
    theta = np.deg2rad(angles)
    x = cfg.rx_positions_m()[None, :]
    # Far-field plane-wave phase across x positions. Sign convention is consistent
    # within simulate() and beamform_angle_fft().
    a = np.exp(-1j * 2*np.pi / cfg.wavelength_m * x * np.sin(theta)[:, None])
    if np.ndim(angle_deg) == 0:
        return a[0]
    return a


def simulate_raw_datacube(cfg: RadarConfig, targets: Sequence[Target]) -> Tuple[np.ndarray, dict]:
    """Generate raw complex received datacube [CPI, pulse, fast_time, rx]."""
    rng = np.random.default_rng(cfg.seed)
    raw = np.zeros((cfg.num_cpi, cfg.num_pulses, cfg.num_fast_time, cfg.num_rx), dtype=np.complex128)
    t_fast = np.arange(cfg.num_fast_time) / cfg.fs_hz

    for icpi in range(cfg.num_cpi):
        for m in range(cfg.num_pulses):
            t_pulse = (icpi * cfg.num_pulses + m) * cfg.pri_s
            for tgt in targets:
                r = tgt.range_at(t_pulse)
                if r <= 0:
                    continue
                tau = 2.0 * r / C0
                # Echo-local time relative to the delayed transmit pulse.
                echo = eval_lfm_continuous(t_fast - tau, cfg)
                if not np.any(echo):
                    continue

                # Baseband propagation phase. This naturally creates Doppler across pulses.
                prop_phase = np.exp(-1j * 4*np.pi * r / cfg.wavelength_m)
                amp = tgt.amplitude
                if cfg.range_attenuation:
                    amp = amp / max(r, 1.0) ** 2
                a = steering_vector_ula(tgt.angle_deg, cfg)
                raw[icpi, m, :, :] += amp * prop_phase * echo[:, None] * a[None, :]

    if cfg.noise_power > 0:
        sigma = np.sqrt(cfg.noise_power / 2.0)
        noise = sigma * (rng.standard_normal(raw.shape) + 1j * rng.standard_normal(raw.shape))
        raw += noise

    meta = {
        "fast_time_s": t_fast,
        "fast_range_m": t_fast * C0 / 2,
        "targets": targets,
    }
    return raw, meta


def matched_filter_range(raw: np.ndarray, cfg: RadarConfig, waveform: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Range-compress raw data by matched filtering along fast time.

    Crops the convolution so range bin r corresponds approximately to delay n/Fs.
    Output shape: [CPI, pulse, range_bin, rx]
    """
    if waveform is None:
        waveform = make_lfm_waveform(cfg)
    h = np.conj(waveform[::-1])
    n_raw = raw.shape[2]
    n_h = len(h)
    n_conv = n_raw + n_h - 1
    n_fft = 1 << int(np.ceil(np.log2(n_conv)))

    H = np.fft.fft(h, n_fft)
    X = np.fft.fft(raw, n_fft, axis=2)
    Y = np.fft.ifft(X * H[None, None, :, None], axis=2)

    start = n_h - 1
    rc = Y[:, :, start:start+n_raw, :]
    range_axis_m = np.arange(n_raw) * C0 / (2.0 * cfg.fs_hz)
    return rc, range_axis_m


def doppler_process(range_cube: np.ndarray, cfg: RadarConfig, window: str = "hann", nfft: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Doppler FFT along slow time/pulse axis.

    Input shape:  [CPI, pulse, range, rx]
    Output shape: [CPI, range, doppler, rx]
    """
    m = range_cube.shape[1]
    if nfft is None:
        nfft = m
    if window == "hann":
        w = np.hanning(m)
    elif window == "rect":
        w = np.ones(m)
    else:
        raise ValueError("window must be 'hann' or 'rect'")

    xw = range_cube * w[None, :, None, None]
    rd = np.fft.fftshift(np.fft.fft(xw, n=nfft, axis=1), axes=1)
    rd = np.transpose(rd, (0, 2, 1, 3))
    fd_axis_hz = np.fft.fftshift(np.fft.fftfreq(nfft, d=cfg.pri_s))
    vel_axis_mps = fd_axis_hz * cfg.wavelength_m / 2.0
    return rd, fd_axis_hz, vel_axis_mps


def beamform_angles(rd_cube: np.ndarray, cfg: RadarConfig, angle_grid_deg: np.ndarray) -> np.ndarray:
    """Delay-and-sum beamforming over rx dimension.

    Input shape:  [CPI, range, doppler, rx]
    Output shape: [CPI, range, doppler, angle]
    """
    A = steering_vector_ula(angle_grid_deg, cfg)  # [angle, rx]
    # Weight with conjugated steering vector and normalize by number of antennas.
    out = np.einsum("crdn,an->crda", rd_cube, np.conj(A), optimize=True) / cfg.num_rx
    return out


def range_doppler_power(rd_cube: np.ndarray, integrate_rx: bool = True) -> np.ndarray:
    """Return power RDI. If integrate_rx, sum over receive antennas."""
    p = np.abs(rd_cube) ** 2
    return np.sum(p, axis=-1) if integrate_rx else p


def ca_cfar_2d(power_map: np.ndarray, guard: Tuple[int, int] = (2, 2), train: Tuple[int, int] = (8, 4), pfa: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """Simple 2D CA-CFAR on a power map.

    power_map shape: [range, doppler]
    guard = (# range guard cells each side, # doppler guard cells each side)
    train = (# range training cells each side, # doppler training cells each side)

    Returns:
        detections: boolean map
        threshold:  threshold map, NaN on untested borders
    """
    if power_map.ndim != 2:
        raise ValueError("power_map must be 2D [range, doppler]")
    gr, gd = guard
    tr, td = train
    nr, nd = power_map.shape
    threshold = np.full_like(power_map, np.nan, dtype=float)
    detections = np.zeros_like(power_map, dtype=bool)

    wr = gr + tr
    wd = gd + td
    total_cells = (2*wr + 1) * (2*wd + 1)
    guard_cells = (2*gr + 1) * (2*gd + 1)
    n_train = total_cells - guard_cells
    if n_train <= 0:
        raise ValueError("training region must contain at least one cell")

    alpha = n_train * (pfa ** (-1.0 / n_train) - 1.0)

    for r in range(wr, nr - wr):
        for d in range(wd, nd - wd):
            block = power_map[r-wr:r+wr+1, d-wd:d+wd+1]
            mask = np.ones(block.shape, dtype=bool)
            mask[tr:tr+2*gr+1, td:td+2*gd+1] = False
            noise_est = np.mean(block[mask])
            threshold[r, d] = alpha * noise_est
            detections[r, d] = power_map[r, d] > threshold[r, d]
    return detections, threshold


def make_default_targets() -> list[Target]:
    """A useful starting scenario for learning."""
    return [
        Target(range_m=2800.0, velocity_mps=18.0, angle_deg=12.0, amplitude=1.0, name="approaching strong"),
        Target(range_m=3300.0, velocity_mps=-12.0, angle_deg=-20.0, amplitude=0.75, name="receding"),
        Target(range_m=3100.0, velocity_mps=3.5, angle_deg=28.0, amplitude=0.45, name="slow weak"),
    ]
