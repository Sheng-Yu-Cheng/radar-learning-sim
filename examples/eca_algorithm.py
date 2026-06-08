from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# 1. Basic signal utilities
# ============================================================

def make_reference_signal(num_samples: int, seed: int = 0) -> np.ndarray:
    """
    Generate a constant-envelope QPSK-like reference signal.
    Compared with Gaussian complex noise, this gives cleaner CAF behaviour,
    which is better for understanding ECA.
    """
    rng = np.random.default_rng(seed)
    phases = rng.integers(0, 4, size=num_samples)
    ref = np.exp(1j * np.pi / 2 * phases)
    return ref.astype(complex)


def delayed_signal(x: np.ndarray, delay: int) -> np.ndarray:
    """Return delayed version x[n-delay]."""
    if delay < 0:
        raise ValueError("delay must be non-negative")

    y = np.zeros_like(x, dtype=complex)
    if delay == 0:
        return x.copy()
    if delay < len(x):
        y[delay:] = x[:-delay]
    return y


def doppler_shift(x: np.ndarray, doppler_hz: float, fs: float) -> np.ndarray:
    """Apply Doppler shift to a complex signal."""
    n = np.arange(len(x))
    return x * np.exp(1j * 2 * np.pi * doppler_hz * n / fs)


# ============================================================
# 2. Simulate passive radar surveillance signal
# ============================================================

def simulate_surveillance_signal(ref: np.ndarray, fs: float):
    """
    Simulate surveillance channel signal.

    surveillance =
        strong direct-path signal
      + static clutter
      + static multipath
      + weak moving target
      + noise
    """
    num_samples = len(ref)
    rng = np.random.default_rng(1)

    # Strong direct-path interference: nearly no delay, zero Doppler
    direct = 12.0 * delayed_signal(ref, delay=2)

    # Static clutter: delayed reference copies, zero Doppler
    clutter_1 = 7.0 * delayed_signal(ref, delay=25)
    clutter_2 = 5.0 * delayed_signal(ref, delay=55)
    clutter_3 = 3.5 * delayed_signal(ref, delay=90)

    # Static multipath
    multipath = 4.0 * delayed_signal(ref, delay=130)

    # Choose target Doppler exactly on an FFT bin for clean peak visibility.
    # If doppler_bins=512 and fs=2000 Hz, bin spacing = 3.90625 Hz.
    target_delay = 80
    target_fd = 125.0  # 32 * 3.90625 Hz
    target_amp = 0.55
    target = target_amp * doppler_shift(
        delayed_signal(ref, target_delay),
        doppler_hz=target_fd,
        fs=fs,
    )

    # Slightly lower noise so the target peak becomes visually clear.
    noise_amp = 0.20
    noise = noise_amp * (rng.normal(size=num_samples) + 1j * rng.normal(size=num_samples))

    surveillance = direct + clutter_1 + clutter_2 + clutter_3 + multipath + target + noise

    truth = {
        'direct_delay': 2,
        'clutter_delays': [25, 55, 90],
        'multipath_delay': 130,
        'target_delay': target_delay,
        'target_fd': target_fd,
    }
    return surveillance, truth


# ============================================================
# 3. ECA cancellation
# ============================================================

def build_reference_delay_matrix(ref: np.ndarray, num_taps: int) -> np.ndarray:
    """Build matrix whose columns are delayed versions of the reference signal."""
    num_samples = len(ref)
    S = np.zeros((num_samples, num_taps), dtype=complex)
    for k in range(num_taps):
        S[:, k] = delayed_signal(ref, k)
    return S


def eca_cancel(surv: np.ndarray, ref: np.ndarray, num_taps: int):
    """
    Extensive Cancellation Algorithm.

    Model:
        surv[n] ≈ sum_k a[k] ref[n-k] + residual[n]
    """
    S = build_reference_delay_matrix(ref, num_taps)
    a_hat, *_ = np.linalg.lstsq(S, surv, rcond=None)
    estimated_disturbance = S @ a_hat
    residual = surv - estimated_disturbance
    return residual, estimated_disturbance, a_hat


# ============================================================
# 4. Simplified CAF / range-Doppler map
# ============================================================

def caf(surv: np.ndarray, ref: np.ndarray, max_delay: int, doppler_bins: int, fs: float):
    """
    Simplified Cross Ambiguity Function.
    For each delay, correlate with delayed reference then FFT over time.
    Output shape: [delay, Doppler]
    """
    num_samples = len(ref)
    rd = np.zeros((max_delay, doppler_bins), dtype=complex)
    window = np.hanning(num_samples)

    for delay in range(max_delay):
        ref_d = delayed_signal(ref, delay)
        product = surv * np.conj(ref_d) * window
        rd[delay, :] = np.fft.fftshift(np.fft.fft(product, n=doppler_bins))

    delay_axis = np.arange(max_delay)
    fd_axis = np.fft.fftshift(np.fft.fftfreq(doppler_bins, d=1 / fs))
    return rd, delay_axis, fd_axis


def to_db(x: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.abs(x) + 1e-12)


# ============================================================
# 5. Plot helpers
# ============================================================

def save_figure(fig, output_dir: Path, filename: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=180, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close(fig)


def plot_range_doppler(rd_map, delay_axis, fd_axis, title, output_dir, filename,
                       target_delay=None, target_fd=None, vmin=None, vmax=None):
    power_db = to_db(rd_map)
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(
        power_db,
        aspect='auto',
        origin='lower',
        extent=[fd_axis[0], fd_axis[-1], delay_axis[0], delay_axis[-1]],
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(im, ax=ax, label='Magnitude (dB)')
    ax.set_xlabel('Doppler frequency (Hz)')
    ax.set_ylabel('Delay bin')
    ax.set_title(title)
    if target_delay is not None and target_fd is not None:
        ax.scatter([target_fd], [target_delay], marker='x', s=120, linewidths=2,
                   label='True moving target', color='red')
        ax.legend(loc='upper right')
    fig.tight_layout()
    save_figure(fig, output_dir, filename)


def plot_eca_coefficients(a_hat, truth, output_dir, filename):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.stem(np.abs(a_hat), basefmt=' ')
    expected_delays = [truth['direct_delay']] + truth['clutter_delays'] + [truth['multipath_delay']]
    for d in expected_delays:
        ax.axvline(d, linestyle='--', alpha=0.5)
    ax.set_xlabel('Reference delay tap k')
    ax.set_ylabel('|a[k]|')
    ax.set_title('ECA estimated coefficients')
    fig.tight_layout()
    save_figure(fig, output_dir, filename)


def plot_zero_doppler_cut(rd_before, rd_after, delay_axis, fd_axis, truth, output_dir, filename):
    zero_idx = np.argmin(np.abs(fd_axis))
    before = to_db(rd_before[:, zero_idx])
    after = to_db(rd_after[:, zero_idx])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(delay_axis, before, label='Before ECA')
    ax.plot(delay_axis, after, label='After ECA')

    expected_delays = [truth['direct_delay']] + truth['clutter_delays'] + [truth['multipath_delay']]
    for d in expected_delays:
        ax.axvline(d, linestyle='--', alpha=0.35, color='gray')

    ax.set_xlabel('Delay bin')
    ax.set_ylabel('Magnitude at zero Doppler (dB)')
    ax.set_title('Zero-Doppler disturbance suppression')
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_dir, filename)


def plot_target_zoom(rd_map, fd_axis, title, target_delay, target_fd, output_dir, filename,
                     delay_half_width=14, fd_half_bins=24):
    power_db = to_db(rd_map)
    target_fd_idx = np.argmin(np.abs(fd_axis - target_fd))

    d0 = max(target_delay - delay_half_width, 0)
    d1 = min(target_delay + delay_half_width + 1, power_db.shape[0])
    f0 = max(target_fd_idx - fd_half_bins, 0)
    f1 = min(target_fd_idx + fd_half_bins + 1, power_db.shape[1])

    local = power_db[d0:d1, f0:f1]
    vmin = np.percentile(local, 10)
    vmax = np.percentile(local, 99.8)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        local,
        aspect='auto',
        origin='lower',
        extent=[fd_axis[f0], fd_axis[f1 - 1], d0, d1 - 1],
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(im, ax=ax, label='Magnitude (dB)')
    ax.scatter([target_fd], [target_delay], marker='x', s=140, linewidths=2,
               label='True moving target', color='red')
    ax.set_xlabel('Doppler frequency (Hz)')
    ax.set_ylabel('Delay bin')
    ax.set_title(title)
    ax.legend(loc='upper right')
    fig.tight_layout()
    save_figure(fig, output_dir, filename)


def plot_target_doppler_cut(rd_before, rd_after, fd_axis, target_delay, target_fd, output_dir, filename):
    before = to_db(rd_before[target_delay, :])
    after = to_db(rd_after[target_delay, :])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(fd_axis, before, label='Before ECA')
    ax.plot(fd_axis, after, label='After ECA')
    ax.axvline(target_fd, linestyle='--', label='True target Doppler', color='red')
    ax.set_xlim(target_fd - 250, target_fd + 250)
    ax.set_xlabel('Doppler frequency (Hz)')
    ax.set_ylabel('Magnitude at target delay (dB)')
    ax.set_title(f'Doppler cut at target delay bin = {target_delay}')
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_dir, filename)


def plot_target_delay_cut(rd_before, rd_after, delay_axis, fd_axis, target_delay, target_fd, output_dir, filename):
    target_fd_idx = np.argmin(np.abs(fd_axis - target_fd))
    before = to_db(rd_before[:, target_fd_idx])
    after = to_db(rd_after[:, target_fd_idx])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(delay_axis, before, label='Before ECA')
    ax.plot(delay_axis, after, label='After ECA')
    ax.axvline(target_delay, linestyle='--', label='True target delay', color='red')
    ax.set_xlim(max(0, target_delay - 40), target_delay + 40)
    ax.set_xlabel('Delay bin')
    ax.set_ylabel(f'Magnitude at Doppler = {target_fd:.2f} Hz (dB)')
    ax.set_title('Delay cut at target Doppler')
    ax.legend()
    fig.tight_layout()
    save_figure(fig, output_dir, filename)


def plot_contrast_map_local(rd_map, delay_axis, fd_axis, title, target_delay, target_fd, output_dir, filename,
                            delay_half_width=18, fd_half_bins=36):
    """
    Local contrast map around the target.
    Contrast = cell power - median of the same delay row.
    Restricting to a local window makes the target easier to see.
    """
    power_db = to_db(rd_map)
    row_background = np.median(power_db, axis=1, keepdims=True)
    contrast = power_db - row_background

    target_fd_idx = np.argmin(np.abs(fd_axis - target_fd))
    d0 = max(target_delay - delay_half_width, 0)
    d1 = min(target_delay + delay_half_width + 1, contrast.shape[0])
    f0 = max(target_fd_idx - fd_half_bins, 0)
    f1 = min(target_fd_idx + fd_half_bins + 1, contrast.shape[1])

    local = contrast[d0:d1, f0:f1]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        local,
        aspect='auto',
        origin='lower',
        extent=[fd_axis[f0], fd_axis[f1 - 1], d0, d1 - 1],
        vmin=-3,
        vmax=12,
    )
    fig.colorbar(im, ax=ax, label='Contrast over row median (dB)')
    ax.scatter([target_fd], [target_delay], marker='x', s=140, linewidths=2,
               label='True moving target', color='red')
    ax.set_xlabel('Doppler frequency (Hz)')
    ax.set_ylabel('Delay bin')
    ax.set_title(title)
    ax.legend(loc='upper right')
    fig.tight_layout()
    save_figure(fig, output_dir, filename)


# ============================================================
# 6. Diagnostics
# ============================================================

def print_target_contrast(rd_before, rd_after, fd_axis, target_delay, target_fd):
    target_fd_idx = np.argmin(np.abs(fd_axis - target_fd))

    def compute_contrast(rd_map):
        power_db = to_db(rd_map)
        target_level = power_db[target_delay, target_fd_idx]

        d0 = max(target_delay - 8, 0)
        d1 = min(target_delay + 9, power_db.shape[0])
        f0 = max(target_fd_idx - 8, 0)
        f1 = min(target_fd_idx + 9, power_db.shape[1])
        patch = power_db[d0:d1, f0:f1].copy()

        center_d = target_delay - d0
        center_f = target_fd_idx - f0
        patch[
            max(center_d - 1, 0): min(center_d + 2, patch.shape[0]),
            max(center_f - 1, 0): min(center_f + 2, patch.shape[1]),
        ] = np.nan

        background = np.nanmedian(patch)
        contrast = target_level - background
        return target_level, background, contrast

    b_level, b_bg, b_contrast = compute_contrast(rd_before)
    a_level, a_bg, a_contrast = compute_contrast(rd_after)

    print('\n=== Moving target local contrast ===')
    print('Before ECA:')
    print(f'  target level     = {b_level:.2f} dB')
    print(f'  local background = {b_bg:.2f} dB')
    print(f'  contrast         = {b_contrast:.2f} dB')
    print('After ECA:')
    print(f'  target level     = {a_level:.2f} dB')
    print(f'  local background = {a_bg:.2f} dB')
    print(f'  contrast         = {a_contrast:.2f} dB')


def print_diagnostics(rd_before, rd_after, fd_axis, truth):
    zero_idx = np.argmin(np.abs(fd_axis))
    target_delay = truth['target_delay']
    target_fd = truth['target_fd']
    target_fd_idx = np.argmin(np.abs(fd_axis - target_fd))

    disturbance_delays = [truth['direct_delay']] + truth['clutter_delays'] + [truth['multipath_delay']]

    print('\n=== Zero-Doppler disturbance suppression ===')
    for d in disturbance_delays:
        before_db = to_db(rd_before[d, zero_idx])
        after_db = to_db(rd_after[d, zero_idx])
        print(
            f'delay {d:3d}: '
            f'before = {before_db:7.2f} dB, '
            f'after = {after_db:7.2f} dB, '
            f'reduction = {before_db - after_db:7.2f} dB'
        )

    before_target = to_db(rd_before[target_delay, target_fd_idx])
    after_target = to_db(rd_after[target_delay, target_fd_idx])

    print('\n=== Moving target cell level ===')
    print(f'target delay = {target_delay}, target Doppler = {target_fd} Hz')
    print(f'before = {before_target:.2f} dB')
    print(f'after  = {after_target:.2f} dB')

    print_target_contrast(rd_before, rd_after, fd_axis, target_delay, target_fd)


# ============================================================
# 7. Main experiment
# ============================================================

def main():
    output_dir = Path('./outputs/eca_algorithm')

    fs = 2000.0
    num_samples = 2048
    max_delay = 160
    doppler_bins = 512
    num_taps = 150  # must cover all clutter / multipath delays

    ref = make_reference_signal(num_samples)
    surv, truth = simulate_surveillance_signal(ref, fs)
    eca_out, estimated_disturbance, a_hat = eca_cancel(surv=surv, ref=ref, num_taps=num_taps)

    rd_before, delay_axis, fd_axis = caf(surv=surv, ref=ref, max_delay=max_delay, doppler_bins=doppler_bins, fs=fs)
    rd_after, _, _ = caf(surv=eca_out, ref=ref, max_delay=max_delay, doppler_bins=doppler_bins, fs=fs)

    target_delay = truth['target_delay']
    target_fd = truth['target_fd']

    before_db = to_db(rd_before)
    after_db = to_db(rd_after)

    # Full maps with their own scales.
    plot_range_doppler(
        rd_before, delay_axis, fd_axis,
        title='Before ECA: raw range-Doppler map',
        output_dir=output_dir,
        filename='01_before_raw_range_doppler.png',
        target_delay=target_delay,
        target_fd=target_fd,
        vmin=np.percentile(before_db, 5),
        vmax=np.percentile(before_db, 99.7),
    )
    plot_range_doppler(
        rd_after, delay_axis, fd_axis,
        title='After ECA: raw range-Doppler map (own color scale)',
        output_dir=output_dir,
        filename='02_after_raw_range_doppler_own_scale.png',
        target_delay=target_delay,
        target_fd=target_fd,
        vmin=np.percentile(after_db, 5),
        vmax=np.percentile(after_db, 99.7),
    )

    # Shared scale view to prove global suppression.
    plot_range_doppler(
        rd_after, delay_axis, fd_axis,
        title='After ECA with BEFORE-ECA color scale: global suppression view',
        output_dir=output_dir,
        filename='03_after_global_suppression_shared_scale.png',
        target_delay=target_delay,
        target_fd=target_fd,
        vmin=np.percentile(before_db, 5),
        vmax=np.percentile(before_db, 99.7),
    )

    # ECA-specific explanatory plots.
    plot_eca_coefficients(a_hat, truth, output_dir, '04_eca_estimated_coefficients.png')
    plot_zero_doppler_cut(rd_before, rd_after, delay_axis, fd_axis, truth, output_dir,
                          '05_zero_doppler_disturbance_suppression.png')

    # Target visibility plots.
    plot_target_zoom(rd_before, fd_axis,
                     title='Before ECA: target-region zoom',
                     target_delay=target_delay,
                     target_fd=target_fd,
                     output_dir=output_dir,
                     filename='06_before_target_region_zoom.png')
    plot_target_zoom(rd_after, fd_axis,
                     title='After ECA: target-region zoom',
                     target_delay=target_delay,
                     target_fd=target_fd,
                     output_dir=output_dir,
                     filename='07_after_target_region_zoom.png')

    plot_target_doppler_cut(rd_before, rd_after, fd_axis, target_delay, target_fd,
                            output_dir, '08_target_delay_doppler_cut.png')
    plot_target_delay_cut(rd_before, rd_after, delay_axis, fd_axis, target_delay, target_fd,
                          output_dir, '09_target_doppler_delay_cut.png')

    plot_contrast_map_local(rd_before, delay_axis, fd_axis,
                            title='Before ECA: local contrast map around target',
                            target_delay=target_delay,
                            target_fd=target_fd,
                            output_dir=output_dir,
                            filename='10_before_local_contrast_map.png')
    plot_contrast_map_local(rd_after, delay_axis, fd_axis,
                            title='After ECA: local contrast map around target',
                            target_delay=target_delay,
                            target_fd=target_fd,
                            output_dir=output_dir,
                            filename='11_after_local_contrast_map.png')

    print_diagnostics(rd_before, rd_after, fd_axis, truth)
    print(f'\nAll figures saved under: {output_dir.resolve()}')


if __name__ == '__main__':
    main()
