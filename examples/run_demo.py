"""
Run this demo from the project root:

    python examples/run_demo.py

It creates a small radar datacube and writes figures to ./outputs.
"""
from pathlib import Path
import sys
import numpy as np

# Allow running this file directly from examples/ or project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from radar_learning_sim import (
    RadarConfig,
    Target,
    make_default_targets,
    simulate_raw_datacube,
    matched_filter_range,
    doppler_process,
    range_doppler_power,
    beamform_angles,
    ca_cfar_2d,
)
from radar_learning_sim.plotting import (
    plot_waveform,
    plot_range_profile,
    plot_rdi,
    plot_range_time,
    plot_doppler_time,
    plot_beam_pattern,
    plot_range_angle,
    plot_cfar_slice,
)


def main():
    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True)

    cfg = RadarConfig(
        fc_hz=10e9,
        fs_hz=20e6,
        prf_hz=5e3,
        pulse_width_s=5e-6,
        bandwidth_hz=10e6,
        num_fast_time=512,
        num_pulses=64,
        num_cpi=4,
        num_rx=8,
        noise_power=2e-4,
        seed=3,
    )

    # Positive velocity means approaching; negative means receding.
    targets = make_default_targets()

    print("Radar summary")
    print(f"  wavelength: {cfg.wavelength_m:.4f} m")
    print(f"  swath range: {cfg.swath_range_m:.1f} m")
    print(f"  unambiguous velocity: +/-{cfg.velocity_axis_unambiguous_mps:.1f} m/s")
    print("Targets")
    for t in targets:
        print(f"  {t.name:18s}: range={t.range_m:.1f} m, v={t.velocity_mps:+.1f} m/s, angle={t.angle_deg:+.1f} deg")

    raw, meta = simulate_raw_datacube(cfg, targets)
    range_cube, range_axis_m = matched_filter_range(raw, cfg)
    rd_cube, fd_axis_hz, vel_axis_mps = doppler_process(range_cube, cfg, window="hann", nfft=64)
    rdi_power = range_doppler_power(rd_cube, integrate_rx=True)

    angle_grid_deg = np.linspace(-60, 60, 121)
    angle_cube = beamform_angles(rd_cube, cfg, angle_grid_deg)

    det, th = ca_cfar_2d(rdi_power[0], guard=(2, 2), train=(8, 6), pfa=1e-3)

    plot_waveform(cfg, outdir / "01_waveform.png")
    plot_range_profile(range_cube, range_axis_m, cpi=0, pulse=0, rx=0, max_range_m=3800, savepath=outdir / "02_range_profile.png")
    plot_rdi(rdi_power, range_axis_m, vel_axis_mps, cpi=0, max_range_m=3800, detections=det, savepath=outdir / "03_rdi_with_cfar.png")
    plot_range_time(range_cube, range_axis_m, cfg, rx=0, max_range_m=3800, savepath=outdir / "04_range_time.png")
    plot_doppler_time(rdi_power, vel_axis_mps, range_axis_m, range_gate_m=(2500, 3500), savepath=outdir / "05_doppler_time.png")
    plot_beam_pattern(cfg, steer_angle_deg=12, savepath=outdir / "06_beam_pattern.png")
    plot_range_angle(angle_cube, range_axis_m, angle_grid_deg, cpi=0, max_range_m=3800, savepath=outdir / "07_range_angle.png")
    plot_cfar_slice(rdi_power[0], th, range_axis_m, vel_axis_mps, savepath=outdir / "08_cfar_slice.png")

    print(f"Done. Figures are in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
