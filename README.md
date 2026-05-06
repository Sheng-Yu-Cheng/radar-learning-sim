# Radar Learning Simulator

這是一個用來學 ADPAR / phased-array radar signal processing 的小型 Python framework。它不是高保真商用雷達模擬器，而是一個**能穩定產生 datacube 與課堂式圖形的學習工具**。

你可以cite到開ADPAR這們課的老師黃彥銘，course link: https://sites.google.com/view/yenming/teaching#h.fq1z9eg4562

它目前支援：

- pulsed LFM waveform
- 多點目標：range、radial velocity、angle、amplitude
- raw complex datacube: `[CPI, pulse/slow-time, fast-time, rx antenna]`
- matched filtering / range compression
- Doppler FFT / Range-Doppler Image
- ULA steering vector 與 delay-and-sum beamforming
- Range-Angle Image
- 2D CA-CFAR on RDI
- Range-Time Image / Velocity-Time Image

## Install

建議建立乾淨環境：

```bash
cd radar_learning_sim
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run demo

```bash
python examples/run_demo.py
```

會在 `outputs/` 產生：

```text
01_waveform.png
02_range_profile.png
03_rdi_with_cfar.png
04_range_time.png
05_doppler_time.png
06_beam_pattern.png
07_range_angle.png
08_cfar_slice.png
```

## Main mental model

The raw datacube is:

```text
x[cpi, m, n, q]

m = slow-time pulse index      -> Doppler / velocity
n = fast-time sample index     -> range
q = antenna element index      -> angle
```

Processing pipeline:

```text
raw datacube
  -> matched_filter_range()
  -> doppler_process()
  -> beamform_angles()
  -> ca_cfar_2d()
  -> plots
```

## Velocity sign convention

In `Target`, positive `velocity_mps` means the target is approaching the radar, so its range decreases with time. This matches the lecture convention where positive radial velocity gives positive Doppler shift `FD = 2v / lambda`.

## A good way to learn with this framework

Try changing one parameter at a time:

- `bandwidth_hz`: see range resolution change.
- `num_pulses`: see Doppler resolution change.
- `prf_hz`: see unambiguous velocity range and Doppler aliasing.
- `num_rx`: see beamwidth change.
- `rx_spacing_m`: try larger than `lambda/2` and observe grating lobes.
- target `amplitude`: see sidelobe masking and CFAR behavior.
- target `angle_deg`: see range-angle separation.
- `train`, `guard`, `pfa` in `ca_cfar_2d`: see CFAR threshold changes.

## Files

```text
radar_learning_sim/
  core.py       # simulator and processing algorithms
  plotting.py   # plot helpers
examples/
  run_demo.py   # end-to-end demonstration
```
