import os
import numpy as np
import matplotlib.pyplot as plt


# =========================
# 0. 輸出資料夾設定
# =========================

output_dir = "./outputs/svd_simulation"
os.makedirs(output_dir, exist_ok=True)


# =========================
# 1. 基本參數設定
# =========================

np.random.seed(7)

num_pulses = 256
num_ranges = 128
prf = 1000.0
target_range_bin = 72

body_doppler_hz = 85.0
micro_doppler_hz = 180.0
micro_mod_depth = 0.45

direct_amp = 18.0
static_clutter_amp = 8.0
vegetation_amp = 2.5
target_amp = 1.2
noise_amp = 0.8


# =========================
# 2. 建立模擬資料矩陣 X
# X shape = slow-time × range
# =========================

t = np.arange(num_pulses) / prf
r = np.arange(num_ranges)

X = np.zeros((num_pulses, num_ranges), dtype=np.complex128)

direct_range_profile = np.exp(-0.5 * ((r - 20) / 5.0) ** 2)
direct_phase = np.exp(1j * 0.2)
X += direct_amp * np.outer(np.ones(num_pulses), direct_range_profile) * direct_phase

for center, amp, width in [(38, 7.0, 3.0), (52, 5.5, 4.0), (95, 4.5, 5.5)]:
    profile = np.exp(-0.5 * ((r - center) / width) ** 2)
    phase = np.exp(1j * np.random.uniform(0, 2 * np.pi))
    X += static_clutter_amp * amp / 7.0 * np.outer(np.ones(num_pulses), profile) * phase

veg_profile = np.exp(-0.5 * ((r - 60) / 9.0) ** 2)
veg_time = (
    1.0
    + 0.25 * np.sin(2 * np.pi * 8 * t)
    + 0.15 * np.sin(2 * np.pi * 17 * t + 0.8)
)
veg_phase = np.exp(1j * 0.4 * np.sin(2 * np.pi * 5 * t))
X += vegetation_amp * np.outer(veg_time * veg_phase, veg_profile)

target_profile = np.exp(-0.5 * ((r - target_range_bin) / 1.8) ** 2)
body_phase = np.exp(1j * 2 * np.pi * body_doppler_hz * t)

micro_mod = 1.0 + micro_mod_depth * np.sin(2 * np.pi * micro_doppler_hz * t)

sideband_phase_1 = 0.25 * np.exp(
    1j * 2 * np.pi * (body_doppler_hz + micro_doppler_hz) * t
)
sideband_phase_2 = 0.25 * np.exp(
    1j * 2 * np.pi * (body_doppler_hz - micro_doppler_hz) * t
)

target_time = body_phase * micro_mod + sideband_phase_1 + sideband_phase_2
X += target_amp * np.outer(target_time, target_profile)

noise = noise_amp * (
    np.random.randn(num_pulses, num_ranges)
    + 1j * np.random.randn(num_pulses, num_ranges)
) / np.sqrt(2)
X += noise


# =========================
# 3. SVD clutter removal
# =========================

def fixed_svd_remove(X, K):
    """
    一般 SVD：固定移除前 K 個 singular components。
    """
    U, s, Vh = np.linalg.svd(X, full_matrices=False)

    X_clutter = np.zeros_like(X)
    for k in range(K):
        X_clutter += s[k] * np.outer(U[:, k], Vh[k, :])

    X_clean = X - X_clutter
    return X_clean, s, K


def adaptive_svd_remove(X, max_remove=8, gap_threshold=2.0):
    """
    adaptive SVD：
    根據相鄰奇異值比例 sigma_k / sigma_{k+1} 自動選擇 K。
    """
    U, s, Vh = np.linalg.svd(X, full_matrices=False)

    search_len = min(max_remove, len(s) - 1)
    ratios = s[:search_len] / (s[1:search_len + 1] + 1e-12)

    best_idx = np.argmax(ratios)
    best_gap = ratios[best_idx]

    if best_gap >= gap_threshold:
        K = best_idx + 1
    else:
        K = 1

    X_clutter = np.zeros_like(X)
    for k in range(K):
        X_clutter += s[k] * np.outer(U[:, k], Vh[k, :])

    X_clean = X - X_clutter
    return X_clean, s, K, ratios


X_fixed, s_fixed, K_fixed = fixed_svd_remove(X, K=4)

X_adapt, s_adapt, K_adapt, ratios = adaptive_svd_remove(
    X,
    max_remove=10,
    gap_threshold=2.0
)


# =========================
# 4. Range-Doppler map
# =========================

def range_doppler_map(X):
    """
    對 slow-time 方向做 FFT，得到 Doppler × range map。
    """
    window = np.hamming(X.shape[0])[:, None]
    RD = np.fft.fftshift(np.fft.fft(X * window, axis=0), axes=0)
    RD_db = 20 * np.log10(np.abs(RD) + 1e-12)
    RD_db -= RD_db.max()
    return RD_db


RD_raw = range_doppler_map(X)
RD_fixed = range_doppler_map(X_fixed)
RD_adapt = range_doppler_map(X_adapt)

doppler_axis = np.fft.fftshift(np.fft.fftfreq(num_pulses, d=1 / prf))
range_axis = np.arange(num_ranges)


# =========================
# 5. 簡單評估 target contrast
# =========================

def target_contrast(RD_db, target_range_bin, target_doppler_hz, doppler_axis):
    """
    粗略計算：
    目標附近 peak 與 zero-Doppler clutter 區域的差距。
    """
    target_doppler_idx = np.argmin(np.abs(doppler_axis - target_doppler_hz))
    zero_doppler_idx = np.argmin(np.abs(doppler_axis - 0.0))

    target_region = RD_db[
        max(0, target_doppler_idx - 1): target_doppler_idx + 2,
        max(0, target_range_bin - 2): target_range_bin + 3
    ]

    clutter_region = RD_db[
        max(0, zero_doppler_idx - 3): zero_doppler_idx + 4,
        max(0, target_range_bin - 8): target_range_bin + 9
    ]

    target_peak = np.max(target_region)
    clutter_peak = np.max(clutter_region)
    return target_peak, clutter_peak, target_peak - clutter_peak


raw_score = target_contrast(RD_raw, target_range_bin, body_doppler_hz, doppler_axis)
fixed_score = target_contrast(RD_fixed, target_range_bin, body_doppler_hz, doppler_axis)
adapt_score = target_contrast(RD_adapt, target_range_bin, body_doppler_hz, doppler_axis)

print("===== SVD 設定 =====")
print(f"固定 SVD 移除 K = {K_fixed}")
print(f"Adaptive SVD 自動選 K = {K_adapt}")
print()

print("===== 目標 peak 與 zero-Doppler clutter 對比 =====")
print(f"原始資料:      target={raw_score[0]:6.2f} dB, clutter={raw_score[1]:6.2f} dB, contrast={raw_score[2]:6.2f} dB")
print(f"固定 SVD:      target={fixed_score[0]:6.2f} dB, clutter={fixed_score[1]:6.2f} dB, contrast={fixed_score[2]:6.2f} dB")
print(f"Adaptive SVD:  target={adapt_score[0]:6.2f} dB, clutter={adapt_score[1]:6.2f} dB, contrast={adapt_score[2]:6.2f} dB")


# =========================
# 6. 畫圖並存檔
# =========================

def save_current_fig(filename):
    """
    儲存目前的 matplotlib figure，然後關閉。
    """
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_rd(RD_db, title, filename):
    plt.figure(figsize=(9, 5))
    plt.imshow(
        RD_db,
        aspect="auto",
        origin="lower",
        extent=[range_axis[0], range_axis[-1], doppler_axis[0], doppler_axis[-1]],
        vmin=-55,
        vmax=0
    )
    plt.colorbar(label="Normalized magnitude (dB)")
    plt.axvline(target_range_bin, linestyle="--", linewidth=1)
    plt.axhline(body_doppler_hz, linestyle="--", linewidth=1)
    plt.xlabel("Range bin")
    plt.ylabel("Doppler frequency (Hz)")
    plt.title(title)
    plt.tight_layout()
    save_current_fig(filename)


# 6-1. 奇異值曲線
plt.figure(figsize=(8, 4))
plt.semilogy(s_adapt, marker="o")
plt.axvline(K_adapt - 0.5, linestyle="--", label=f"Adaptive K = {K_adapt}")
plt.axvline(K_fixed - 0.5, linestyle=":", label=f"Fixed K = {K_fixed}")
plt.xlabel("Singular value index")
plt.ylabel("Singular value")
plt.title("Singular values of the slow-time x range matrix")
plt.legend()
plt.tight_layout()
save_current_fig("01_singular_values.png")


# 6-2. 相鄰奇異值比例
plt.figure(figsize=(8, 4))
plt.plot(np.arange(1, len(ratios) + 1), ratios, marker="o")
plt.axhline(2.0, linestyle="--", label="gap threshold")
plt.xlabel("k")
plt.ylabel("sigma_k / sigma_{k+1}")
plt.title("Adaptive SVD rank selection by singular-value gap")
plt.legend()
plt.tight_layout()
save_current_fig("02_singular_value_gap_ratio.png")


# 6-3. Range-Doppler maps
plot_rd(
    RD_raw,
    "Original range-Doppler map",
    "03_original_range_doppler.png"
)

plot_rd(
    RD_fixed,
    f"Fixed SVD clutter removal, K = {K_fixed}",
    "04_fixed_svd_range_doppler.png"
)

plot_rd(
    RD_adapt,
    f"Adaptive SVD clutter removal, K = {K_adapt}",
    "05_adaptive_svd_range_doppler.png"
)


# 6-4. 目標 range bin 的 Doppler cut
plt.figure(figsize=(9, 5))
plt.plot(doppler_axis, RD_raw[:, target_range_bin], label="Original")
plt.plot(doppler_axis, RD_fixed[:, target_range_bin], label=f"Fixed SVD K={K_fixed}")
plt.plot(doppler_axis, RD_adapt[:, target_range_bin], label=f"Adaptive SVD K={K_adapt}")
plt.axvline(body_doppler_hz, linestyle="--", label="Drone body Doppler")
plt.axvline(body_doppler_hz + micro_doppler_hz, linestyle=":", label="Micro-Doppler sideband")
plt.axvline(body_doppler_hz - micro_doppler_hz, linestyle=":")
plt.xlabel("Doppler frequency (Hz)")
plt.ylabel("Normalized magnitude (dB)")
plt.title("Doppler cut at drone range bin")
plt.ylim([-70, 5])
plt.legend()
plt.tight_layout()
save_current_fig("06_doppler_cut_at_drone_range.png")


print()
print(f"所有圖片已儲存到：{output_dir}")