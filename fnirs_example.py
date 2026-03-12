"""
Real fNIRS Analysis: Time series, COFRE spectrum, and Welch spectrum
=====================================================================
"""

from neuropipeline import fNIRS, fNIRSPreprocessor
import neuropipeline.fnirs.visualizer as nplv
from cofre import (
    COFREBank, COFREConfig, frequency_resolution, rise_time,
    optimal_tau_for_frequency, rise_time_seconds, optimal_tau_for_rise_time
)
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# ---------------------------------------------------------------------------
# 1. Load and split the fNIRS data
# ---------------------------------------------------------------------------
fnirs_obj = fNIRS(r"C:\Users\adama\dev\NIRWizard\examples\example_snirf_data\raw.snirf")
pp = fNIRSPreprocessor() # Create preprocesssor
pp.set_optical_density(True) # Configure
pp.set_hemoglobin_concentration(True)
pp.set_motion_correction(False)
pp.set_temporal_filtering(True, lowcut=0.005, highcut=2, order=100)
pp.set_detrending(False)
pp.set_normalization(False)

pp.print() # Inspect the settings

fnirs_obj.preprocess(pp) # Pass the preprocesser only

hbo_data, ch_names, hbr_data, _ = fnirs_obj.split()

channels = [34, 1, 2, 3]  # Use all 4 channels
fs = fnirs_obj.sampling_frequency

# hbo_data layout is (channels × samples)
print(f"hbo_data.shape: {hbo_data.shape}  (channels x samples)")

n_samples = hbo_data.shape[1]
duration = n_samples / fs
t = np.arange(n_samples) / fs

print(f"\nSampling frequency : {fs:.2f} Hz")
print(f"Duration           : {duration:.1f} s  ({duration / 60:.1f} min)")
print(f"Total samples      : {n_samples}")
print(f"Channels           : {[ch_names[c] for c in channels]}")
print(f"FFT resolution     : {1.0 / duration * 1e3:.4f} mHz")

# ---------------------------------------------------------------------------
# 2. Configure COFRE — adapt tau to recording length
# ---------------------------------------------------------------------------
# Rule of thumb: rise time (beta=0.5) should be at most ~50% of recording
# so the filter has enough settled data to produce a meaningful estimate.
#
# For a 177 s recording at 5.09 Hz (905 samples):
#   - tau = 8.65 -> rise time ~ 778 s (way too long!)
#   - tau = 5.0  -> rise time ~ 46 s  (26% of recording, reasonable)
#   - tau = 4.0  -> rise time ~ 19 s  (11% of recording, conservative)
#
# We lose some frequency resolution but get usable estimates.

# Pick tau so rise time (beta=0.5) ~ 35% of recording
target_rise_samples = 0.15 * n_samples
tau = optimal_tau_for_rise_time(target_rise_samples, beta=0.3)

# Don't let tau go below 3.0 (too coarse) or above 8.65 (paper's value)
tau = float(np.clip(tau, 3.0, 8.65))

print(f"\nAdapted tau = {tau:.4f} for {duration:.0f} s recording")

cofre_cfg = COFREConfig(
    fs=fs,
    freq_min_hz=0.003,
    freq_max_hz=fs / 2,  # up to Nyquist
    n_filters=300,
    tau=tau,
    alpha=0.75,
    beta=0.25,
    log_spacing=True,
)
bank = COFREBank(cofre_cfg)
bank.summary()

# Warn if rise time still exceeds recording
rt_sec = rise_time_seconds(tau, fs, beta=0.5)
if rt_sec > duration:
    print(f"\n!! WARNING: Rise time ({rt_sec:.1f}s) > recording ({duration:.1f}s).")
    print(f"   Low-frequency estimates (endothelial band) will be unreliable.")
    print(f"   Consider longer recordings for sub-20mHz analysis.\n")
elif rt_sec > 0.5 * duration:
    print(f"\n!! Note: Rise time ({rt_sec:.1f}s) is {rt_sec/duration*100:.0f}% of recording.")
    print(f"   Estimates are usable but variance will be elevated.\n")
else:
    print(f"\nRise time ({rt_sec:.1f}s) is {rt_sec/duration*100:.0f}% of recording -- good.\n")

# ---------------------------------------------------------------------------
# 3. ENMRC band definitions for shading
# ---------------------------------------------------------------------------
ENMRC_BANDS = [
    ("Endothelial",  0.003, 0.020, "#e74c3c", "E"),
    ("Neurogenic",   0.020, 0.050, "#e67e22", "N"),
    ("Myogenic",     0.050, 0.150, "#2ecc71", "M"),
    ("Respiratory",  0.150, 0.400, "#3498db", "R"),
    ("Cardiac",      0.400, 2.000, "#9b59b6", "C"),
]


def shade_enmrc(ax, label=False):
    """Add ENMRC band shading to a frequency-axis plot."""
    for name, flo, fhi, color, short in ENMRC_BANDS:
        ax.axvspan(flo * 1e3, fhi * 1e3, alpha=0.10, color=color)
        if label:
            mid = np.sqrt(flo * fhi) * 1e3
            ylim = ax.get_ylim()
            ax.text(mid, ylim[1] * 0.4, short, ha="center", fontsize=8,
                    fontweight="bold", color=color, alpha=0.8)


# ---------------------------------------------------------------------------
# 4. Compute spectra for channel 0
# ---------------------------------------------------------------------------
ch_idx = channels[0]
signal = hbo_data[ch_idx, :]
label  = ch_names[ch_idx] if ch_idx < len(ch_names) else f"Ch {ch_idx}"

print(f"\nComputing COFRE for channel: {label}...")
bank.process_vectorized(signal)
freqs_c_mhz, psd_c = bank.get_spectrum_mhz()

nperseg = min(n_samples, int(fs * 128))
if nperseg < 64:
    nperseg = n_samples
freqs_w, psd_w = welch(signal, fs=fs, nperseg=nperseg,
                       noverlap=nperseg // 2, window="hann")
freqs_w_mhz = freqs_w * 1e3

f_min_mhz = cofre_cfg.freq_min_hz * 1e3
f_max_mhz = fs / 2 * 1e3

# ---------------------------------------------------------------------------
# 5. Band-proportional x-axis scale
#    Each ENMRC band gets equal visual width, with linear interpolation inside.
# ---------------------------------------------------------------------------
_BAND_BOUNDS_MHZ = [b[1] * 1e3 for b in ENMRC_BANDS] + [ENMRC_BANDS[-1][2] * 1e3]
# = [3, 20, 50, 150, 400, 2000]

def _band_forward(f):
    """Frequency (mHz) → equal-band space [0, N_bands]."""
    f = np.asarray(f, dtype=float)
    out = np.empty_like(f)
    bounds = _BAND_BOUNDS_MHZ
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        mask = (f >= lo) & (f <= hi)
        out[mask] = i + (f[mask] - lo) / (hi - lo)
    out[f < bounds[0]]  = (f[f < bounds[0]]  - bounds[0])  / (bounds[1]  - bounds[0])
    out[f > bounds[-1]] = (len(bounds) - 1) + (f[f > bounds[-1]] - bounds[-2]) / (bounds[-1] - bounds[-2])
    return out

def _band_inverse(x):
    """Equal-band space → frequency (mHz)."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    bounds = _BAND_BOUNDS_MHZ
    n = len(bounds) - 1
    for i in range(n):
        mask = (x >= i) & (x <= i + 1)
        out[mask] = bounds[i] + (x[mask] - i) * (bounds[i + 1] - bounds[i])
    out[x < 0] = bounds[0] + x[x < 0] * (bounds[1] - bounds[0])
    out[x > n] = bounds[-2] + (x[x > n] - (n - 1)) * (bounds[-1] - bounds[-2])
    return out

# Tick positions at band boundaries (evenly spaced visually)
_BAND_TICKS = _BAND_BOUNDS_MHZ
_BAND_TICK_LABELS = ["3", "20", "50", "150", "400", "2000"]

# ---------------------------------------------------------------------------
# 6. README figure — 2×2: (COFRE | Welch) × (Band-proportional | Log)
# ---------------------------------------------------------------------------
COFRE_COLOR = "#2980b9"
WELCH_COLOR = "#2c3e50"

fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
fig.suptitle(
    f"COFRE vs Welch — fNIRS HbO "
    f"(τ={tau:.2f}, {cofre_cfg.n_filters} filters, fs={fs:.2f} Hz)",
    fontsize=12, fontweight="bold",
)

# (row, col) → (freqs, psd, color, xscale, yscale)
panels = [
    (axes[0, 0], freqs_c_mhz, psd_c, COFRE_COLOR, "band", "linear"),
    (axes[0, 1], freqs_c_mhz, psd_c, COFRE_COLOR, "log",  "log"),
    (axes[1, 0], freqs_w_mhz, psd_w, WELCH_COLOR, "band", "linear"),
    (axes[1, 1], freqs_w_mhz, psd_w, WELCH_COLOR, "log",  "log"),
]

col_titles = ["Band-proportional scale", "Log scale"]
row_labels  = ["COFRE PSD", "Welch PSD"]

for ax, freqs, psd, color, xscale, yscale in panels:
    is_band = xscale == "band"

    # Apply scale before plotting so data coordinates are correct
    if is_band:
        ax.set_xscale("function", functions=(_band_forward, _band_inverse))
    else:
        ax.set_xscale("log")
    ax.set_yscale(yscale)

    ax.plot(freqs, psd, color=color, linewidth=0.9, zorder=3)

    # ENMRC band shading + letter label at visual midpoint of each band
    for name, flo, fhi, bc, short in ENMRC_BANDS:
        flo_m, fhi_m = flo * 1e3, fhi * 1e3
        ax.axvspan(flo_m, fhi_m, alpha=0.13, color=bc, zorder=1)
        # Visual midpoint: arithmetic centre in band-space → maps to linear midpoint per band
        mid = (flo_m + fhi_m) / 2 if is_band else np.sqrt(flo_m * fhi_m)
        ax.text(mid, 1.01, short, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, fontweight="bold", color=bc)

    ax.set_xlim(f_min_mhz, f_max_mhz)
    ax.set_xlabel("Frequency (mHz)", fontsize=9)
    ax.set_ylabel("PSD", fontsize=9)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=8)

    if is_band:
        ax.set_xticks(_BAND_TICKS)
        ax.set_xticklabels(_BAND_TICK_LABELS)

# Row and column headers
for col, title in enumerate(col_titles):
    axes[0, col].set_title(title, fontsize=10, fontweight="bold", pad=14)

for row, rlabel in enumerate(row_labels):
    axes[row, 0].set_ylabel(rlabel, fontsize=10, fontweight="bold", labelpad=8)

fig.savefig("cofre_vs_welch_fnirs.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved: cofre_vs_welch_fnirs.png")