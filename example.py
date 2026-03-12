"""
COFRE vs Welch PSD comparison
==============================
Generates a synthetic signal with two sinusoidal components buried in noise,
then estimates the PSD with both COFRE and Welch's method side-by-side.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# Use the installed package, or fall back to the local module
try:
    from cofre_spectrum import cofre_estimate
except ImportError:
    from cofre import cofre_estimate


# ---------------------------------------------------------------------------
# Synthetic signal
# ---------------------------------------------------------------------------
fs     = 20.0       # Hz  — sampling rate
T      = 600.0      # s   — signal duration
t      = np.arange(0, T, 1.0 / fs)

f1, f2 = 0.05, 0.3  # Hz  — two target sinusoids

signal = (
    2.0 * np.sin(2 * np.pi * f1 * t) +   # strong component at f1
    0.8 * np.sin(2 * np.pi * f2 * t) +   # weaker component at f2
    np.random.default_rng(42).normal(0, 1.0, len(t))  # white noise
)

# ---------------------------------------------------------------------------
# COFRE estimate
# ---------------------------------------------------------------------------
cofre_freqs, cofre_psd = cofre_estimate(
    signal,
    fs=fs,
    freq_min_hz=0.003,
    freq_max_hz=2.0,
    n_filters=300,
    tau=8.65,
)

# ---------------------------------------------------------------------------
# Welch estimate
# ---------------------------------------------------------------------------
# nperseg controls frequency resolution: longer segment = finer resolution
nperseg = int(fs * 120)   # 120-second windows
welch_freqs, welch_psd = welch(signal, fs=fs, nperseg=nperseg)
# Trim to the same frequency range as COFRE
mask = (welch_freqs >= cofre_freqs[0]) & (welch_freqs <= cofre_freqs[-1])
welch_freqs = welch_freqs[mask]
welch_psd   = welch_psd[mask]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig.suptitle("PSD Comparison: COFRE vs Welch", fontsize=14)

for ax, freqs, psd, label, color in [
    (axes[0], cofre_freqs, cofre_psd, "COFRE (τ=8.65, 300 filters)", "steelblue"),
    (axes[1], welch_freqs, welch_psd, f"Welch (nperseg={nperseg})", "darkorange"),
]:
    ax.semilogy(freqs, psd, color=color, linewidth=1.2, label=label)
    for f, name in [(f1, "f₁"), (f2, "f₂")]:
        ax.axvline(f, color="red", linestyle="--", linewidth=0.9, alpha=0.7,
                   label=f"{name} = {f} Hz" if freqs is cofre_freqs else None)
    ax.set_ylabel("PSD")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

axes[1].set_xlabel("Frequency (Hz)")
plt.tight_layout()
plt.savefig("cofre_vs_welch.png", dpi=150)
plt.show()
print("Saved cofre_vs_welch.png")
