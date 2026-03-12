# cofre-spectrum

**COFRE** — Complex-Pole Filter Representation for spectral Estimation.

A Python implementation of the algorithm described in:

> Pinto Orellana, M. A., Mirtaheri, P., & Hammer, H. L. (2021). *The Complex-Pole Filter Representation (COFRE) for spectral modeling of fNIRS signals*. arXiv:2105.13476

The Complex-Pole Filter Representation (COFRE) method estimates the power spectral density (PSD) at arbitrary target frequencies using a bank of first-order complex-pole IIR filters each tuned to a single frequency. It is particularly effective for non-stationary or short signals where classical FFT-based methods does not offer sufficent frequency resolution.

## Example — real fNIRS data

PSD estimated on a real fNIRS HbO recording. Left column uses a **band-proportional x-axis** (ENMRC: Endothelial, Neurogenic, Myogenic, Respiratory, Cardiac frequency bands). Right column uses a standard log scale.

![COFRE vs Welch PSD on fNIRS data](https://raw.githubusercontent.com/adamaske/cofre_spectrum/main/cofre_vs_welch_fnirs.png)

Welch estimate computed with [`scipy.signal.welch`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html) (Hann window, 50 % overlap). COFRE resolves low-frequency oscillatory structure that Welch smears out due to limited segment length.

## Installation

```bash
pip install cofre-spectrum
```

## Quick start

```python
from cofre_spectrum import cofre_estimate
import numpy as np

fs = 20.0                        # sampling rate (Hz)
signal = np.random.randn(10_000)

freqs, psd = cofre_estimate(signal, fs=fs)
```

## Full API

### One-liner

```python
freqs, psd = cofre_estimate(
    x,
    fs=20.0,           # sampling rate (Hz)
    freq_min_hz=0.003, # lowest frequency of interest
    freq_max_hz=2.0,   # highest frequency of interest
    n_filters=200,     # number of log-spaced filters
    tau=8.65,          # bandwidth parameter
)
```

### Filter bank with full control

```python
from cofre_spectrum import COFREBank, COFREConfig

cfg = COFREConfig(
    fs=20.0,
    freq_min_hz=0.003,
    freq_max_hz=2.0,
    n_filters=200,
    tau=8.65,          # set τ directly …
    # delta_omega_hz=0.05  # … or derive τ from desired Hz resolution
)

bank = COFREBank(cfg)
bank.process_vectorized(signal)   # batch mode (faster)
freqs, psd = bank.get_spectrum()  # Hz, PSD

bank.summary()  # print configuration table
```

### Single filter (online / streaming)

```python
from cofre_spectrum import COFREFilter

filt = COFREFilter(freq_hz=0.1, fs=20.0, tau=8.65)
for sample in stream:
    filt.update(sample)
print(filt.spectrum_estimate)   # PSD at 0.1 Hz
```

### Choosing τ from a desired resolution

```python
from cofre_spectrum import optimal_tau_for_frequency

# τ that gives 0.05 Hz resolution at α = 0.5
tau = optimal_tau_for_frequency(delta_omega=0.05 / 20.0, alpha=0.5)
```

## Parameter guide

| Parameter | Effect |
|-----------|--------|
| `tau` (τ) | Higher τ → narrower bandwidth, finer frequency resolution, longer rise time |
| `n_filters` | More filters → denser spectral grid |
| `log_spacing` | Log-spaced frequencies (default) suit logarithmic spectral analysis |
| `alpha` (α) | Cut-off fraction for frequency resolution definition (Eq. 18) |
| `beta` (β) | Cut-off fraction for rise-time definition (Eq. 20) |

## Reference

Pinto Orellana, M. A., Mirtaheri, P., & Hammer, H. L. (2021).
*The Complex-Pole Filter Representation (COFRE) for spectral modeling of fNIRS signals*.
[arXiv:2105.13476](https://arxiv.org/abs/2105.13476)

If you use COFRE in your work, please cite the original paper above.
