"""
COFRE: Complex-Pole Filter Representation for Spectral Estimation
==================================================================

Implementation following Pinto Orellana, Mirtaheri & Hammer (2021).
arXiv:2105.13476v1

Core idea
---------
Estimate the power spectral density at specific target frequencies using
a bank of single-complex-pole IIR filters. Each filter is a first-order
autoregressive process with a complex coefficient φ = ρ·exp(j·2π·ω*),
where ω* is the normalized target frequency and ρ controls the bandwidth.

Key equations referenced (paper numbering):
  Eq. 10  — Filter recursion:      y(t) = φ·y(t-1) + x(t)
  Eq. 13  — Magnitude response:    M(ω) = |1 - ρ·exp(j·2π·(ω*-ω))|^{-1}
  Eq. 18  — Frequency resolution:  Δω = f(ρ, α)
  Eq. 19  — Frequency-optimal τ:   τ = g(Δω, α)
  Eq. 20  — Rise time:             t* = log(1-β) / log(ρ) - 1
  Eq. 29  — Spectrum estimator:    Ŝ(ω*) = Var[y(t)] / e^{-2τ}
  Eq. 31  — Recursive estimator:   online update rule

Quick start
-----------
**Batch estimation (one-liner)**::

    from cofre import cofre_estimate
    import numpy as np

    fs = 20.0                          # Hz — sampling rate
    signal = np.random.randn(10_000)   # your time-series data
    freqs, psd = cofre_estimate(signal, fs=fs)

**Full control via COFREBank**::

    from cofre import COFREBank, COFREConfig

    cfg = COFREConfig(
        fs=20.0,            # sampling rate (Hz)
        freq_min_hz=0.003,  # lowest frequency of interest (Hz)
        freq_max_hz=2.0,    # highest frequency of interest (Hz)
        n_filters=200,      # number of log-spaced filters
        tau=8.65,           # bandwidth parameter (higher → narrower bandwidth)
    )
    bank = COFREBank(cfg)
    bank.process_vectorized(signal)    # fast batch mode
    freqs, psd = bank.get_spectrum()   # arrays of equal length

    bank.summary()  # prints configuration details

**Single filter (online / streaming)**::

    from cofre import COFREFilter

    filt = COFREFilter(freq_hz=0.1, fs=20.0, tau=8.65)
    for sample in stream:
        filt.update(sample)
    print(filt.spectrum_estimate)   # PSD at 0.1 Hz

**Choosing τ from a desired frequency resolution**::

    from cofre import optimal_tau_for_frequency

    # Achieve 0.05 Hz resolution at α=0.5
    tau = optimal_tau_for_frequency(delta_omega=0.05 / 20.0, alpha=0.5)

Public API
----------
Classes:
    COFREConfig   — dataclass holding all filter-bank hyper-parameters
    COFREBank     — filter bank; main entry point for batch processing
    COFREFilter   — single complex-pole IIR filter (online capable)

Functions:
    cofre_estimate            — convenience one-liner (returns freqs, psd)
    bandwidth_to_rho          — convert τ → pole modulus ρ
    rho_to_bandwidth          — convert ρ → τ
    frequency_resolution      — compute Δω for given τ and α  (Eq. 18)
    optimal_tau_for_frequency — compute τ for a desired Δω    (Eq. 19)
    rise_time                 — compute transient rise time t* (Eq. 20)
    rise_time_seconds         — rise time in seconds
    optimal_tau_for_rise_time — compute τ for a target rise time
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Filter parameter computation (Lemmas 6–9 from the paper)
# ---------------------------------------------------------------------------

def bandwidth_to_rho(tau: float) -> float:
    """Convert bandwidth parameter τ to pole modulus ρ.  (ρ = 1 - e^{-τ})"""
    return 1.0 - np.exp(-tau)


def rho_to_bandwidth(rho: float) -> float:
    """Convert pole modulus ρ to bandwidth parameter τ.  (τ = -log(1 - ρ))"""
    assert 0 < rho < 1, "ρ must be in (0, 1) for stability"
    return -np.log(1.0 - rho)


def frequency_resolution(tau: float, alpha: float = 0.5) -> float:
    """
    Compute the frequency resolution Δω (normalized) for a given bandwidth τ
    and cut-off factor α.  (Lemma 6, Eq. 18)

    Parameters
    ----------
    tau   : bandwidth parameter (τ > 0)
    alpha : cut-off factor in (0, 1) — fraction of peak gain that defines
            the resolution boundary

    Returns
    -------
    delta_omega : frequency resolution (normalized, ω = f/fs)
    """
    rho = bandwidth_to_rho(tau)
    cos_arg = (1.0 / (2.0 * rho)) * (rho**2 + 1.0 - alpha**(-2) * (1.0 - rho)**2)
    # Clamp to [-1, 1] for numerical safety
    cos_arg = np.clip(cos_arg, -1.0, 1.0)
    return np.arccos(cos_arg) / (2.0 * np.pi)


def optimal_tau_for_frequency(delta_omega: float, alpha: float = 0.5) -> float:
    """
    Compute the minimum bandwidth τ that achieves a target frequency
    resolution Δω under cut-off α.  (Corollary 7, Eq. 19)

    Parameters
    ----------
    delta_omega : desired frequency resolution (normalized, ω = f/fs)
    alpha       : cut-off factor in (0, 1)

    Returns
    -------
    tau : optimal bandwidth parameter
    """
    c = np.cos(2.0 * np.pi * delta_omega)
    a_inv2 = alpha**(-2)
    denom = 1.0 - a_inv2

    # Quadratic solution for ρ (Eq. 51, taking the stable root)
    discriminant = c**2 - denom**2
    assert discriminant >= 0, (
        f"No valid τ exists for Δω={delta_omega}, α={alpha} "
        f"(discriminant={discriminant:.6e})"
    )
    rho = (c - a_inv2) / denom - np.sqrt(discriminant) / denom
    assert 0 < rho < 1, f"Computed ρ={rho} is outside (0,1)"
    return rho_to_bandwidth(rho)


def rise_time(tau: float, beta: float = 0.5) -> float:
    """
    Compute the rise (transient settling) time t* in sample points.
    (Lemma 8, Eq. 20)

    Parameters
    ----------
    tau  : bandwidth parameter
    beta : cut-off factor in (0, 1) — fraction of steady-state amplitude

    Returns
    -------
    t_star : rise time in samples
    """
    rho = bandwidth_to_rho(tau)
    return np.log(1.0 - beta) / np.log(rho) - 1.0


def rise_time_seconds(tau: float, fs: float, beta: float = 0.5) -> float:
    """Rise time converted to seconds."""
    return rise_time(tau, beta) / fs


def optimal_tau_for_rise_time(t_star: float, beta: float = 0.5) -> float:
    """
    Compute the minimum bandwidth τ that achieves a target rise time t*.
    (Corollary 9, Eq. 21)

    Parameters
    ----------
    t_star : desired rise time in samples
    beta   : cut-off factor in (0, 1)

    Returns
    -------
    tau : optimal bandwidth parameter
    """
    rho = (1.0 - beta) ** (1.0 / (1.0 + t_star))
    return rho_to_bandwidth(rho)


# ---------------------------------------------------------------------------
# Single COFRE filter
# ---------------------------------------------------------------------------

class COFREFilter:
    """
    A single complex-pole narrow bandpass filter.

    Implements the recursion y(t) = φ·y(t-1) + x(t)  (Eq. 10)
    where φ = ρ·exp(j·2π·ω*).

    The filter tracks a running variance of |y(t)|² to produce
    the spectrum estimate at ω* via Eq. 29.
    """

    def __init__(self, freq_hz: float, fs: float, tau: float):
        """
        Parameters
        ----------
        freq_hz : target frequency in Hz
        fs      : sampling rate in Hz
        tau     : bandwidth parameter (controls resolution vs. rise time)
        """
        self.freq_hz = freq_hz
        self.fs = fs
        self.tau = tau

        # Normalized frequency ω* = f / fs
        self.omega_star = freq_hz / fs

        # Pole modulus and complex coefficient
        self.rho = bandwidth_to_rho(tau)
        self.phi = self.rho * np.exp(1j * 2.0 * np.pi * self.omega_star)

        # Filter state
        self.y = 0.0 + 0.0j    # complex filter output y(t)
        self.sum_y2 = 0.0       # running sum of |y(t)|²
        self.n = 0              # number of samples processed

    def reset(self):
        """Reset filter state."""
        self.y = 0.0 + 0.0j
        self.sum_y2 = 0.0
        self.n = 0

    def update(self, x: float) -> complex:
        """
        Process one new sample x(t).  (Eq. 10 / Eq. 32)

        Returns the complex filter output y(t).
        """
        # y(t) = φ · y(t-1) + x(t)
        self.y = self.phi * self.y + x
        self.n += 1
        self.sum_y2 += np.abs(self.y) ** 2
        return self.y

    def process(self, x: np.ndarray) -> np.ndarray:
        """
        Process an entire signal array.

        Returns array of complex filter outputs.
        """
        T = len(x)
        y_out = np.empty(T, dtype=complex)
        for t in range(T):
            y_out[t] = self.update(x[t])
        return y_out

    @property
    def spectrum_estimate(self) -> float:
        """
        Current PSD estimate at ω*.  (Eq. 29)

        Ŝ(ω*) = Var_biased[y(t)] / e^{-2τ}
               = (1/T · Σ|y(t)|²) / (1-ρ)²
        """
        if self.n == 0:
            return 0.0
        biased_var = self.sum_y2 / self.n
        return biased_var / (1.0 - self.rho) ** 2

    @property
    def freq_resolution_hz(self) -> float:
        """Frequency resolution in Hz (at α=0.5)."""
        return frequency_resolution(self.tau, alpha=0.5) * self.fs

    @property
    def rise_time_samples(self) -> float:
        """Rise time in samples (at β=0.5)."""
        return rise_time(self.tau, beta=0.5)

    @property
    def rise_time_sec(self) -> float:
        """Rise time in seconds (at β=0.5)."""
        return self.rise_time_samples / self.fs


# ---------------------------------------------------------------------------
# COFRE filter bank — the full spectrum estimator
# ---------------------------------------------------------------------------

@dataclass
class COFREConfig:
    """Configuration for a COFRE filter bank."""
    fs: float = 20.0                # sampling rate (Hz)
    freq_min_hz: float = 0.003      # lowest target frequency (Hz)
    freq_max_hz: float = 2.0        # highest target frequency (Hz)
    n_filters: int = 200            # number of filters (log-spaced)
    tau: Optional[float] = None     # if set, use this τ for all filters
    delta_omega_hz: Optional[float] = None  # if set, compute τ from desired resolution
    alpha: float = 0.75             # cut-off for frequency resolution
    beta: float = 0.25              # cut-off for rise time
    log_spacing: bool = True        # log-spaced frequencies (better for ENMRC)


class COFREBank:
    """
    A bank of COFRE filters for spectrum estimation across a frequency range.

    Usage
    -----
    >>> bank = COFREBank(COFREConfig(fs=20.0, n_filters=100))
    >>> bank.process(signal)
    >>> freqs, psd = bank.get_spectrum()
    """

    def __init__(self, cfg: COFREConfig):
        self.cfg = cfg

        # Build the array of target frequencies
        if cfg.log_spacing:
            self.freqs_hz = np.geomspace(cfg.freq_min_hz, cfg.freq_max_hz,
                                         cfg.n_filters)
        else:
            self.freqs_hz = np.linspace(cfg.freq_min_hz, cfg.freq_max_hz,
                                        cfg.n_filters)

        # Determine τ for each filter
        if cfg.tau is not None:
            # Uniform τ across all filters
            taus = np.full(cfg.n_filters, cfg.tau)
        elif cfg.delta_omega_hz is not None:
            # Compute τ from desired frequency resolution
            delta_omega_norm = cfg.delta_omega_hz / cfg.fs
            tau_val = optimal_tau_for_frequency(delta_omega_norm, cfg.alpha)
            taus = np.full(cfg.n_filters, tau_val)
        else:
            # Default: use τ = 8.65 as in the paper's analysis
            taus = np.full(cfg.n_filters, 8.65)

        # Create the filter bank
        self.filters = [
            COFREFilter(freq_hz=f, fs=cfg.fs, tau=t)
            for f, t in zip(self.freqs_hz, taus)
        ]

        self._processed = False

    def reset(self):
        """Reset all filters."""
        for filt in self.filters:
            filt.reset()
        self._processed = False

    def process(self, x: np.ndarray):
        """
        Run the entire signal through all filters.

        This is the batch mode.  For each sample x(t), every filter
        is updated via the O(1) recursion (Eq. 10).
        """
        self.reset()
        T = len(x)
        for t in range(T):
            sample = x[t]
            for filt in self.filters:
                filt.update(sample)
        self._processed = True

    def process_vectorized(self, x: np.ndarray):
        """
        Vectorized batch processing — much faster for large signals.

        Runs each filter over the full signal using a loop over filters
        (not samples), leveraging numpy for the inner sample loop.
        """
        self.reset()
        T = len(x)
        for filt in self.filters:
            # Vectorized recursion for this filter
            y = 0.0 + 0.0j
            sum_y2 = 0.0
            phi = filt.phi
            for t in range(T):
                y = phi * y + x[t]
                sum_y2 += np.abs(y) ** 2
            filt.y = y
            filt.sum_y2 = sum_y2
            filt.n = T
        self._processed = True

    def get_spectrum(self):
        """
        Retrieve the estimated PSD at each filter's target frequency.

        Returns
        -------
        freqs_hz : np.ndarray — target frequencies in Hz
        psd      : np.ndarray — estimated power spectral density
        """
        assert self._processed, "Call .process(x) or .process_vectorized(x) first"
        psd = np.array([f.spectrum_estimate for f in self.filters])
        return self.freqs_hz.copy(), psd

    def get_spectrum_mhz(self):
        """Same as get_spectrum but frequencies in mHz."""
        freqs, psd = self.get_spectrum()
        return freqs * 1e3, psd

    def summary(self):
        """Print a summary of the filter bank configuration."""
        f0 = self.filters[0]
        print("COFRE Filter Bank Summary")
        print("=" * 50)
        print(f"  Sampling rate     : {self.cfg.fs} Hz")
        print(f"  Frequency range   : {self.cfg.freq_min_hz*1e3:.1f} – "
              f"{self.cfg.freq_max_hz*1e3:.1f} mHz")
        print(f"  Number of filters : {self.cfg.n_filters}")
        print(f"  Spacing           : {'log' if self.cfg.log_spacing else 'linear'}")
        print(f"  τ (bandwidth)     : {f0.tau:.4f}")
        print(f"  ρ (pole modulus)  : {f0.rho:.6f}")
        print(f"  Freq. resolution  : {f0.freq_resolution_hz*1e3:.4f} mHz  (α={self.cfg.alpha})")
        print(f"  Rise time         : {f0.rise_time_samples:.1f} samples = "
              f"{f0.rise_time_sec:.2f} s  (β={self.cfg.beta})")


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def cofre_estimate(x: np.ndarray, fs: float = 20.0,
                   freq_min_hz: float = 0.003, freq_max_hz: float = 2.0,
                   n_filters: int = 200, tau: float = 8.65):
    """
    One-liner COFRE spectrum estimation.

    Returns (freqs_hz, psd).
    """
    cfg = COFREConfig(fs=fs, freq_min_hz=freq_min_hz, freq_max_hz=freq_max_hz,
                      n_filters=n_filters, tau=tau)
    bank = COFREBank(cfg)
    bank.process_vectorized(x)
    return bank.get_spectrum()