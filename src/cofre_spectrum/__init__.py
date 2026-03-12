"""
cofre-spectrum — COFRE spectral estimation for Python
======================================================

Install::

    pip install cofre-spectrum

Quick start::

    from cofre_spectrum import cofre_estimate
    freqs, psd = cofre_estimate(signal, fs=20.0)

See ``cofre_spectrum.core`` or the project README for the full API.
"""

from cofre_spectrum.core import (
    # Classes
    COFREConfig,
    COFREBank,
    COFREFilter,
    # Convenience
    cofre_estimate,
    # Parameter helpers
    bandwidth_to_rho,
    rho_to_bandwidth,
    frequency_resolution,
    optimal_tau_for_frequency,
    rise_time,
    rise_time_seconds,
    optimal_tau_for_rise_time,
)

__all__ = [
    "COFREConfig",
    "COFREBank",
    "COFREFilter",
    "cofre_estimate",
    "bandwidth_to_rho",
    "rho_to_bandwidth",
    "frequency_resolution",
    "optimal_tau_for_frequency",
    "rise_time",
    "rise_time_seconds",
    "optimal_tau_for_rise_time",
]

__version__ = "0.1.0"
__author__ = "COFRE Authors"
