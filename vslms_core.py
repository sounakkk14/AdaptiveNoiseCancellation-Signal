"""
Headless VSLMS pipeline shared by the web app (app.py).

Uses the VSS-NLMS filter from adaptive_filter.py with automatic FFT
interference detection, without any Tkinter dependency, so it can run
on a web server. Also runs fixed step-size NLMS at mu_max and mu_min as
the baselines VSLMS is compared against.
"""

import numpy as np

from adaptive_filter import (vslms_filter, nlms_filter, generate_reference,
                             default_filter_order)
from spectral_analysis import compute_fft
from interference_detector import detect_interference
from snr import evaluate_filter
from signal_generator import generate_clean_signal
from ecg_test import (generate_ecg, add_powerline_interference,
                      add_emg_noise, add_baseline_wander)


def _cancel(t, clean, noisy, fs, mu_max, mu_min, threshold, min_freq):
    freqs_fft, spectrum = compute_fft(noisy, fs)
    detected = detect_interference(freqs_fft, spectrum,
                                   threshold=threshold, min_freq=min_freq)
    ref = generate_reference(noisy, detected, fs)
    M   = default_filter_order(fs)
    cleaned, _, mu_hist = vslms_filter(noisy, ref, mu_max=mu_max,
                                       mu_min=mu_min, filter_order=M, fs=fs)
    fixed = {mu: nlms_filter(noisy, ref, mu=mu, filter_order=M)
             for mu in (mu_max, mu_min)}
    return {
        "t": t, "clean": clean, "noisy": noisy, "cleaned": cleaned,
        "mu_hist": mu_hist, "filter_order": M, "fixed": fixed,
        "freqs_fft": freqs_fft, "spectrum": spectrum,
        "spectrum_cleaned": compute_fft(cleaned, fs)[1],
        "detected": detected,
        "metrics": {k: float(v) for k, v in
                    evaluate_filter(clean, noisy, cleaned).items()},
        "fixed_snr": {mu: float(evaluate_filter(clean, noisy, out)
                                ["SNR Improvement (dB)"])
                      for mu, out in fixed.items()},
    }


def run_synthetic(fs=1000, duration=10, n_noise=3, amplitude=0.8,
                  mu_max=0.3, mu_min=0.005, threshold=0.15, seed=None):
    rng   = np.random.default_rng(seed)
    t     = np.arange(0, duration, 1 / fs)
    clean = generate_clean_signal(t)
    actual = np.sort(rng.integers(35, 150, n_noise))
    noise = sum(amplitude * np.sin(2 * np.pi * f * t) for f in actual)
    r = _cancel(t, clean, clean + noise, fs, mu_max, mu_min,
                threshold, min_freq=35)
    r["actual"] = actual
    return r


def run_ecg(fs=1000, duration=10, heart_rate=72, powerline=50.0,
            noise_amplitude=0.3, add_emg=True, add_wander=True,
            mu_max=0.3, mu_min=0.005, threshold=0.1, seed=None):
    if seed is not None:
        np.random.seed(seed)  # add_emg_noise uses the global RNG
    t, clean = generate_ecg(duration=duration, fs=fs, heart_rate=heart_rate)
    noisy, actual = add_powerline_interference(
        clean, fs, frequency=powerline, amplitude=noise_amplitude)
    if add_emg:
        noisy = add_emg_noise(noisy)
    if add_wander:
        noisy = add_baseline_wander(noisy, fs)
    r = _cancel(t, clean, noisy, fs, mu_max, mu_min,
                threshold, min_freq=45)
    r["actual"] = actual
    return r
