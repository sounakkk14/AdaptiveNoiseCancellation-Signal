"""
Headless VSLMS pipeline shared by the web app (app.py).

Same algorithm as Gui.PY (NLMS-normalised variable step-size LMS with
automatic FFT interference detection), without any Tkinter dependency,
so it can run on a web server.
"""

import numpy as np

from spectral_analysis import compute_fft
from interference_detector import detect_interference
from snr import evaluate_filter
from signal_generator import generate_clean_signal
from ecg_test import (generate_ecg, add_powerline_interference,
                      add_emg_noise, add_baseline_wander)


def vslms_filter(noisy, reference, mu_max=0.3, mu_min=0.001,
                 M=128, alpha=0.999, beta=0.005):
    """
    Variable step-size, NLMS-normalised LMS filter.

    mu(n)    = mu_max / (1 + beta * E[e^2(n)])
    w(n + 1) = w(n) + (mu(n) / ||x(n)||^2) * e(n) * x(n)
    """
    N    = len(noisy)
    w    = np.zeros(M)
    out  = np.zeros(N)
    mu_h = np.zeros(N)
    ep   = np.var(noisy) + 1e-8

    for n in range(M, N):
        x  = reference[n - M:n][::-1]
        e  = noisy[n] - np.dot(w, x)
        ep = alpha * ep + (1 - alpha) * (e ** 2)
        mu = np.clip(mu_max / (1.0 + beta * ep), mu_min, mu_max)
        w += (mu / (np.dot(x, x) + 1e-8)) * e * x
        out[n]  = e
        mu_h[n] = mu

    return out, mu_h


def build_reference(noisy, detected, fs):
    """Synthesise a reference noise signal from the detected frequencies."""
    t   = np.arange(len(noisy)) / fs
    ref = np.zeros(len(noisy))
    for f in detected:
        if 0 < f < fs / 2:
            ref += np.sin(2 * np.pi * f * t)
    if np.max(np.abs(ref)) > 0:
        ref = ref / np.max(np.abs(ref)) * np.std(noisy) * 0.5
    return ref


def _cancel(t, clean, noisy, fs, mu_max, mu_min, threshold, min_freq):
    freqs_fft, spectrum = compute_fft(noisy, fs)
    detected = detect_interference(freqs_fft, spectrum,
                                   threshold=threshold, min_freq=min_freq)
    ref = build_reference(noisy, detected, fs)
    M   = int(np.clip(fs * 0.01, 64, 512))
    cleaned, mu_hist = vslms_filter(noisy, ref, mu_max=mu_max,
                                    mu_min=mu_min, M=M)
    return {
        "t": t, "clean": clean, "noisy": noisy, "cleaned": cleaned,
        "error": cleaned, "mu_hist": mu_hist, "filter_order": M,
        "freqs_fft": freqs_fft, "spectrum": spectrum,
        "spectrum_cleaned": compute_fft(cleaned, fs)[1],
        "detected": detected,
        "metrics": {k: float(v) for k, v in
                    evaluate_filter(clean, noisy, cleaned).items()},
    }


def run_synthetic(fs=1000, duration=10, n_noise=3, amplitude=0.8,
                  mu_max=0.3, mu_min=0.001, threshold=0.15, seed=None):
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
            mu_max=0.3, mu_min=0.001, threshold=0.1, seed=None):
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
