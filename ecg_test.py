import numpy as np
import matplotlib.pyplot as plt


def _gaussian(t, mu, sigma):
    return np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))


def generate_ecg(duration=10, fs=1000, heart_rate=72):
    """
    Generate a realistic synthetic ECG signal with PQRST morphology.
    """
    t = np.arange(0, duration, 1 / fs)
    ecg = np.zeros(len(t))

    beat_period = int(fs * 60 / heart_rate)

    components = {
        'P': ( 0.15, -0.20, 0.025),
        'Q': (-0.10, -0.05, 0.010),
        'R': ( 1.00,  0.00, 0.008),
        'S': (-0.25,  0.05, 0.010),
        'T': ( 0.30,  0.18, 0.040),
    }

    beat_centers = np.arange(beat_period // 2, len(t), beat_period)

    for center in beat_centers:
        for wave, (amp, offset, width) in components.items():
            peak = center + int(offset * fs)
            if 0 < peak < len(t):
                t_norm = t - t[peak]
                ecg += amp * _gaussian(t_norm, 0, width)

    return t, ecg


def add_powerline_interference(ecg, fs, frequency=50.0, amplitude=0.3):
    """Add 50/60 Hz power-line interference + harmonic."""
    t = np.arange(len(ecg)) / fs
    noise  = amplitude * np.sin(2 * np.pi * frequency * t)
    noise += (amplitude * 0.4) * np.sin(2 * np.pi * 2 * frequency * t)
    noisy_ecg = ecg + noise
    interference_freqs = np.array([frequency, 2 * frequency])
    return noisy_ecg, interference_freqs


def add_emg_noise(ecg, amplitude=0.08):
    """Add broadband EMG muscle artifact."""
    return ecg + amplitude * np.random.randn(len(ecg))


def add_baseline_wander(ecg, fs, frequency=0.3, amplitude=0.15):
    """Add low-frequency baseline wander from breathing."""
    t = np.arange(len(ecg)) / fs
    return ecg + amplitude * np.sin(2 * np.pi * frequency * t)


def run_ecg_test(fs=1000, duration=10, heart_rate=72,
                 powerline_freq=50.0, noise_amplitude=0.3,
                 add_emg=True, add_wander=True):
    """
    Full ECG noise cancellation pipeline.

    Returns
    -------
    t, clean_ecg, noisy_ecg, cleaned_ecg, results, mu_history, error_signal
    """

    from adaptive_filter import remove_interference_full
    from interference_detector import detect_interference
    from spectral_analysis import compute_fft
    from snr import evaluate_filter, print_snr_report

    print("\n" + "="*50)
    print("   ECG REAL-WORLD NOISE CANCELLATION TEST")
    print("="*50)
    print(f"  Sampling Frequency : {fs} Hz")
    print(f"  Duration           : {duration} s")
    print(f"  Heart Rate         : {heart_rate} BPM")
    print(f"  Power-line noise   : {powerline_freq} Hz")
    print("="*50)

    # Step 1: Generate clean ECG
    t, clean_ecg = generate_ecg(duration=duration, fs=fs,
                                 heart_rate=heart_rate)

    # Step 2: Add noise layers
    noisy_ecg, interference_freqs = add_powerline_interference(
        clean_ecg, fs,
        frequency=powerline_freq,
        amplitude=noise_amplitude
    )

    if add_emg:
        noisy_ecg = add_emg_noise(noisy_ecg)
        print("  + EMG muscle noise added")

    if add_wander:
        noisy_ecg = add_baseline_wander(noisy_ecg, fs)
        print("  + Baseline wander added")

    print(f"\n  Known interference : {interference_freqs} Hz")

    # Step 3: Auto-detect interference via FFT
    freqs_fft, spectrum = compute_fft(noisy_ecg, fs)
    detected = detect_interference(freqs_fft, spectrum,
                                    threshold=0.1, min_freq=45)
    print(f"  Detected           : {np.round(detected, 1)} Hz")

    # Step 4: VSLMS filtering — returns 3 values
    cleaned_ecg, error_signal, mu_history = remove_interference_full(
        noisy_ecg, freqs_fft, fs, detected
    )
    cleaned_ecg = np.real(cleaned_ecg)

    # Step 5: Evaluate quality
    results = evaluate_filter(clean_ecg, noisy_ecg, cleaned_ecg)
    print_snr_report(results)

    # Explicitly return all 7 values
    return t, clean_ecg, noisy_ecg, cleaned_ecg, results, mu_history, error_signal


def plot_ecg_results(t, clean_ecg, noisy_ecg, cleaned_ecg,
                     fs=1000, zoom_seconds=3):
    """4-panel ECG comparison with zoomed heartbeat view."""

    zoom_samples = zoom_seconds * fs

    fig, axes = plt.subplots(4, 1, figsize=(13, 10))
    fig.suptitle("ECG Real-World Noise Cancellation — VSLMS",
                 fontsize=14, fontweight='bold')

    axes[0].plot(t, clean_ecg, color='#2ecc71', linewidth=0.8)
    axes[0].set_title("Clean ECG (Reference)")
    axes[0].set_ylabel("Amplitude (mV)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, noisy_ecg, color='#e74c3c', linewidth=0.6)
    axes[1].set_title("Noisy ECG (Power-line + EMG + Baseline Wander)")
    axes[1].set_ylabel("Amplitude (mV)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, cleaned_ecg, color='#3498db', linewidth=0.8)
    axes[2].set_title("Recovered ECG After VSLMS Filtering")
    axes[2].set_ylabel("Amplitude (mV)")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t[:zoom_samples], clean_ecg[:zoom_samples],
                 color='#2ecc71', linewidth=1.2, label='Clean', alpha=0.9)
    axes[3].plot(t[:zoom_samples], cleaned_ecg[:zoom_samples],
                 color='#3498db', linewidth=1.0, label='Recovered',
                 linestyle='--', alpha=0.9)
    axes[3].set_title(f"Zoomed — First {zoom_seconds}s (Clean vs Recovered)")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Amplitude (mV)")
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_ecg_spectrum(t, clean_ecg, noisy_ecg, cleaned_ecg, fs=1000):
    """3-panel FFT spectrum comparison showing spike removal."""

    from spectral_analysis import compute_fft

    freqs_c, spec_c = compute_fft(clean_ecg,   fs)
    freqs_n, spec_n = compute_fft(noisy_ecg,   fs)
    freqs_r, spec_r = compute_fft(cleaned_ecg, fs)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8))
    fig.suptitle("ECG Frequency Spectrum Comparison",
                 fontsize=13, fontweight='bold')

    lim = min(300, fs // 2)

    axes[0].plot(freqs_c, spec_c, color='#2ecc71', linewidth=0.8)
    axes[0].set_title("Clean ECG Spectrum")
    axes[0].set_ylabel("Magnitude")
    axes[0].set_xlim(0, lim)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(freqs_n, spec_n, color='#e74c3c', linewidth=0.8)
    axes[1].set_title("Noisy ECG Spectrum (power-line spike visible)")
    axes[1].set_ylabel("Magnitude")
    axes[1].set_xlim(0, lim)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(freqs_r, spec_r, color='#3498db', linewidth=0.8)
    axes[2].set_title("Recovered ECG Spectrum (spike removed)")
    axes[2].set_ylabel("Magnitude")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_xlim(0, lim)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()