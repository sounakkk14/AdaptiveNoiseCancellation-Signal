# === All imports at the top ===
import matplotlib.pyplot as plt
from visualization import plot_signals, plot_fft, plot_recovered_signal
from adaptive_filter import remove_interference_full, plot_convergence
from interference_detector import detect_interference
from signal_generator import create_test_signal
from spectral_analysis import compute_fft
from snr import evaluate_filter, print_snr_report, plot_snr_comparison
from ecg_test import run_ecg_test, plot_ecg_results, plot_ecg_spectrum

print("=== Intelligent Noise Detection System (VSLMS) ===")
print("\nSelect Test Mode:")
print("  1. Synthetic signal test")
print("  2. Real-world ECG test")

mode = input("\nEnter mode (1 or 2): ").strip()

# ──────────────────────────────
# MODE 1: Synthetic Signal Test
# ──────────────────────────────
if mode == "1":

    fs        = int(input("Enter sampling frequency (Hz): "))
    duration  = float(input("Enter signal duration (seconds): "))
    num_noise = int(input("Enter number of interference frequencies: "))

    t, clean_signal, noisy_signal, freqs = create_test_signal(
        fs=fs,
        duration=duration,
        num_interference=num_noise
    )
    print("Actual interference frequencies:", freqs)

    plot_signals(t, clean_signal, noisy_signal)

    freqs_fft, spectrum = compute_fft(noisy_signal, fs)
    plot_fft(freqs_fft, spectrum)

    detected = detect_interference(freqs_fft, spectrum)
    print("\nDetected interference frequencies:")
    for f in detected:
        print(round(f, 2), "Hz")

    cleaned_signal, error_signal, mu_history = remove_interference_full(
        noisy_signal, freqs_fft, fs, detected
    )

    plot_recovered_signal(t, clean_signal, noisy_signal, cleaned_signal)
    plot_convergence(mu_history, error_signal)

    results = evaluate_filter(clean_signal, noisy_signal, cleaned_signal)
    print_snr_report(results)
    plot_snr_comparison(results)

# ──────────────────────────────
# MODE 2: ECG Real-World Test
# ──────────────────────────────
elif mode == "2":

    print("\nECG Test Configuration:")
    fs_input         = input("Sampling frequency Hz [press Enter for 1000]: ").strip()
    duration_input   = input("Duration seconds [press Enter for 10]: ").strip()
    hr_input         = input("Heart rate BPM [press Enter for 72]: ").strip()
    pl_input         = input("Power-line frequency Hz, 50=India 60=US [press Enter for 50]: ").strip()

    fs         = int(fs_input)       if fs_input       else 1000
    duration   = float(duration_input) if duration_input else 10.0
    heart_rate = int(hr_input)       if hr_input       else 72
    pl_freq    = float(pl_input)     if pl_input       else 50.0

    # run_ecg_test returns exactly 7 values
    t, clean_ecg, noisy_ecg, cleaned_ecg, results, mu_history, error_signal = run_ecg_test(
        fs=fs,
        duration=duration,
        heart_rate=heart_rate,
        powerline_freq=pl_freq
    )

    plot_ecg_results(t, clean_ecg, noisy_ecg, cleaned_ecg, fs=fs)
    plot_ecg_spectrum(t, clean_ecg, noisy_ecg, cleaned_ecg, fs=fs)
    plot_convergence(mu_history, error_signal)
    plot_snr_comparison(results)

else:
    print("Invalid mode. Please enter 1 or 2.")