# === All imports at the top ===
import matplotlib.pyplot as plt
from visualization import plot_signals, plot_fft, plot_recovered_signal
from adaptive_filter import remove_interference_full, plot_convergence
from interference_detector import detect_interference
from signal_generator import create_test_signal
from spectral_analysis import compute_fft
from snr import evaluate_filter, print_snr_report, plot_snr_comparison

print("=== Intelligent Noise Detection System (VSLMS) ===")

fs       = int(input("Enter sampling frequency (Hz): "))
duration = float(input("Enter signal duration (seconds): "))
num_noise = int(input("Enter number of interference frequencies: "))

# Generate signals
t, clean_signal, noisy_signal, freqs = create_test_signal(
    fs=fs,
    duration=duration,
    num_interference=num_noise
)
print("Actual interference frequencies:", freqs)

# Plot clean vs noisy
plot_signals(t, clean_signal, noisy_signal)

# Compute FFT
freqs_fft, spectrum = compute_fft(noisy_signal, fs)

# Plot FFT spectrum
plot_fft(freqs_fft, spectrum)

# Detect interference automatically
detected = detect_interference(freqs_fft, spectrum)

print("\nDetected interference frequencies:")
for f in detected:
    print(round(f, 2), "Hz")

# Apply VSLMS adaptive filter
cleaned_signal, error_signal, mu_history = remove_interference_full(
    noisy_signal,
    freqs_fft,
    fs,
    detected
)

# Plot recovered signal
plot_recovered_signal(t, clean_signal, noisy_signal, cleaned_signal)

# Plot VSLMS convergence (step-size + error curve)
plot_convergence(mu_history, error_signal)

# SNR Quality Report
results = evaluate_filter(clean_signal, noisy_signal, cleaned_signal)
print_snr_report(results)
plot_snr_comparison(results)