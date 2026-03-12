import numpy as np
import matplotlib.pyplot as plt


def calculate_snr(clean_signal, noisy_signal):
    """
    Calculate SNR between clean and noisy signal.

    SNR (dB) = 10 * log10(signal_power / noise_power)

    Parameters
    ----------
    clean_signal : original clean signal (reference)
    noisy_signal : signal with interference added

    Returns
    -------
    snr_db : SNR in decibels
    """

    signal_power = np.mean(clean_signal ** 2)
    noise = noisy_signal - clean_signal
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return float('inf')

    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db


def calculate_mse(clean_signal, recovered_signal):
    """
    Mean Squared Error between clean and recovered signal.
    Lower = better recovery.
    """
    return np.mean((clean_signal - recovered_signal) ** 2)


def calculate_correlation(clean_signal, recovered_signal):
    """
    Pearson correlation between clean and recovered signal.
    Closer to 1.0 = better recovery.
    """
    corr = np.corrcoef(clean_signal, recovered_signal)[0, 1]
    return corr


def evaluate_filter(clean_signal, noisy_signal, cleaned_signal):
    """
    Full filter quality report.

    Parameters
    ----------
    clean_signal    : original noise-free signal
    noisy_signal    : signal with interference
    cleaned_signal  : signal after filtering

    Returns
    -------
    results : dict with all metrics
    """

    snr_before = calculate_snr(clean_signal, noisy_signal)
    snr_after  = calculate_snr(clean_signal, cleaned_signal)
    snr_improvement = snr_after - snr_before

    mse_before = calculate_mse(clean_signal, noisy_signal)
    mse_after  = calculate_mse(clean_signal, cleaned_signal)

    corr_before = calculate_correlation(clean_signal, noisy_signal)
    corr_after  = calculate_correlation(clean_signal, cleaned_signal)

    results = {
        "SNR Before Filtering (dB)" : round(snr_before, 2),
        "SNR After Filtering (dB)"  : round(snr_after, 2),
        "SNR Improvement (dB)"      : round(snr_improvement, 2),
        "MSE Before Filtering"      : round(mse_before, 6),
        "MSE After Filtering"       : round(mse_after, 6),
        "Correlation Before"        : round(corr_before, 4),
        "Correlation After"         : round(corr_after, 4),
    }

    return results


def print_snr_report(results):
    """
    Print a formatted SNR quality report to console.
    """

    print("\n" + "="*45)
    print("       FILTER QUALITY REPORT")
    print("="*45)

    for metric, value in results.items():
        print(f"  {metric:<30} : {value}")

    print("="*45)

    # Interpretation
    snr_improvement = results["SNR Improvement (dB)"]
    corr_after      = results["Correlation After"]

    print("\n  Interpretation:")
    if snr_improvement > 10:
        print("  ✔ Excellent filtering — strong SNR improvement")
    elif snr_improvement > 5:
        print("  ✔ Good filtering — noticeable improvement")
    elif snr_improvement > 0:
        print("  ~ Moderate filtering — some improvement")
    else:
        print("  ✘ Filtering degraded the signal — check parameters")

    if corr_after > 0.95:
        print("  ✔ Recovered signal closely matches original")
    elif corr_after > 0.80:
        print("  ~ Partial recovery — some distortion remains")
    else:
        print("  ✘ Poor recovery — signal shape changed significantly")

    print("="*45 + "\n")


def plot_snr_comparison(results):
    """
    Bar chart comparing SNR and correlation before vs after filtering.
    """

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Filter Quality Metrics", fontsize=14, fontweight='bold')

    # SNR comparison
    axes[0].bar(["Before", "After"],
                [results["SNR Before Filtering (dB)"],
                 results["SNR After Filtering (dB)"]],
                color=["#e74c3c", "#2ecc71"], width=0.4)
    axes[0].set_title("SNR (dB)")
    axes[0].set_ylabel("dB")

    # MSE comparison
    axes[1].bar(["Before", "After"],
                [results["MSE Before Filtering"],
                 results["MSE After Filtering"]],
                color=["#e74c3c", "#2ecc71"], width=0.4)
    axes[1].set_title("Mean Squared Error")
    axes[1].set_ylabel("MSE")

    # Correlation comparison
    axes[2].bar(["Before", "After"],
                [results["Correlation Before"],
                 results["Correlation After"]],
                color=["#e74c3c", "#2ecc71"], width=0.4)
    axes[2].set_title("Correlation with Clean Signal")
    axes[2].set_ylabel("Correlation")
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()