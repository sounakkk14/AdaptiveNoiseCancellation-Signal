import numpy as np


def vslms_filter(noisy_signal, reference_signal, mu_max=0.1, mu_min=0.001,
                 filter_order=32, alpha=0.9, beta=0.5):
    """
    Variable Step-Size LMS (VSLMS) Adaptive Filter.

    Rectification over base paper (fixed step-size LMS):
    - Step size mu adapts dynamically based on error magnitude
    - Faster convergence when error is large
    - Better stability when error is small (near convergence)

    Parameters
    ----------
    noisy_signal      : observed signal (signal + interference)
    reference_signal  : reference noise signal (used to adapt filter)
    mu_max            : maximum step size (controls convergence speed)
    mu_min            : minimum step size (controls stability at convergence)
    filter_order      : number of LMS filter taps (higher = better freq resolution)
    alpha             : smoothing factor for power estimate (0 < alpha < 1)
    beta              : step-size scaling sensitivity (0 < beta < 1)

    Returns
    -------
    cleaned_signal    : noise-cancelled output signal
    error_signal      : residual error over time (useful for SNR analysis)
    mu_history        : step-size variation over time (shows adaptation)
    """

    N = len(noisy_signal)
    M = filter_order

    # Initialize filter weights
    weights = np.zeros(M)

    # Outputs
    cleaned_signal = np.zeros(N)
    error_signal   = np.zeros(N)
    mu_history     = np.zeros(N)

    # Power estimate for step-size adaptation
    error_power = 1e-6  # small initial value to avoid division by zero

    for n in range(M, N):

        # Reference input window (past M samples)
        x = reference_signal[n - M:n][::-1]

        # Filter output (estimated interference)
        y = np.dot(weights, x)

        # Error = noisy signal - estimated interference
        e = noisy_signal[n] - y

        # Update error power estimate (exponential moving average)
        error_power = alpha * error_power + (1 - alpha) * (e ** 2)

        # Variable step size: large error → large mu, small error → small mu
        mu = mu_min + (mu_max - mu_min) * np.exp(-beta * error_power)
        mu = np.clip(mu, mu_min, mu_max)

        # LMS weight update: w = w + mu * e * x
        weights = weights + mu * e * x

        # Store outputs
        cleaned_signal[n] = e
        error_signal[n]   = e
        mu_history[n]     = mu

    return cleaned_signal, error_signal, mu_history


def generate_reference(noisy_signal, detected_freqs, fs):
    """
    Synthesize a reference interference signal from detected frequencies.

    In real systems this would come from a physical reference sensor.
    Here we reconstruct it from the automatically detected interference
    frequencies — fulfilling the 'automatic detection' rectification.

    Parameters
    ----------
    noisy_signal    : observed noisy signal
    detected_freqs  : interference frequencies from interference_detector
    fs              : sampling frequency (Hz)

    Returns
    -------
    reference : synthesized reference noise signal
    """

    N = len(noisy_signal)
    t = np.arange(N) / fs
    reference = np.zeros(N)

    for f in detected_freqs:
        if f > 0:
            reference += np.sin(2 * np.pi * f * t)

    # Normalize reference
    if np.max(np.abs(reference)) > 0:
        reference = reference / np.max(np.abs(reference))

    return reference


def remove_interference(noisy_signal, freqs_fft, fs, interference_freqs,
                        mu_max=0.05, mu_min=0.001, filter_order=64):
    """
    Main entry point — compatible with existing main.py call signature.

    Builds a synthetic reference from detected interference frequencies
    then applies VSLMS adaptive filtering to cancel noise.

    Parameters
    ----------
    noisy_signal        : observed noisy signal
    freqs_fft           : FFT frequency axis (from spectral_analysis)
    fs                  : sampling frequency
    interference_freqs  : detected interference frequencies (Hz)
    mu_max              : max VSLMS step size
    mu_min              : min VSLMS step size
    filter_order        : number of LMS filter taps

    Returns
    -------
    cleaned_signal : noise-cancelled signal (real-valued)
    """

    reference = generate_reference(noisy_signal, interference_freqs, fs)

    cleaned_signal, error_signal, mu_history = vslms_filter(
        noisy_signal,
        reference,
        mu_max=mu_max,
        mu_min=mu_min,
        filter_order=filter_order
    )

    return np.real(cleaned_signal)


def remove_interference_full(noisy_signal, freqs_fft, fs, interference_freqs,
                             mu_max=0.05, mu_min=0.001, filter_order=64):
    """
    Same as remove_interference but also returns convergence data
    for use with plot_convergence().
    """

    reference = generate_reference(noisy_signal, interference_freqs, fs)

    cleaned_signal, error_signal, mu_history = vslms_filter(
        noisy_signal,
        reference,
        mu_max=mu_max,
        mu_min=mu_min,
        filter_order=filter_order
    )

    return np.real(cleaned_signal), error_signal, mu_history


def plot_convergence(mu_history, error_signal):
    """
    Plot step-size adaptation and error convergence curves.
    Visually demonstrates VSLMS advantage over fixed-step LMS.
    """

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle("VSLMS Adaptive Filter — Convergence Analysis", fontsize=13,
                 fontweight='bold')

    axes[0].plot(mu_history, color='steelblue', linewidth=0.8)
    axes[0].set_title("Variable Step Size μ Over Time")
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("Step Size (μ)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(error_signal ** 2, color='tomato', linewidth=0.6)
    axes[1].set_title("Squared Error — Convergence Curve")
    axes[1].set_xlabel("Sample")
    axes[1].set_ylabel("e²(n)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()