import numpy as np


def default_filter_order(fs):
    """Filter taps scaled with sampling rate (about 10 ms of reference)."""
    return int(np.clip(fs * 0.01, 64, 512))


def vslms_filter(noisy_signal, reference_signal, mu_max=0.3, mu_min=0.005,
                 filter_order=64, fs=1000, alpha=None, C=1e-2):
    """
    Variable Step-Size NLMS adaptive filter
    (Shin, Sayed & Song, IEEE Signal Processing Letters, 2004).

    Rectification over base paper (fixed step-size LMS):
    - The step size is driven by the part of the error that is still
      correlated with the reference, i.e. the residual interference.
    - Large residual interference -> mu near mu_max -> fast convergence.
    - Interference cancelled      -> mu falls to mu_min -> low misadjustment.
    - The wanted signal is uncorrelated with the reference, so it does not
      hold the step size up (a plain E[e^2] rule cannot tell them apart,
      because in noise cancellation the error *is* the cleaned signal).

    Update equations
    ----------------
    p(n)     = alpha * p(n-1) + (1 - alpha) * x(n) e(n) / ||x(n)||^2
    p^(n)    = p(n) / (1 - alpha^k)        (bias correction, k = samples so far)
    mu(n)    = mu_max * ||p^(n)||^2 / (||p^(n)||^2 + C),  floored at mu_min

    The bias correction (as in the Adam optimiser) stops p(n) reading as
    near zero while the average is still filling, so mu starts near mu_max.
    w(n + 1) = w(n) + mu(n) * e(n) * x(n) / ||x(n)||^2

    Parameters
    ----------
    noisy_signal      : observed signal (signal + interference)
    reference_signal  : reference noise signal (used to adapt filter)
    mu_max            : maximum step size (controls convergence speed)
    mu_min            : minimum step size (keeps the filter able to track)
    filter_order      : number of filter taps
    fs                : sampling frequency (Hz), sets the default alpha
    alpha             : smoothing factor for p(n); default 1 - 1/fs,
                        i.e. p(n) averages over about 1 second
    C                 : positive constant setting how quickly mu falls

    Returns
    -------
    cleaned_signal    : noise-cancelled output signal
    error_signal      : filter error e(n) (equal to the cleaned signal)
    mu_history        : step-size variation over time (shows adaptation)
    """

    N = len(noisy_signal)
    M = filter_order
    if alpha is None:
        alpha = 1 - 1 / fs

    weights = np.zeros(M)
    p       = np.zeros(M)
    alpha_k = 1.0

    cleaned_signal = np.zeros(N)
    mu_history     = np.zeros(N)

    for n in range(M, N):

        # Reference input window (past M samples)
        x = reference_signal[n - M:n][::-1]
        x_power = np.dot(x, x) + 1e-8

        # Error = noisy signal - estimated interference
        e = noisy_signal[n] - np.dot(weights, x)

        # Smoothed error-reference correlation -> variable step size
        p  = alpha * p + (1 - alpha) * x * e / x_power
        alpha_k *= alpha
        p_hat = p / (1 - alpha_k)
        pp = np.dot(p_hat, p_hat)
        mu = max(mu_min, mu_max * pp / (pp + C))

        # Normalised LMS weight update
        weights += (mu / x_power) * e * x

        cleaned_signal[n] = e
        mu_history[n]     = mu

    return cleaned_signal, cleaned_signal.copy(), mu_history


def nlms_filter(noisy_signal, reference_signal, mu=0.3, filter_order=64):
    """
    Fixed step-size NLMS, the baseline VSLMS is compared against.

    Returns the cleaned signal.
    """

    N = len(noisy_signal)
    M = filter_order
    weights = np.zeros(M)
    cleaned_signal = np.zeros(N)

    for n in range(M, N):
        x = reference_signal[n - M:n][::-1]
        e = noisy_signal[n] - np.dot(weights, x)
        weights += (mu / (np.dot(x, x) + 1e-8)) * e * x
        cleaned_signal[n] = e

    return cleaned_signal


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
        if 0 < f < fs / 2:
            reference += np.sin(2 * np.pi * f * t)

    # Scale reference relative to the observed signal
    if np.max(np.abs(reference)) > 0:
        reference = (reference / np.max(np.abs(reference))
                     * np.std(noisy_signal) * 0.5)

    return reference


def remove_interference(noisy_signal, freqs_fft, fs, interference_freqs,
                        mu_max=0.3, mu_min=0.005, filter_order=None):
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
    filter_order        : number of filter taps (default: scaled with fs)

    Returns
    -------
    cleaned_signal : noise-cancelled signal (real-valued)
    """

    cleaned_signal, _, _ = remove_interference_full(
        noisy_signal, freqs_fft, fs, interference_freqs,
        mu_max=mu_max, mu_min=mu_min, filter_order=filter_order)

    return cleaned_signal


def remove_interference_full(noisy_signal, freqs_fft, fs, interference_freqs,
                             mu_max=0.3, mu_min=0.005, filter_order=None):
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
        filter_order=filter_order or default_filter_order(fs),
        fs=fs
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
    axes[1].set_title("Squared Filter Output e²(n)")
    axes[1].set_xlabel("Sample")
    axes[1].set_ylabel("e²(n)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
