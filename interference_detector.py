import numpy as np
from scipy.signal import find_peaks


def detect_interference(freqs, spectrum, threshold=0.15, min_freq=35):
    """
    Detect interference frequencies using spectral peak detection.

    Parameters
    ----------
    freqs     : frequency axis from FFT
    spectrum  : magnitude spectrum
    threshold : normalized peak height threshold (0–1). Lower = more sensitive.
    min_freq  : ignore peaks below this Hz (protects the base signal at 5/15/30 Hz)

    Returns
    -------
    detected_freqs : array of detected interference frequencies (Hz)
    """

    # Normalize spectrum
    norm_spectrum = spectrum / np.max(spectrum)

    # Detect peaks with minimum distance between them
    peaks, properties = find_peaks(
        norm_spectrum,
        height=threshold,
        distance=5          # at least 5 bins apart
    )

    # Extract frequencies at peaks
    detected_freqs = freqs[peaks]

    # Filter out the base signal frequencies (below min_freq Hz)
    detected_freqs = detected_freqs[detected_freqs >= min_freq]

    return detected_freqs