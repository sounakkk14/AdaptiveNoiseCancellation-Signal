

import numpy as np


def generate_time_vector(duration=2, fs=1000):
    """
    Create time vector

    Parameters
    ----------
    duration : seconds
    fs : sampling frequency

    Returns
    -------
    t : numpy array
    """

    t = np.arange(0, duration, 1/fs)

    return t


def generate_clean_signal(t):
    """
    Generate a multi-frequency base signal.

    This simulates a realistic signal such as
    biomedical or communication waveform.
    """

    signal = (
        np.sin(2*np.pi*5*t) +
        0.6*np.sin(2*np.pi*15*t) +
        0.3*np.sin(2*np.pi*30*t)
    )

    return signal


def generate_random_interference(t,
                                 num_interference=3,
                                 freq_range=(30,150),
                                 amplitude=0.8):
    """
    Generate random interference tones.

    Parameters
    ----------
    num_interference : number of interfering frequencies
    freq_range : frequency range for interference
    amplitude : noise amplitude

    Returns
    -------
    noise : interference signal
    frequencies : list of frequencies used
    """

    freqs = np.random.randint(freq_range[0],
                              freq_range[1],
                              num_interference)

    noise = np.zeros(len(t))

    for f in freqs:

        noise += amplitude * np.sin(2*np.pi*f*t)

    return noise, freqs


def generate_noisy_signal(signal, noise):
    """
    Combine signal and noise.
    """

    return signal + noise


def create_test_signal(duration=2,
                       fs=1000,
                       num_interference=3):
    """
    Full pipeline function.

    Returns:
        t
        clean_signal
        noisy_signal
        interference_freqs
    """

    t = generate_time_vector(duration, fs)

    clean = generate_clean_signal(t)

    noise, freqs = generate_random_interference(
        t,
        num_interference=num_interference
    )

    noisy = generate_noisy_signal(clean, noise)

    return t, clean, noisy, freqs