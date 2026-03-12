import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram


def compute_fft(signal, fs):

    N = len(signal)

    yf = fft(signal)

    xf = fftfreq(N, 1/fs)

    magnitude = np.abs(yf[:N//2])

    frequencies = xf[:N//2]

    return frequencies, magnitude


def compute_spectrogram(signal, fs):

    f, t, Sxx = spectrogram(signal, fs)

    return f, t, Sxx