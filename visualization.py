import matplotlib.pyplot as plt
def plot_signals(t, clean_signal, noisy_signal):

    plt.figure(figsize=(10,6))

    plt.subplot(2,1,1)
    plt.title("Clean Signal")
    plt.plot(t, clean_signal)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.subplot(2,1,2)
    plt.title("Noisy Signal")
    plt.plot(t, noisy_signal)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()


def plot_fft(freqs, spectrum):

    plt.figure(figsize=(8,4))

    plt.plot(freqs, spectrum)

    plt.title("FFT Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    plt.show()


def plot_recovered_signal(t, clean_signal, noisy_signal, cleaned_signal):

    plt.figure(figsize=(10,7))

    plt.subplot(3,1,1)
    plt.title("Original Clean Signal")
    plt.plot(t, clean_signal)

    plt.subplot(3,1,2)
    plt.title("Noisy Signal")
    plt.plot(t, noisy_signal)

    plt.subplot(3,1,3)
    plt.title("Recovered Signal After Filtering")
    plt.plot(t, cleaned_signal)

    plt.tight_layout()
    plt.show()