# 🔊 Intelligent Noise Detection System
### Adaptive LMS-Based Noise Cancellation for Biomedical Signals

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![NumPy](https://img.shields.io/badge/NumPy-scientific-orange?logo=numpy)
![SciPy](https://img.shields.io/badge/SciPy-signal%20processing-green?logo=scipy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-visualization-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

This project implements an **Intelligent Noise Detection and Cancellation System** using a **Variable Step-Size LMS (VSLMS) adaptive filter** combined with **FFT-based automatic interference detection**.

It is based on the research paper:
> *"Adaptive LMS-Based Noise Cancellation for Biomedical Signals"*

### Rectifications over the base paper:
| Base Paper Limitation | This Project's Improvement |
|---|---|
| Fixed step-size LMS | **Variable step-size** — μ adapts dynamically based on error power |
| Assumed/known interference frequency | **Automatic FFT peak detection** — no prior knowledge needed |
| Limited convergence speed | **NLMS-style normalization** — stable and fast at any sampling rate |

---

## 🗂️ Project Structure

```
MINIPROJECT/
│
├── main.py                  # Entry point — Mode 1 (synthetic) or Mode 2 (ECG)
├── signal_generator.py      # Generate synthetic multi-frequency test signals
├── spectral_analysis.py     # FFT and spectrogram computation
├── interference_detector.py # Auto-detect interference via FFT peak detection
├── adaptive_filter.py       # VSLMS adaptive filter (core algorithm)
├── visualization.py         # Signal, FFT, and recovery plots
├── snr_calculator.py        # SNR, MSE, Correlation quality metrics
└── ecg_test.py              # Real-world ECG noise cancellation test
```

---

## ⚙️ How It Works

```
Noisy Signal
     │
     ▼
[FFT Spectral Analysis]
     │
     ▼
[Interference Detection] ← automatic peak detection, no assumed frequency
     │
     ▼
[Reference Signal Generation] ← synthesized from detected frequencies
     │
     ▼
[VSLMS Adaptive Filter] ← variable step-size, NLMS normalized
     │
     ▼
Cleaned Signal
     │
     ▼
[SNR Quality Report + Plots]
```

### Variable Step-Size Update Rule:
```
μ(n) = μ_max / (1 + β × E[e²(n)])

w(n+1) = w(n) + (μ(n) / ||x(n)||²) × e(n) × x(n)
```
- Large error → large μ → **fast convergence**
- Small error → small μ → **low misadjustment, stable**

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy scipy matplotlib
```

### Run the project

```bash
python main.py
```

You will be prompted to choose a test mode:

```
=== Intelligent Noise Detection System (VSLMS) ===

Select Test Mode:
  1. Synthetic signal test
  2. Real-world ECG test

Enter mode (1 or 2):
```

---

## 🧪 Mode 1 — Synthetic Signal Test

Tests the system on a generated multi-frequency signal with random interference tones.

**Example input:**
```
Enter sampling frequency (Hz): 1000
Enter signal duration (seconds): 10
Enter number of interference frequencies: 3
```

**Output:**
- Clean vs Noisy signal plot
- FFT frequency spectrum
- Recovered signal after VSLMS filtering
- Convergence curve (step-size μ + squared error)
- SNR / MSE / Correlation quality report

---

## 🫀 Mode 2 — Real-World ECG Test

Tests the system on a realistic synthetic ECG signal corrupted with:
- **50 Hz / 60 Hz power-line interference** (+ harmonic)
- **EMG muscle artifact** (broadband noise)
- **Baseline wander** (0.3 Hz breathing drift)

**Example input:**
```
Sampling frequency (Hz): 1000
Duration (seconds): 10
Heart rate BPM: 72
Power-line frequency Hz [50 India / 60 US]: 50
```

**Output:**
- 4-panel ECG plot (clean / noisy / recovered / zoomed heartbeat comparison)
- FFT spectrum before and after (power-line spike removal visible)
- VSLMS convergence curve
- SNR quality report

---

## 📊 Sample Results

| Metric | Before Filtering | After Filtering |
|---|---|---|
| SNR (dB) | −3.44 | ~10–15 |
| MSE | 1.60 | ~0.05–0.2 |
| Correlation | 0.55 | ~0.90–0.97 |

> Results vary with sampling frequency, signal duration, and number of interference frequencies.

---

## 📦 Module Descriptions

### `signal_generator.py`
Generates a clean multi-frequency base signal (5 Hz + 15 Hz + 30 Hz) and overlays random interference tones in a configurable frequency range.

### `spectral_analysis.py`
Computes FFT magnitude spectrum and spectrogram using `scipy.fft` and `scipy.signal`.

### `interference_detector.py`
Detects interference frequencies automatically by finding peaks in the normalized FFT spectrum. Protects base signal frequencies using a `min_freq` floor.

### `adaptive_filter.py`
Implements the **Variable Step-Size LMS (VSLMS)** adaptive filter:
- Synthesizes a reference signal from detected frequencies
- Adapts step-size dynamically per sample
- Uses NLMS normalization for stability at high sampling rates
- Auto-scales filter order based on `fs`

### `visualization.py`
Plots clean/noisy signals, FFT spectrum, and 3-panel recovered signal comparison.

### `snr_calculator.py`
Computes **SNR (dB)**, **MSE**, and **Pearson Correlation** before and after filtering. Prints a formatted quality report with interpretation and renders a comparison bar chart.

### `ecg_test.py`
Generates a realistic PQRST ECG signal and corrupts it with real-world biomedical noise types. Runs the full VSLMS pipeline and plots ECG-specific comparisons including a zoomed heartbeat panel and frequency spectrum comparison.

---

## 📈 Outputs at a Glance

| Plot | Description |
|---|---|
| Signal Comparison | Clean vs Noisy time-domain |
| FFT Spectrum | Interference spikes in frequency domain |
| Recovered Signal | 3-panel: clean / noisy / filtered |
| Convergence Curve | μ over time + squared error drop |
| SNR Bar Chart | Before vs after on SNR, MSE, Correlation |
| ECG 4-panel | Full ECG + zoomed heartbeat recovery |
| ECG Spectrum | Power-line spike removal visible |

---

## 🧠 Algorithm: VSLMS vs Fixed LMS

```
Fixed LMS:    w(n+1) = w(n) + μ × e(n) × x(n)        ← μ is constant
VSLMS:        w(n+1) = w(n) + μ(n) × e(n) × x(n)     ← μ(n) adapts every sample
```

**Why VSLMS is better:**
- Fixed μ forces a tradeoff: large μ = fast but unstable, small μ = stable but slow
- VSLMS resolves this — starts large (fast convergence), shrinks as signal stabilizes

---

## 📚 References

- S. Haykin, *Adaptive Filter Theory*, 5th ed., Pearson, 2014
- B. Widrow & S. D. Stearns, *Adaptive Signal Processing*, Prentice Hall, 1985
- Base Paper: *"Adaptive LMS-Based Noise Cancellation for Biomedical Signals"* (Paper 2)

---

## 👤 Author

> Mini Project — Signal Processing  
> Adaptive Noise Cancellation using FFT + VSLMS  
> Python 3.x | VS Code
