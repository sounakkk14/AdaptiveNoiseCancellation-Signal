"""
Intelligent Noise Detection System — web app
Variable Step-Size LMS adaptive noise cancellation, by SOUNAK C.

Run locally:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from vslms_core import run_synthetic, run_ecg

AUTHOR   = "SOUNAK C"
YEAR     = "2026"
REPO_URL = "https://github.com/sounakkk14/AdaptiveNoiseCancellation-Signal"

# Series colours (validated categorical slots, dark-surface steps)
C_CLEAN   = "#199e70"   # aqua
C_NOISY   = "#d95926"   # orange
C_CLEANED = "#3987e5"   # blue
C_MUTED   = "#8b949e"
GRID      = "rgba(139,148,158,0.15)"
MAX_POINTS = 4000       # points drawn per trace (display only)

st.set_page_config(page_title="VSLMS Noise Cancellation",
                   page_icon="〰️", layout="wide")


# ── Helpers ──────────────────────────────────────────────────

@st.cache_data(show_spinner=False, max_entries=32)
def process(mode, params, seed):
    if mode == "ecg":
        return run_ecg(seed=seed, **params)
    return run_synthetic(seed=seed, **params)


def thin(*arrays):
    """Stride arrays down to MAX_POINTS for plotting."""
    step = max(1, len(arrays[0]) // MAX_POINTS)
    return [a[::step] for a in arrays]


def style(fig, height):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def line(x, y, name, color, width=1.2, dash=None):
    return go.Scatter(x=x, y=y, name=name, mode="lines",
                      line=dict(color=color, width=width, dash=dash))


def fmt_freqs(freqs):
    return ", ".join(f"{f:g} Hz" for f in np.round(freqs, 1)) or "none"


# ── Sidebar ──────────────────────────────────────────────────

if "seed" not in st.session_state:
    st.session_state.seed = 2026

with st.sidebar:
    st.markdown("### ◈ Test mode")
    mode_label = st.radio(
        "Test mode", ["Synthetic signal", "ECG (biomedical)"],
        label_visibility="collapsed")
    mode = "ecg" if mode_label.startswith("ECG") else "synthetic"

    st.markdown("### ⚙ Signal")
    fs       = st.slider("Sampling frequency (Hz)", 500, 5000, 1000, 100)
    duration = st.slider("Duration (s)", 2, 20, 10)

    if mode == "synthetic":
        n_noise   = st.slider("Interference frequencies", 1, 8, 3)
        amplitude = st.slider("Noise amplitude", 0.1, 2.0, 0.8, 0.1)
        signal_params = dict(fs=fs, duration=duration,
                             n_noise=n_noise, amplitude=amplitude)
        default_threshold = 0.15
    else:
        heart_rate = st.slider("Heart rate (BPM)", 40, 180, 72)
        powerline  = st.radio("Power-line frequency",
                              [50.0, 60.0], horizontal=True,
                              format_func=lambda f: f"{f:g} Hz "
                              + ("(India/EU)" if f == 50 else "(US)"))
        noise_amp  = st.slider("Power-line amplitude", 0.05, 1.0, 0.3, 0.05)
        add_emg    = st.checkbox("Add EMG muscle noise", True)
        add_wander = st.checkbox("Add baseline wander", True)
        signal_params = dict(fs=fs, duration=duration,
                             heart_rate=heart_rate, powerline=powerline,
                             noise_amplitude=noise_amp,
                             add_emg=add_emg, add_wander=add_wander)
        default_threshold = 0.10

    st.markdown("### ⚡ VSLMS tuning")
    mu_max    = st.slider("μ max (step size)", 0.01, 0.5, 0.3, 0.01)
    mu_min    = st.slider("μ min (floor)", 0.001, 0.05, 0.001, 0.001,
                          format="%.3f")
    threshold = st.slider("Detection threshold", 0.05, 0.5,
                          default_threshold, 0.01)

    if st.button("🎲 New random signal", width="stretch"):
        st.session_state.seed = int(np.random.randint(0, 1_000_000))

    st.caption(f"Seed {st.session_state.seed}. Results update "
               "automatically when you change a setting.")

params = dict(signal_params, mu_max=mu_max, mu_min=mu_min,
              threshold=threshold)

if mu_min >= mu_max:
    st.error("μ min must be smaller than μ max.")
    st.stop()


# ── Header ───────────────────────────────────────────────────

st.title("〰️ Intelligent Noise Detection System")
st.markdown(
    "Adaptive noise cancellation with a **Variable Step-Size LMS (VSLMS)** "
    "filter. Interference is found automatically from the FFT spectrum, "
    "then removed sample by sample. "
    f"Built by **{AUTHOR}** · [Source on GitHub]({REPO_URL})")

with st.spinner("Running VSLMS filter…"):
    r = process(mode, params, st.session_state.seed)

m = r["metrics"]

# ── Metric tiles ─────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
c1.metric("SNR improvement", f"{m['SNR Improvement (dB)']:+.2f} dB")
c2.metric("SNR after filtering", f"{m['SNR After Filtering (dB)']:.2f} dB",
          f"from {m['SNR Before Filtering (dB)']:.2f} dB",
          delta_color="off", delta_arrow="off")
c3.metric("MSE after filtering", f"{m['MSE After Filtering']:.4f}",
          f"from {m['MSE Before Filtering']:.4f}", delta_color="off", delta_arrow="off")
c4.metric("Correlation with clean", f"{m['Correlation After']:.3f}",
          f"from {m['Correlation Before']:.3f}", delta_color="off", delta_arrow="off")

d1, d2 = st.columns(2)
d1.markdown(f"**Injected interference:** {fmt_freqs(r['actual'])}")
d2.markdown(f"**Auto-detected:** {fmt_freqs(r['detected'])}"
            f" · filter order M = {r['filter_order']}")

if len(r["detected"]) == 0:
    st.warning("No interference was detected, so the filter had nothing "
               "to cancel. Try lowering the detection threshold.")

tab_sig, tab_fft, tab_conv, tab_metrics, tab_about = st.tabs(
    ["Signals", "Spectrum", "Convergence", "Metrics", "How it works"])

# ── Signals ──────────────────────────────────────────────────

with tab_sig:
    t, clean, noisy, cleaned = thin(r["t"], r["clean"], r["noisy"],
                                    r["cleaned"])
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.07,
                        subplot_titles=("Clean signal (reference)",
                                        "Noisy signal",
                                        "Recovered after VSLMS"))
    fig.add_trace(line(t, clean, "Clean", C_CLEAN), 1, 1)
    fig.add_trace(line(t, noisy, "Noisy", C_NOISY), 2, 1)
    fig.add_trace(line(t, cleaned, "Recovered", C_CLEANED), 3, 1)
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    style(fig, 620).update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")

    zoom = min(2.0, duration)
    n = int(zoom * fs)
    fig = go.Figure([
        line(r["t"][:n], r["clean"][:n], "Clean", C_CLEAN, 2),
        line(r["t"][:n], r["cleaned"][:n], "Recovered", C_CLEANED, 1.5,
             "dot"),
    ])
    fig.update_layout(title=f"Zoom: first {zoom:g} s, clean vs recovered")
    fig.update_xaxes(title_text="Time (s)")
    st.plotly_chart(style(fig, 320), width="stretch")
    st.caption("The first few milliseconds of the recovered signal are "
               "zero while the filter fills its tap buffer.")

# ── Spectrum ─────────────────────────────────────────────────

with tab_fft:
    lim = min(300, fs // 2)
    k = r["freqs_fft"] <= lim
    fig = go.Figure([
        line(r["freqs_fft"][k], r["spectrum"][k], "Noisy", C_NOISY),
        line(r["freqs_fft"][k], r["spectrum_cleaned"][k], "Recovered",
             C_CLEANED),
    ])
    for f in r["detected"]:
        fig.add_vline(x=f, line=dict(color=C_MUTED, width=1, dash="dash"),
                      annotation_text=f"{f:g} Hz",
                      annotation_font_color=C_MUTED)
    fig.update_layout(title="FFT magnitude spectrum "
                            "(dashed lines = detected interference)")
    fig.update_xaxes(title_text="Frequency (Hz)")
    fig.update_yaxes(title_text="Magnitude")
    st.plotly_chart(style(fig, 460), width="stretch")

# ── Convergence ──────────────────────────────────────────────

with tab_conv:
    M = r["filter_order"]
    idx = np.arange(M, len(r["mu_hist"]))   # skip the tap-buffer warm-up
    win = max(1, int(0.05 * fs))            # 50 ms moving average
    resid = (r["cleaned"] - r["clean"]) ** 2
    learning = np.convolve(resid, np.ones(win) / win, mode="same")
    i, mu, lc = thin(idx, r["mu_hist"][M:], learning[M:])
    col_a, col_b = st.columns(2)
    fig = go.Figure([line(i, mu, "μ(n)", C_CLEANED)])
    fig.update_layout(title="Variable step size μ(n)", showlegend=False)
    fig.update_xaxes(title_text="Sample n")
    col_a.plotly_chart(style(fig, 380), width="stretch")

    fig = go.Figure([line(i, lc, "Residual error", C_NOISY, 1.5)])
    fig.update_layout(title="Learning curve: (recovered − clean)², 50 ms "
                            "average", showlegend=False)
    fig.update_xaxes(title_text="Sample n")
    col_b.plotly_chart(style(fig, 380), width="stretch")
    st.caption("μ(n) = μ_max / (1 + β·E[e²(n)]): the step size is large "
               "while the error is large (fast convergence) and shrinks "
               "as the filter settles (low misadjustment). The learning "
               "curve drops as the filter weights converge.")

# ── Metrics ──────────────────────────────────────────────────

with tab_metrics:
    table = pd.DataFrame({
        "Before filtering": [m["SNR Before Filtering (dB)"],
                             m["MSE Before Filtering"],
                             m["Correlation Before"]],
        "After filtering":  [m["SNR After Filtering (dB)"],
                             m["MSE After Filtering"],
                             m["Correlation After"]],
    }, index=["SNR (dB)", "MSE", "Correlation with clean signal"])
    table["Change"] = table["After filtering"] - table["Before filtering"]
    st.dataframe(table.style.format("{:.4f}"), width="stretch")

    imp, corr = m["SNR Improvement (dB)"], m["Correlation After"]
    verdict = ("Excellent filtering: strong SNR improvement" if imp > 10
               else "Good filtering: noticeable improvement" if imp > 5
               else "Moderate filtering: some improvement" if imp > 0
               else "Filtering did not improve the signal; check the "
                    "parameters")
    st.markdown(f"**Verdict:** {verdict}. Correlation after filtering is "
                f"{corr:.3f}.")
    if mode == "ecg" and (signal_params["add_emg"]
                          or signal_params["add_wander"]):
        st.info("VSLMS targets the narrow-band power-line interference. "
                "Broadband EMG noise and baseline wander are not in the "
                "reference signal, so they remain in the output and limit "
                "the achievable SNR. Untick them in the sidebar to see "
                "power-line cancellation alone.")

# ── About ────────────────────────────────────────────────────

with tab_about:
    st.markdown(r"""
#### Pipeline
1. **FFT spectral analysis** of the noisy signal.
2. **Automatic interference detection**: peaks in the normalised spectrum
   above the threshold, ignoring the low band where the wanted signal lives.
3. **Reference synthesis**: a sum of sinusoids at the detected frequencies.
4. **VSLMS adaptive filter** estimates the interference from the reference
   and subtracts it; the filter error *is* the cleaned signal.
5. **Quality report**: SNR, MSE and correlation against the clean signal.

#### Update rule
$$\mu(n) = \frac{\mu_{max}}{1 + \beta\,\hat{E}[e^2(n)]},\qquad
w(n+1) = w(n) + \frac{\mu(n)}{\lVert x(n)\rVert^2}\,e(n)\,x(n)$$

#### Improvements over the base paper
| Base paper | This project |
|---|---|
| Fixed step-size LMS | Variable step size that adapts to error power |
| Interference frequency assumed known | Detected automatically from the FFT |
| Convergence depends on signal scale | NLMS normalisation, filter order scaled with sampling rate |
""")

st.divider()
st.caption(f"© {AUTHOR} {YEAR} · Adaptive Signal Processing mini project · "
           f"[GitHub]({REPO_URL})")
