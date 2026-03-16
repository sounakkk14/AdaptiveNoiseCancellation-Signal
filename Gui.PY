"""
╔══════════════════════════════════════════════════════════╗
║   INTELLIGENT NOISE DETECTION SYSTEM — VSLMS            ║
║   Developed by: SOUNAK C                                 ║
║   Adaptive LMS-Based Noise Cancellation                  ║
╚══════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec

# ── Signature ──────────────────────────────────────────────
AUTHOR    = "SOUNAK C"
PROJECT   = "Intelligent Noise Detection System"
YEAR      = "2026"
WATERMARK = f"© {AUTHOR} | {YEAR}"


# ══════════════════════════════════════════════════════════
#  SPLASH SCREEN  (uses root.after — no threads)
# ══════════════════════════════════════════════════════════

def show_splash(root):
    splash = tk.Toplevel()
    splash.overrideredirect(True)
    splash.configure(bg="#0a0a0f")

    w, h = 620, 340
    sw = splash.winfo_screenwidth()
    sh = splash.winfo_screenheight()
    splash.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    canvas = tk.Canvas(splash, width=w, height=h,
                        bg="#0a0a0f", highlightthickness=0)
    canvas.pack()

    canvas.create_rectangle(2, 2, w-2, h-2,
                             outline="#00f5ff", width=2)
    canvas.create_rectangle(6, 6, w-6, h-6,
                             outline="#1a3a3a", width=1)

    canvas.create_text(w//2, 80,
                        text="INTELLIGENT NOISE",
                        font=("Courier New", 26, "bold"),
                        fill="#00f5ff")
    canvas.create_text(w//2, 118,
                        text="DETECTION SYSTEM",
                        font=("Courier New", 26, "bold"),
                        fill="#00f5ff")

    canvas.create_line(60, 145, w-60, 145,
                        fill="#1a3a3a", width=1)

    canvas.create_text(w//2, 170,
                        text="Variable Step-Size LMS Adaptive Filter",
                        font=("Courier New", 12),
                        fill="#7ecfcf")

    canvas.create_text(w//2, 215,
                        text=f"Developed by  {AUTHOR}",
                        font=("Courier New", 14, "bold"),
                        fill="#ffffff")

    canvas.create_text(w//2, 245,
                        text="Adaptive Signal Processing  |  Python",
                        font=("Courier New", 10),
                        fill="#445566")

    bar_x, bar_y = 60, 282
    bar_w, bar_h = w - 120, 10
    canvas.create_rectangle(bar_x, bar_y,
                             bar_x + bar_w, bar_y + bar_h,
                             outline="#1a3a3a", fill="#111122")

    progress_rect = canvas.create_rectangle(
        bar_x, bar_y, bar_x, bar_y + bar_h,
        outline="", fill="#00f5ff")

    status_text = canvas.create_text(
        w//2, 308, text="Initializing...",
        font=("Courier New", 9), fill="#445566")

    steps = [
        "Loading signal modules...",
        "Initializing VSLMS engine...",
        "Configuring FFT pipeline...",
        "Preparing GUI...",
        "Ready.",
    ]

    step_index = [0]

    def next_step():
        i = step_index[0]
        if i < len(steps):
            progress = (i + 1) / len(steps)
            fill_w   = int(bar_w * progress)
            canvas.coords(progress_rect,
                          bar_x, bar_y,
                          bar_x + fill_w, bar_y + bar_h)
            canvas.itemconfig(status_text, text=steps[i])
            step_index[0] += 1
            splash.after(400, next_step)
        else:
            splash.after(300, finish)

    def finish():
        splash.destroy()
        root.deiconify()

    splash.after(100, next_step)


# ══════════════════════════════════════════════════════════
#  SIGNAL PROCESSING
# ══════════════════════════════════════════════════════════

def generate_clean_signal(t):
    return (np.sin(2*np.pi*5*t) +
            0.6*np.sin(2*np.pi*15*t) +
            0.3*np.sin(2*np.pi*30*t))


def generate_noisy_signal(t, fs, num_interference, amplitude=0.8):
    clean = generate_clean_signal(t)
    freqs = np.random.randint(35, 150, num_interference)
    noise = np.zeros(len(t))
    for f in freqs:
        noise += amplitude * np.sin(2*np.pi*f*t)
    return clean, clean + noise, freqs


def compute_fft(signal, fs):
    from scipy.fft import fft, fftfreq
    N   = len(signal)
    yf  = fft(signal)
    xf  = fftfreq(N, 1/fs)
    mag = np.abs(yf[:N//2])
    frq = xf[:N//2]
    return frq, mag


def detect_interference(freqs, spectrum,
                         threshold=0.15, min_freq=35):
    from scipy.signal import find_peaks
    norm = spectrum / (np.max(spectrum) + 1e-8)
    peaks, _ = find_peaks(norm, height=threshold, distance=5)
    detected = freqs[peaks]
    return detected[detected >= min_freq]


def vslms_filter(noisy, reference,
                  mu_max=0.3, mu_min=0.001,
                  M=128, alpha=0.999, beta=0.005):
    N    = len(noisy)
    w    = np.zeros(M)
    out  = np.zeros(N)
    err  = np.zeros(N)
    mu_h = np.zeros(N)
    ep   = np.var(noisy) + 1e-8
    for n in range(M, N):
        x  = reference[n-M:n][::-1]
        y  = np.dot(w, x)
        e  = noisy[n] - y
        ep = alpha*ep + (1-alpha)*(e**2)
        mu = np.clip(mu_max/(1.0 + beta*ep), mu_min, mu_max)
        xp = np.dot(x, x) + 1e-8
        w += (mu/xp)*e*x
        out[n]  = e
        err[n]  = e
        mu_h[n] = mu
    return np.real(out), err, mu_h


def run_processing(fs, duration, n_noise, amp,
                    mu_max, mu_min, threshold):
    t = np.arange(0, duration, 1/fs)
    clean, noisy, actual_freqs = generate_noisy_signal(
        t, fs, n_noise, amplitude=amp)

    freqs_fft, spectrum = compute_fft(noisy, fs)
    detected = detect_interference(
        freqs_fft, spectrum,
        threshold=threshold, min_freq=35)

    N   = len(noisy)
    t_r = np.arange(N) / fs
    ref = np.zeros(N)
    for f in detected:
        if 0 < f < fs/2:
            ref += np.sin(2*np.pi*f*t_r)
    if np.max(np.abs(ref)) > 0:
        ref = (ref / np.max(np.abs(ref)) *
               np.std(noisy) * 0.5)

    M = int(np.clip(fs*0.01, 64, 512))
    cleaned, error, mu_hist = vslms_filter(
        noisy, ref,
        mu_max=mu_max, mu_min=mu_min, M=M)

    def snr_db(clean, sig):
        sp  = np.mean(clean**2)
        np_ = np.mean((sig - clean)**2)
        return 10*np.log10(sp / (np_ + 1e-8))

    snr_b = snr_db(clean, noisy)
    snr_a = snr_db(clean, cleaned)

    return {
        "t": t, "clean": clean,
        "noisy": noisy, "cleaned": cleaned,
        "error": error, "mu_hist": mu_hist,
        "freqs_fft": freqs_fft, "spectrum": spectrum,
        "detected": detected, "actual_freqs": actual_freqs,
        "snr_before": snr_b, "snr_after": snr_a,
        "snr_improvement": snr_a - snr_b,
        "mse_before": float(np.mean((clean-noisy)**2)),
        "mse_after":  float(np.mean((clean-cleaned)**2)),
        "corr_after": float(np.corrcoef(clean, cleaned)[0, 1]),
    }


# ══════════════════════════════════════════════════════════
#  WATERMARK
# ══════════════════════════════════════════════════════════

def add_watermark(fig):
    fig.text(0.99, 0.01, WATERMARK,
             ha='right', va='bottom',
             fontsize=7, color='#888888',
             alpha=0.6, style='italic',
             fontfamily='monospace')


# ══════════════════════════════════════════════════════════
#  MAIN GUI
# ══════════════════════════════════════════════════════════

class NoiseCancellationGUI:

    BG      = "#0d1117"
    PANEL   = "#161b22"
    ACCENT  = "#00f5ff"
    ACCENT2 = "#7ee787"
    RED     = "#ff6b6b"
    TEXT    = "#e6edf3"
    SUBTEXT = "#8b949e"
    BORDER  = "#30363d"
    FONT    = "Courier New"

    def __init__(self, root):
        self.root         = root
        self.root.title(f"{PROJECT}  —  {AUTHOR}")
        self.root.configure(bg=self.BG)
        self.root.geometry("1400x860")
        self.root.minsize(1200, 750)
        self.result       = None
        self.anim         = None
        self.anim_running = False
        self._build_ui()

    # ── Layout ─────────────────────────────────────────────
    def _build_ui(self):
        hdr = tk.Frame(self.root, bg=self.BG, pady=8)
        hdr.pack(fill="x", padx=20)
        tk.Label(hdr,
                 text="◈  INTELLIGENT NOISE DETECTION SYSTEM",
                 font=(self.FONT, 15, "bold"),
                 fg=self.ACCENT, bg=self.BG).pack(side="left")
        tk.Label(hdr,
                 text=f"Developed by {AUTHOR}  |  VSLMS  |  {YEAR}",
                 font=(self.FONT, 9),
                 fg=self.SUBTEXT, bg=self.BG).pack(
                     side="right", pady=6)

        tk.Frame(self.root, bg=self.ACCENT,
                 height=1).pack(fill="x", padx=20,
                                 pady=(0, 8))

        body = tk.Frame(self.root, bg=self.BG)
        body.pack(fill="both", expand=True,
                   padx=20, pady=(0, 8))

        self._build_controls(body)
        self._build_plot_area(body)

    def _build_controls(self, parent):
        ctrl = tk.Frame(parent, bg=self.PANEL, width=265)
        ctrl.pack(side="left", fill="y", padx=(0, 10))
        ctrl.pack_propagate(False)

        def section(txt):
            tk.Label(ctrl, text=txt,
                     font=(self.FONT, 9, "bold"),
                     fg=self.ACCENT, bg=self.PANEL,
                     anchor="w").pack(fill="x", padx=14,
                                       pady=(14, 2))
            tk.Frame(ctrl, bg=self.BORDER,
                     height=1).pack(fill="x", padx=14,
                                     pady=(0, 6))

        def slider(label, from_, to, default, res=1):
            tk.Label(ctrl, text=label,
                     font=(self.FONT, 8),
                     fg=self.SUBTEXT, bg=self.PANEL,
                     anchor="w").pack(fill="x", padx=14,
                                       pady=(3, 0))
            var = tk.DoubleVar(value=default)
            tk.Scale(ctrl, from_=from_, to=to,
                     orient="horizontal", variable=var,
                     resolution=res,
                     bg=self.PANEL, fg=self.TEXT,
                     troughcolor=self.BG,
                     activebackground=self.ACCENT,
                     highlightthickness=0, bd=0,
                     showvalue=True,
                     font=(self.FONT, 8)).pack(
                         fill="x", padx=14, pady=(0, 4))
            return var

        section("⚙  PARAMETERS")
        self.var_fs       = slider("Sampling Frequency (Hz)",
                                    500, 5000, 1000, 100)
        self.var_duration = slider("Duration (seconds)",
                                    2, 20, 10, 1)
        self.var_noise    = slider("Interference Frequencies",
                                    1, 8, 3, 1)
        self.var_amp      = slider("Noise Amplitude",
                                    0.1, 2.0, 0.8, 0.1)

        section("⚡  VSLMS TUNING")
        self.var_mumax  = slider("μ Max (step size)",
                                   0.01, 0.5, 0.3, 0.01)
        self.var_mumin  = slider("μ Min (floor)",
                                   0.001, 0.05, 0.001, 0.001)
        self.var_thresh = slider("Detection Threshold",
                                   0.05, 0.5, 0.15, 0.01)

        section("◈  MODE")
        self.var_mode = tk.StringVar(value="static")
        for txt, val in [("Static Analysis",    "static"),
                          ("Real-Time Animation", "realtime")]:
            tk.Radiobutton(ctrl, text=txt,
                           variable=self.var_mode, value=val,
                           font=(self.FONT, 8),
                           fg=self.TEXT, bg=self.PANEL,
                           selectcolor=self.BG,
                           activebackground=self.PANEL,
                           activeforeground=self.ACCENT).pack(
                               anchor="w", padx=20, pady=2)

        section("▶  ACTIONS")

        def btn(txt, cmd, color=None):
            c = color or self.ACCENT
            tk.Button(ctrl, text=txt, command=cmd,
                      font=(self.FONT, 9, "bold"),
                      fg=self.BG, bg=c,
                      activebackground=self.TEXT,
                      relief="flat", bd=0,
                      pady=7, cursor="hand2").pack(
                          fill="x", padx=14, pady=3)

        btn("▶  RUN ANALYSIS", self._run)
        btn("⟳  RESET",        self._reset,     "#444455")
        btn("💾  SAVE PLOTS",  self._save_plots, "#2ea043")

        tk.Frame(ctrl, bg=self.BORDER, height=1).pack(
            fill="x", padx=14, pady=(12, 6))

        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(ctrl, textvariable=self.status_var,
                 font=(self.FONT, 8), fg=self.ACCENT2,
                 bg=self.PANEL, wraplength=225,
                 justify="left", anchor="w").pack(
                     fill="x", padx=14, pady=2)

        tk.Frame(ctrl, bg=self.BORDER, height=1).pack(
            fill="x", padx=14, pady=(8, 4))

        mf = tk.Frame(ctrl, bg=self.PANEL)
        mf.pack(fill="x", padx=14, pady=2)
        self._mlbl = {}
        for key in ["SNR Before", "SNR After",
                     "Improvement", "MSE Before",
                     "MSE After", "Correlation"]:
            row = tk.Frame(mf, bg=self.PANEL)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"{key}:",
                     font=(self.FONT, 7),
                     fg=self.SUBTEXT, bg=self.PANEL,
                     width=14, anchor="w").pack(side="left")
            lbl = tk.Label(row, text="—",
                           font=(self.FONT, 7, "bold"),
                           fg=self.TEXT, bg=self.PANEL)
            lbl.pack(side="left")
            self._mlbl[key] = lbl

        tk.Label(ctrl, text=f"◈ {AUTHOR}  {YEAR}",
                 font=(self.FONT, 8, "bold"),
                 fg="#334455", bg=self.PANEL).pack(
                     side="bottom", pady=8)

    def _build_plot_area(self, parent):
        self.plot_frame = tk.Frame(parent, bg=self.BG)
        self.plot_frame.pack(side="left", fill="both",
                              expand=True)

        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(11, 7),
                               facecolor=self.BG)
        self.fig.subplots_adjust(
            hspace=0.42, wspace=0.3,
            left=0.06, right=0.97,
            top=0.93, bottom=0.08)

        gs = GridSpec(3, 3, figure=self.fig)
        self.ax_clean   = self.fig.add_subplot(gs[0, :2])
        self.ax_noisy   = self.fig.add_subplot(gs[1, :2])
        self.ax_cleaned = self.fig.add_subplot(gs[2, :2])
        self.ax_fft     = self.fig.add_subplot(gs[0, 2])
        self.ax_conv    = self.fig.add_subplot(gs[1, 2])
        self.ax_snr     = self.fig.add_subplot(gs[2, 2])

        self._placeholder_text()
        add_watermark(self.fig)

        self.canvas = FigureCanvasTkAgg(
            self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(
            fill="both", expand=True)
        self.canvas.draw()

    def _placeholder_text(self):
        for ax, title in [
            (self.ax_clean,   "Clean Signal"),
            (self.ax_noisy,   "Noisy Signal"),
            (self.ax_cleaned, "Recovered Signal"),
            (self.ax_fft,     "FFT Spectrum"),
            (self.ax_conv,    "Convergence"),
            (self.ax_snr,     "SNR Comparison"),
        ]:
            ax.cla()
            ax.set_facecolor("#0d1117")
            ax.set_title(title, color=self.ACCENT,
                         fontsize=8, fontfamily=self.FONT)
            ax.text(0.5, 0.5, "Run analysis to populate",
                    ha="center", va="center",
                    color="#334455", fontsize=8,
                    fontfamily=self.FONT,
                    transform=ax.transAxes)
            for sp in ax.spines.values():
                sp.set_edgecolor("#223344")
            ax.tick_params(colors="#556677", labelsize=7)

    # ── Run ────────────────────────────────────────────────
    def _run(self):
        if self.anim_running:
            self._stop_animation()
        self.status_var.set("Processing...")
        self.root.update()

        try:
            result = run_processing(
                fs        = int(self.var_fs.get()),
                duration  = int(self.var_duration.get()),
                n_noise   = int(self.var_noise.get()),
                amp       = float(self.var_amp.get()),
                mu_max    = float(self.var_mumax.get()),
                mu_min    = float(self.var_mumin.get()),
                threshold = float(self.var_thresh.get()),
            )
            self.result = result
            self._update_metrics(result)

            if self.var_mode.get() == "realtime":
                self._run_realtime(result)
            else:
                self._plot_static(result)

            self.status_var.set(
                f"Done. {len(result['detected'])} freq(s) detected. "
                f"SNR +{result['snr_improvement']:.1f} dB."
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set(f"Error: {e}")

    def _update_metrics(self, r):
        self._mlbl["SNR Before"].config(
            text=f"{r['snr_before']:.2f} dB", fg=self.RED)
        self._mlbl["SNR After"].config(
            text=f"{r['snr_after']:.2f} dB", fg=self.ACCENT2)
        imp = r["snr_improvement"]
        self._mlbl["Improvement"].config(
            text=f"+{imp:.2f} dB" if imp > 0 else f"{imp:.2f} dB",
            fg=self.ACCENT2 if imp > 0 else self.RED)
        self._mlbl["MSE Before"].config(
            text=f"{r['mse_before']:.4f}", fg=self.RED)
        self._mlbl["MSE After"].config(
            text=f"{r['mse_after']:.4f}", fg=self.ACCENT2)
        self._mlbl["Correlation"].config(
            text=f"{r['corr_after']:.4f}", fg=self.ACCENT)

    # ── Static plots ───────────────────────────────────────
    def _plot_static(self, r):
        t       = r["t"]
        clean   = r["clean"]
        noisy   = r["noisy"]
        cleaned = r["cleaned"]
        step    = max(1, len(t)//2000)

        def sax(ax, title, xl="", yl=""):
            ax.set_facecolor("#0d1117")
            ax.set_title(title, color=self.ACCENT,
                         fontsize=8, fontfamily=self.FONT,
                         pad=4)
            ax.set_xlabel(xl, color=self.SUBTEXT,
                          fontsize=7, fontfamily=self.FONT)
            ax.set_ylabel(yl, color=self.SUBTEXT,
                          fontsize=7, fontfamily=self.FONT)
            ax.tick_params(colors="#556677", labelsize=6)
            ax.grid(True, alpha=0.1, color="#334455")
            for sp in ax.spines.values():
                sp.set_edgecolor("#223344")

        self.ax_clean.cla()
        self.ax_clean.plot(t[::step], clean[::step],
                            color=self.ACCENT2, lw=0.8)
        sax(self.ax_clean, "Clean Signal", yl="Amplitude")

        self.ax_noisy.cla()
        self.ax_noisy.plot(t[::step], noisy[::step],
                            color=self.RED, lw=0.6)
        sax(self.ax_noisy, "Noisy Signal", yl="Amplitude")

        self.ax_cleaned.cla()
        self.ax_cleaned.plot(t[::step], cleaned[::step],
                              color=self.ACCENT, lw=0.8)
        sax(self.ax_cleaned, "Recovered Signal (VSLMS)",
            xl="Time (s)", yl="Amplitude")

        self.ax_fft.cla()
        self.ax_fft.plot(r["freqs_fft"], r["spectrum"],
                          color="#7ee787", lw=0.7)
        for f in r["detected"]:
            self.ax_fft.axvline(f, color=self.RED,
                                 lw=0.8, alpha=0.7,
                                 linestyle="--")
        self.ax_fft.set_xlim(0, 300)
        sax(self.ax_fft, "FFT Spectrum",
            xl="Hz", yl="Magnitude")

        self.ax_conv.cla()
        eq  = r["error"]**2
        win = max(1, len(eq)//500)
        sm  = np.convolve(eq, np.ones(win)/win, mode="same")
        self.ax_conv.plot(sm, color="#f0883e", lw=0.7)
        sax(self.ax_conv, "Error Convergence",
            xl="Sample", yl="e²(n)")

        self.ax_snr.cla()
        vals   = [r["snr_before"], r["snr_after"]]
        colors = [self.RED, self.ACCENT2]
        bars   = self.ax_snr.bar(["Before", "After"],
                                   vals, color=colors,
                                   width=0.4)
        for bar, val in zip(bars, vals):
            self.ax_snr.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f"{val:.1f}", ha="center",
                fontsize=7, color=self.TEXT,
                fontfamily=self.FONT)
        sax(self.ax_snr, "SNR (dB)", yl="dB")

        add_watermark(self.fig)
        self.canvas.draw()

    # ── Real-time animation ────────────────────────────────
    def _run_realtime(self, r):
        self.anim_running = True
        self.status_var.set("Real-time animation running...")

        t       = r["t"]
        clean   = r["clean"]
        noisy   = r["noisy"]
        cleaned = r["cleaned"]
        step    = max(1, len(t)//2000)

        T  = t[::step]
        C  = clean[::step]
        N  = noisy[::step]
        CL = cleaned[::step]
        EQ = (r["error"][::step])**2
        CHUNK = max(1, len(T)//80)

        for ax in self.fig.get_axes():
            ax.cla()
            ax.set_facecolor("#0d1117")
            for sp in ax.spines.values():
                sp.set_edgecolor("#223344")
            ax.tick_params(colors="#556677", labelsize=6)
            ax.grid(True, alpha=0.1, color="#334455")

        self.ax_clean.plot(T, C, color=self.ACCENT2, lw=0.8)
        self.ax_clean.set_title("Clean Signal",
            color=self.ACCENT, fontsize=8,
            fontfamily=self.FONT)

        self.ax_noisy.plot(T, N, color=self.RED, lw=0.6)
        self.ax_noisy.set_title("Noisy Signal",
            color=self.ACCENT, fontsize=8,
            fontfamily=self.FONT)

        self.ax_fft.plot(r["freqs_fft"], r["spectrum"],
                          color="#7ee787", lw=0.7)
        for f in r["detected"]:
            self.ax_fft.axvline(f, color=self.RED,
                                 lw=0.8, linestyle="--")
        self.ax_fft.set_xlim(0, 300)
        self.ax_fft.set_title("FFT Spectrum",
            color=self.ACCENT, fontsize=8,
            fontfamily=self.FONT)

        self.ax_cleaned.set_title(
            "Recovered Signal — Building...",
            color=self.ACCENT, fontsize=8,
            fontfamily=self.FONT)
        self.ax_cleaned.set_xlim(T[0], T[-1])
        self.ax_cleaned.set_ylim(
            np.min(CL)-0.2, np.max(CL)+0.2)

        self.ax_conv.set_title("Convergence",
            color=self.ACCENT, fontsize=8,
            fontfamily=self.FONT)
        self.ax_conv.set_xlim(0, len(T))
        self.ax_conv.set_ylim(
            0, max(np.percentile(EQ, 95), 0.01))

        self.ax_snr.set_title("SNR (dB)",
            color=self.ACCENT, fontsize=8,
            fontfamily=self.FONT)

        line_cl,  = self.ax_cleaned.plot(
            [], [], color=self.ACCENT, lw=0.9)
        line_err, = self.ax_conv.plot(
            [], [], color="#f0883e", lw=0.7)

        def update(frame):
            if not self.anim_running:
                return line_cl, line_err

            end  = min((frame+1)*CHUNK, len(T))
            line_cl.set_data(T[:end], CL[:end])
            line_err.set_data(np.arange(end), EQ[:end])

            prog  = end / len(T)
            snr_i = (r["snr_before"] +
                     prog*(r["snr_after"]-r["snr_before"]))

            self.ax_snr.cla()
            self.ax_snr.set_facecolor("#0d1117")
            for sp in self.ax_snr.spines.values():
                sp.set_edgecolor("#223344")
            bars = self.ax_snr.bar(
                ["Before", "Now"],
                [r["snr_before"], snr_i],
                color=[self.RED, self.ACCENT2], width=0.4)
            for bar, val in zip(
                    bars, [r["snr_before"], snr_i]):
                self.ax_snr.text(
                    bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.2,
                    f"{val:.1f}", ha="center",
                    fontsize=7, color=self.TEXT,
                    fontfamily=self.FONT)
            self.ax_snr.set_title("SNR (dB)",
                color=self.ACCENT, fontsize=8,
                fontfamily=self.FONT)
            self.ax_snr.tick_params(
                colors="#556677", labelsize=6)

            if end >= len(T):
                self.ax_cleaned.set_title(
                    "Recovered Signal — Complete ✓",
                    color=self.ACCENT2, fontsize=8,
                    fontfamily=self.FONT)
                self.anim_running = False

            add_watermark(self.fig)
            return line_cl, line_err

        total = (len(T)//CHUNK) + 1
        self.anim = animation.FuncAnimation(
            self.fig, update,
            frames=total, interval=30,
            blit=False, repeat=False)
        self.canvas.draw()

    def _stop_animation(self):
        if self.anim:
            self.anim.event_source.stop()
        self.anim_running = False

    def _reset(self):
        self._stop_animation()
        self._placeholder_text()
        for lbl in self._mlbl.values():
            lbl.config(text="—", fg=self.TEXT)
        self.status_var.set("Ready.")
        add_watermark(self.fig)
        self.canvas.draw()

    def _save_plots(self):
        if self.result is None:
            messagebox.showwarning(
                "No Data", "Run analysis first.")
            return
        fname = f"VSLMS_{AUTHOR.replace(' ', '_')}_{YEAR}.png"
        self.fig.savefig(fname, dpi=150,
                          bbox_inches="tight",
                          facecolor=self.BG)
        self.status_var.set(f"Saved → {fname}")
        messagebox.showinfo("Saved",
                             f"Plot saved as:\n{fname}")


# ══════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    show_splash(root)
    app = NoiseCancellationGUI(root)
    root.mainloop()