"""Measurements that stand in for ears: numbers and pictures Claude can reason about."""

from __future__ import annotations

import io

import numpy as np
from scipy import ndimage

from . import dsp
from .timeutil import fmt_time

EPS = 1e-12

BANDS: list[tuple[str, float, float]] = [
    ("sub 20-60", 20, 60),
    ("bass 60-120", 60, 120),
    ("low 120-250", 120, 250),
    ("low-mid 250-500", 250, 500),
    ("mid 500-1k", 500, 1000),
    ("upper-mid 1k-2k", 1000, 2000),
    ("presence 2k-4k", 2000, 4000),
    ("brilliance 4k-8k", 4000, 8000),
    ("air 8k-20k", 8000, 20000),
]


def db(x: float | np.ndarray) -> float | np.ndarray:
    return 20 * np.log10(np.maximum(x, 1e-10))


def band_levels(clip: np.ndarray, sr: int) -> dict[str, float]:
    """RMS level of each band in dBFS (averaged over time and channels)."""
    if len(clip) < 64:
        return {name: -200.0 for name, _, _ in BANDS}
    S, freqs, _, n_fft, _ = dsp.stft(clip, sr, 93.0)
    # Mean power spectrum, scaled so a full-scale sine reads about 0 dBFS in its band.
    power = np.mean(np.abs(S) ** 2, axis=(0, 2)) * 2.0
    out = {}
    for name, lo, hi in BANDS:
        sel = (freqs >= lo) & (freqs < hi)
        out[name] = round(float(10 * np.log10(np.sum(power[sel]) + EPS)), 1) if sel.any() else -200.0
    return out


def basic_stats(clip: np.ndarray, sr: int) -> dict:
    if len(clip) == 0:
        return {}
    peak = float(np.max(np.abs(clip)))
    rms = float(np.sqrt(np.mean(clip ** 2)))
    mono = clip.mean(axis=1)
    spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono)))) ** 2
    freqs = np.fft.rfftfreq(len(mono), 1 / sr)
    centroid = float(np.sum(freqs * spec) / (np.sum(spec) + EPS))
    stats = {
        "peak_dbfs": round(float(db(peak)), 1),
        "rms_dbfs": round(float(db(rms)), 1),
        "crest_db": round(float(db(peak) - db(rms)), 1),
        "spectral_centroid_hz": round(centroid),
        "clipped_samples": int(np.sum(np.abs(clip) >= 0.999)),
    }
    if clip.shape[1] == 2:
        l, r = clip[:, 0], clip[:, 1]
        denom = np.sqrt(np.sum(l ** 2) * np.sum(r ** 2)) + EPS
        stats["stereo_correlation"] = round(float(np.sum(l * r) / denom), 2)
    return stats


def compare_bands(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {k: round(a[k] - b[k], 1) for k in a}


def band_timeline(clip: np.ndarray, sr: int, t_offset: float, step_s: float) -> list[dict]:
    """Band levels in consecutive slices, to find *when* a band misbehaves."""
    S, freqs, times, _, _ = dsp.stft(clip, sr, 93.0)
    power = np.mean(np.abs(S) ** 2, axis=0) * 2.0
    rows = []
    n_steps = max(1, int(np.ceil(len(clip) / sr / step_s)))
    for i in range(n_steps):
        sel_t = (times >= i * step_s) & (times < (i + 1) * step_s)
        if not sel_t.any():
            continue
        frame = power[:, sel_t].mean(axis=1)
        row = {"time": fmt_time(t_offset + i * step_s)}
        for name, lo, hi in BANDS:
            sel = (freqs >= lo) & (freqs < hi)
            row[name.split()[0]] = round(float(10 * np.log10(np.sum(frame[sel]) + EPS)), 1)
        rows.append(row)
    return rows


# ------------------------------------------------------------------ detection


def _flux(S_pow: np.ndarray, freqs: np.ndarray, lo: float, hi: float, hop_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Band level (dB) per frame and its rise above the level of the previous ~60 ms."""
    sel = (freqs >= lo) & (freqs < hi)
    level = 10 * np.log10(np.sum(S_pow[sel], axis=0) + 1e-14)
    back = max(2, int(round(0.06 / hop_s)))
    prev = ndimage.minimum_filter1d(ndimage.uniform_filter1d(level, back, mode="nearest"),
                                    back, origin=0, mode="nearest")
    prev = np.concatenate([np.full(back, prev[0]), prev[:-back]])
    return level, level - prev


def find_thumps(clip: np.ndarray, sr: int, *, max_hz=150.0, min_rise_db=5.0) -> list[dict]:
    """Low-frequency bumps that are not notes: pedal thumps, bench/floor knocks.

    A struck note raises the low band *and* its harmonics up through the mids.
    A thump raises only the bottom, so we look for low-band onsets that bring
    almost no new energy above the low band.
    """
    mono = clip.mean(axis=1, keepdims=True)
    S, freqs, times, n_fft, hop = dsp.stft(mono, sr, 46.0, overlap=8)
    power = np.abs(S[0]) ** 2
    hop_s = hop / sr
    low_lvl, low_rise = _flux(power, freqs, 20.0, max_hz, hop_s)
    _, mid_rise = _flux(power, freqs, 300.0, 3000.0, hop_s)
    floor = np.percentile(low_lvl, 20)
    low_sel = (freqs >= 20.0) & (freqs < max_hz)
    hi_sel = (freqs >= max_hz * 1.25) & (freqs < 5000.0)
    ahead, behind = max(2, int(0.05 / hop_s)), max(2, int(0.07 / hop_s))
    lead = max(1, int(0.025 / hop_s))
    hits = []
    min_gap = max(1, int(0.12 / hop_s))
    cand = np.flatnonzero((low_rise >= min_rise_db) & (low_lvl > floor + 3.0))
    last = -min_gap
    for i in cand:
        w0, w1 = max(0, i - min_gap // 2), min(len(times), i + min_gap // 2 + 1)
        if low_rise[i] < low_rise[w0:w1].max():
            continue  # not the local peak
        if i - last < min_gap:
            continue
        # How much power arrived below max_hz versus above it. A struck note brings its
        # harmonics with it; a thump brings almost nothing above the low band.
        b0, b1 = max(0, i - lead - behind), max(1, i - lead)
        a1 = min(len(times), i + ahead)
        # Per-bin increases only, so other notes dying away cannot mask a note's new harmonics.
        grew = np.maximum(power[:, i:a1].mean(axis=1) - power[:, b0:b1].mean(axis=1), 0.0)
        d_low, d_hi = grew[low_sel].sum(), grew[hi_sel].sum()
        if d_low <= 0 or d_hi > 0.04 * d_low:
            continue
        mid = float(mid_rise[max(0, i - 3) : i + 4].max())
        last = i
        # Frame centre sits on the rise; the thump body follows it.
        hits.append({
            "time": fmt_time(times[i]), "seconds": round(float(times[i]), 3),
            "low_rise_db": round(float(low_rise[i]), 1),
            "mid_rise_db": round(mid, 1),
            "low_level_dbfs": round(float(low_lvl[i] + 10 * np.log10(2.0)), 1),
        })
    return hits


def find_clicks(clip: np.ndarray, sr: int, *, threshold=10.0, max_ms=3.0) -> list[dict]:
    """Clicks/ticks in any channel, as sample spans plus readable times."""
    merged: dict[int, dict] = {}
    for ch in range(clip.shape[1]):
        for hit in dsp.find_clicks(clip[:, ch], sr, threshold=threshold, max_ms=max_ms):
            key = int(round(hit["time"] * 500))  # 2 ms buckets merge the same click across channels
            best = merged.get(key) or merged.get(key - 1) or merged.get(key + 1)
            if best is None:
                merged[key] = dict(hit)
            else:
                best["start"] = min(best["start"], hit["start"])
                best["end"] = max(best["end"], hit["end"])
                best["strength"] = max(best["strength"], hit["strength"])
    return sorted(merged.values(), key=lambda h: h["start"])


# ------------------------------------------------------------------- pictures


def _log_spectrogram(clip: np.ndarray, sr: int, fmin: float, fmax: float, rows: int = 320, cols: int = 1400):
    """dBFS spectrogram on a log-frequency grid, pooled to at most rows x cols."""
    dur = len(clip) / sr
    # Short regions get a short window (time detail), long ones a long window (pitch detail).
    win_ms = float(np.clip(dur * 12.0, 23.0, 186.0))
    mono = clip.mean(axis=1, keepdims=True)
    S, freqs, times, n_fft, hop = dsp.stft(mono, sr, win_ms, overlap=8 if dur < 20 else 4)
    power = np.abs(S[0]) ** 2 * 4.0  # hann window: full-scale sine peaks near 0 dB
    fmax = min(fmax, sr / 2)
    edges = np.geomspace(fmin, fmax, rows + 1)
    bin_idx = np.clip(np.searchsorted(freqs, edges), 1, len(freqs) - 1)
    grid = np.empty((rows, power.shape[1]))
    for r in range(rows):
        lo, hi = bin_idx[r], max(bin_idx[r + 1], bin_idx[r] + 1)
        if hi - lo == 1 and lo > 0:
            # Fewer FFT bins than rows down here: interpolate between neighbours.
            fc = np.sqrt(edges[r] * edges[r + 1])
            j = int(np.clip(np.searchsorted(freqs, fc), 1, len(freqs) - 1))
            t = (fc - freqs[j - 1]) / (freqs[j] - freqs[j - 1])
            grid[r] = power[j - 1] * (1 - t) + power[j] * t
        else:
            grid[r] = power[lo:hi].max(axis=0)
    if grid.shape[1] > cols:
        factor = int(np.ceil(grid.shape[1] / cols))
        pad = (-grid.shape[1]) % factor
        g = np.pad(grid, ((0, 0), (0, pad)), mode="edge")
        grid = g.reshape(rows, -1, factor).max(axis=2)
    return 10 * np.log10(grid + 1e-14), edges


def _time_formatter(t_offset: float):
    from matplotlib.ticker import FuncFormatter

    def f(x, _pos):
        t = x + t_offset
        m, s = int(t // 60), t - 60 * int(t // 60)
        return f"{m}:{s:05.2f}" if (s % 1) > 1e-6 else f"{m}:{int(s):02d}"

    return FuncFormatter(f)


def render_spectrogram(
    panels: list[tuple[str, np.ndarray]],
    sr: int,
    t_offset: float,
    *,
    fmin: float = 20.0,
    fmax: float = 20000.0,
    db_range: tuple[float, float] = (-110.0, -20.0),
    marks: list[tuple[float, str]] | None = None,
    highlight: tuple[float, float] | None = None,
    diff: bool = False,
    waveform: bool = True,
) -> bytes:
    """JPEG image of one or more stacked spectrograms sharing a time axis.

    ``panels`` are (title, clip) pairs covering the same time span. With
    ``diff`` and two panels, a third panel shows second-minus-first in dB.
    ``marks`` are (seconds, label) vertical markers, ``highlight`` shades a
    time range; both use absolute track time.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter

    grids = [(_title, *_log_spectrogram(clip, sr, fmin, fmax)) for _title, clip in panels]
    dur = len(panels[0][1]) / sr
    n_spec = len(grids) + (1 if diff and len(grids) == 2 else 0)
    heights = ([1.0] if waveform else []) + [3.2] * n_spec
    fig, axes = plt.subplots(len(heights), 1, figsize=(14, 1.2 + 3.0 * n_spec + (1.0 if waveform else 0)),
                             sharex=True, gridspec_kw={"height_ratios": heights}, dpi=100)
    axes = np.atleast_1d(axes)
    ax_i = 0

    if waveform:
        ax = axes[0]
        clip = panels[-1][1]
        mono = clip.mean(axis=1)
        n = len(mono)
        cols = min(n, 2800)
        size = n // cols
        env = mono[: size * cols].reshape(cols, size)
        t = (np.arange(cols) + 0.5) * size / sr
        ax.fill_between(t, env.min(axis=1), env.max(axis=1), color="#3b6ea8", linewidth=0)
        peak = max(float(np.max(np.abs(mono))), 1e-4)
        ax.set_ylim(-peak * 1.1, peak * 1.1)
        ax.set_ylabel(f"wave\npeak {db(peak):.0f} dB", fontsize=8)
        ax.set_yticks([])
        ax_i = 1

    freq_ticks = [f for f in (20, 30, 50, 100, 200, 300, 500, 1000, 2000, 3000, 5000, 10000, 20000)
                  if fmin <= f <= min(fmax, sr / 2)]

    def style(ax, title):
        ax.set_yscale("log")
        ax.set_ylim(fmin, min(fmax, sr / 2))
        ax.yaxis.set_major_locator(FixedLocator(freq_ticks))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"))
        ax.yaxis.set_minor_locator(FixedLocator([]))
        ax.set_ylabel("Hz")
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(True, color="white", alpha=0.18, linewidth=0.5)

    for title, grid, edges in grids:
        ax = axes[ax_i]
        ax_i += 1
        x = np.linspace(0, dur, grid.shape[1] + 1)
        im = ax.pcolormesh(x, edges, grid, cmap="magma", vmin=db_range[0], vmax=db_range[1],
                           shading="flat", rasterized=True)
        style(ax, title)
        fig.colorbar(im, ax=ax, pad=0.01, fraction=0.025, label="dBFS")

    if diff and len(grids) == 2:
        ax = axes[ax_i]
        (_, g0, edges), (_, g1, _) = grids
        cols = min(g0.shape[1], g1.shape[1])
        audible = np.maximum(g0[:, :cols], g1[:, :cols]) > db_range[0] + 5
        delta = np.where(audible, g1[:, :cols] - g0[:, :cols], 0.0)
        x = np.linspace(0, dur, cols + 1)
        im = ax.pcolormesh(x, edges, delta, cmap="RdBu_r", vmin=-12, vmax=12, shading="flat", rasterized=True)
        style(ax, f"change: {grids[1][0]} minus {grids[0][0]} (blue = removed, red = added)")
        ax.grid(True, color="black", alpha=0.12, linewidth=0.5)
        fig.colorbar(im, ax=ax, pad=0.01, fraction=0.025, label="dB")
    if waveform:
        # Keep the waveform the same width as the spectrograms (they lose room to colorbars).
        fig.colorbar(im, ax=axes[0], pad=0.01, fraction=0.025).ax.set_visible(False)

    for ax in axes:
        if highlight:
            ax.axvspan(highlight[0] - t_offset, highlight[1] - t_offset, color="#00e5ff", alpha=0.10, linewidth=0)
            for edge in highlight:
                ax.axvline(edge - t_offset, color="#00e5ff", linewidth=0.9, alpha=0.9)
        for seconds, _label in marks or []:
            ax.axvline(seconds - t_offset, color="#7CFC00", linewidth=0.9, linestyle="--", alpha=0.9)
    for seconds, label in marks or []:
        axes[0].annotate(label, (seconds - t_offset, 1.0), xycoords=("data", "axes fraction"),
                         fontsize=7, color="#1b7f1b", ha="center", va="bottom")

    axes[-1].set_xlim(0, dur)
    axes[-1].xaxis.set_major_formatter(_time_formatter(t_offset))
    axes[-1].xaxis.set_major_locator(plt.MaxNLocator(nbins=16, steps=[1, 2, 2.5, 5, 10]))
    axes[-1].set_xlabel("time in the track (m:ss)")
    fig.tight_layout()
    buf = io.BytesIO()
    # JPEG keeps a noisy spectrogram near 200 KB; as PNG it can pass the 1 MB tool-result limit.
    fig.savefig(buf, format="jpeg", pil_kwargs={"quality": 85, "optimize": True})
    plt.close(fig)
    return buf.getvalue()
