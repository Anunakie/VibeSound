"""Signal processing behind the repair tools.

Every processor takes ``segment`` (float64, shape (n, channels)) that includes
real context audio on both sides of the region, plus the region's sample bounds
inside that segment, and returns a processed segment of the same shape. The
project store crossfades the result back into the untouched audio.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg, ndimage, signal

EPS = 1e-12


# ----------------------------------------------------------------------- STFT


def nfft_for(sr: int, win_ms: float) -> int:
    """Power-of-two window length closest to ``win_ms`` at this sample rate."""
    return int(2 ** np.clip(round(np.log2(sr * win_ms * 1e-3)), 6, 16))


def stft(x: np.ndarray, sr: int, win_ms: float = 93.0, overlap: int = 4):
    """Return (S, freqs, times, n_fft, hop); S has shape (channels, bins, frames)."""
    n_fft = nfft_for(sr, win_ms)
    hop = n_fft // overlap
    if len(x) < n_fft:
        x = np.pad(x, ((0, n_fft - len(x)), (0, 0)))
    freqs, times, S = signal.stft(
        x.T, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop,
        boundary="zeros", padded=True, axis=-1,
    )
    return S, freqs, times, n_fft, hop


def istft(S: np.ndarray, sr: int, n_fft: int, hop: int, n: int) -> np.ndarray:
    _, y = signal.istft(S, fs=sr, window="hann", nperseg=n_fft, noverlap=n_fft - hop,
                        boundary=True, time_axis=-1, freq_axis=-2)
    y = y.T
    if len(y) < n:
        y = np.pad(y, ((0, n - len(y)), (0, 0)))
    return y[:n]


def band_weight(freqs: np.ndarray, lo: float, hi: float, shoulder_oct: float = 0.5) -> np.ndarray:
    """1 inside [lo, hi], raised-cosine roll-off over ``shoulder_oct`` octaves outside."""
    lo, hi = max(float(lo), 1.0), max(float(hi), float(lo) + 1.0)
    f = np.maximum(freqs, 1e-3)
    octs_below = np.log2(lo / f)
    octs_above = np.log2(f / hi)
    dist = np.maximum(np.maximum(octs_below, octs_above), 0.0)
    if shoulder_oct <= 0:
        return (dist == 0).astype(float)
    return 0.5 + 0.5 * np.cos(np.pi * np.clip(dist / shoulder_oct, 0.0, 1.0))


def _frames_between(times: np.ndarray, sr: int, a: int, b: int) -> np.ndarray:
    idx = np.flatnonzero((times >= a / sr) & (times <= b / sr))
    if len(idx) == 0:
        idx = np.array([int(np.argmin(np.abs(times - (a + b) / (2 * sr))))])
    return idx


# ------------------------------------------------------------------ tone tools


def reduce_band(segment, sr, ra, rb, *, low_hz=150.0, high_hz=450.0, amount_db=4.0,
                dynamic=True, shoulder_oct=0.5):
    """Turn down a frequency band, optionally only when it is actually built up.

    With ``dynamic`` the cut follows the band's own level: full cut on the
    frames where the band is loudest (sustained, piled-up low notes) and little
    or none on quiet frames, so the body of the instrument is kept.
    """
    S, freqs, times, n_fft, hop = stft(segment, sr, 93.0)
    w = band_weight(freqs, low_hz, high_hz, shoulder_oct)
    scale = np.ones(len(times))
    if dynamic:
        power = np.sum((np.abs(S) ** 2) * w[None, :, None], axis=(0, 1))
        level = 10 * np.log10(power + EPS)
        med = np.median(level[_frames_between(times, sr, ra, rb)])
        scale = np.clip((level - (med - 9.0)) / 12.0, 0.0, 1.0)
        smooth = max(1, int(round(0.12 * sr / hop)))
        scale = ndimage.uniform_filter1d(scale, size=smooth, mode="nearest")
    gain_db = -abs(amount_db) * w[:, None] * scale[None, :]
    S = S * (10.0 ** (gain_db / 20.0))[None]
    return istft(S, sr, n_fft, hop, len(segment))


def _biquad_mag(kind: str, freq: float, gain_db: float, q: float, sr: int, freqs: np.ndarray) -> np.ndarray:
    """Magnitude response of an RBJ cookbook biquad at ``freqs``."""
    freq = float(np.clip(freq, 5.0, sr * 0.49))
    q = max(float(q), 0.05)
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2 * q)
    if kind == "bell":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    elif kind == "low_shelf":
        k = 2 * np.sqrt(A) * alpha
        b = [A * ((A + 1) - (A - 1) * cw + k), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - k)]
        a = [(A + 1) + (A - 1) * cw + k, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - k]
    elif kind == "high_shelf":
        k = 2 * np.sqrt(A) * alpha
        b = [A * ((A + 1) + (A - 1) * cw + k), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - k)]
        a = [(A + 1) - (A - 1) * cw + k, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - k]
    elif kind == "highpass":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "lowpass":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    else:
        raise ValueError(f"unknown EQ band type {kind!r}; use bell, low_shelf, high_shelf, highpass or lowpass")
    _, h = signal.freqz(b, a, worN=freqs, fs=sr)
    return np.abs(h)


def eq_response(bands: list[dict], sr: int, freqs: np.ndarray) -> np.ndarray:
    mag = np.ones(len(freqs))
    for band in bands:
        kind = str(band.get("type", "bell"))
        default_q = 0.707 if kind != "bell" else 1.0
        mag *= _biquad_mag(kind, float(band["freq"]), float(band.get("gain_db", 0.0)),
                           float(band.get("q", default_q)), sr, freqs)
    return mag


def eq(segment, sr, ra, rb, *, bands: list[dict]):
    """Zero-phase parametric EQ (no pre-ringing-prone phase shift at the crossfades)."""
    S, freqs, times, n_fft, hop = stft(segment, sr, 93.0)
    S = S * eq_response(bands, sr, freqs)[None, :, None]
    return istft(S, sr, n_fft, hop, len(segment))


def gain(segment, sr, ra, rb, *, gain_db: float):
    return segment * 10 ** (gain_db / 20.0)


# -------------------------------------------------------------- spectral repair


def spectral_repair(segment, sr, ra, rb, *, low_hz=20.0, high_hz=20000.0, mode="attenuate",
                    strength=1.0, win_ms=46.0, context_ms=80.0, shoulder_oct=0.15):
    """Rebuild a time/frequency box from the audio just before and after it.

    This is the "paint it away" tool. For every frequency bin in the box the
    target level is interpolated between the typical level before and after the
    region. ``attenuate`` only ever turns bins down to that target (safe:
    sustained notes that pass through the box are kept); ``replace`` sets the
    box to the target outright.
    """
    S, freqs, times, n_fft, hop = stft(segment, sr, win_ms, overlap=8)
    mag = np.abs(S)
    half_win = n_fft / (2 * sr)
    t0, t1 = ra / sr, rb / sr
    inside = np.flatnonzero((times >= t0 - half_win) & (times <= t1 + half_win))
    if len(inside) == 0:
        return segment
    ctx = max(context_ms * 1e-3, 4 * hop / sr)
    before = np.flatnonzero((times < t0 - half_win) & (times >= t0 - half_win - ctx))
    after = np.flatnonzero((times > t1 + half_win) & (times <= t1 + half_win + ctx))
    if len(before) == 0 and len(after) == 0:
        return segment
    ref_b = np.median(mag[:, :, before], axis=2) if len(before) else None
    ref_a = np.median(mag[:, :, after], axis=2) if len(after) else None
    if ref_b is None:
        ref_b = ref_a
    if ref_a is None:
        ref_a = ref_b
    pos = (np.arange(len(inside)) + 1.0) / (len(inside) + 1.0)
    log_b, log_a = np.log(ref_b + EPS), np.log(ref_a + EPS)
    target = np.exp(log_b[:, :, None] * (1 - pos)[None, None, :] + log_a[:, :, None] * pos[None, None, :])

    cur = mag[:, :, inside]
    new = np.minimum(cur, target) if mode == "attenuate" else target
    w = band_weight(freqs, low_hz, high_hz, shoulder_oct)[None, :, None] * float(np.clip(strength, 0.0, 1.0))
    out_mag = cur + w * (new - cur)
    S[:, :, inside] *= out_mag / (cur + EPS)
    return istft(S, sr, n_fft, hop, len(segment))


# --------------------------------------------------------------------- denoise


def noise_profile(clip: np.ndarray, sr: int, win_ms: float = 46.0) -> np.ndarray:
    """Average magnitude spectrum of a noise-only clip, shape (channels, bins)."""
    S, *_ = stft(clip, sr, win_ms)
    return np.sqrt(np.mean(np.abs(S) ** 2, axis=2))


def denoise(segment, sr, ra, rb, *, profile: np.ndarray | None = None, amount_db=8.0,
            sensitivity=1.5, win_ms=46.0):
    """Broadband noise reduction by smoothed spectral subtraction.

    ``profile`` is the spectrum of a noise-only stretch (room tone, action
    noise between phrases). Without one, the quietest moments of the segment
    stand in for it, which is less reliable under sustained notes.
    """
    S, freqs, times, n_fft, hop = stft(segment, sr, win_ms)
    power = np.abs(S) ** 2
    if profile is None:
        # 10th percentile of a noise magnitude is ~0.37x its RMS; undo that bias.
        noise_pow = (np.percentile(np.abs(S), 10, axis=2) / 0.37) ** 2
    else:
        if profile.shape[-1] != power.shape[1]:
            raise ValueError("noise profile was measured with a different window size")
        if profile.shape[0] != power.shape[0]:
            profile = np.repeat(np.mean(profile, axis=0, keepdims=True), power.shape[0], axis=0)
        noise_pow = profile ** 2
    smoothed = ndimage.uniform_filter(power, size=(1, 3, 5), mode="nearest")
    wiener = np.clip(1.0 - (sensitivity ** 2) * noise_pow[:, :, None] / (smoothed + EPS), 0.0, 1.0)
    floor = 10 ** (-abs(amount_db) / 20.0)
    g = floor + (1.0 - floor) * np.sqrt(wiener)
    return istft(S * g, sr, n_fft, hop, len(segment))


# ------------------------------------------------- linear prediction, declick


def lpc(x: np.ndarray, order: int) -> np.ndarray:
    """Autocorrelation-method LPC polynomial [1, a1..ap] for a 1-D signal."""
    x = np.asarray(x, dtype=np.float64)
    order = int(min(order, max(2, len(x) // 3)))
    xw = x * np.hanning(len(x))
    r = signal.correlate(xw, xw, mode="full", method="fft")[len(xw) - 1 : len(xw) + order]
    if r[0] <= 0:
        return np.concatenate([[1.0], np.zeros(order)])
    r = r.copy()
    r[0] *= 1.0 + 1e-6  # a touch of white noise keeps the solve well conditioned
    coeffs = linalg.solve_toeplitz(r[:-1], -r[1:])
    return np.concatenate([[1.0], coeffs])


def ar_fill(x: np.ndarray, u0: int, u1: int, order: int | None = None) -> np.ndarray:
    """Fill x[u0:u1] with the least-squares AR interpolation (Janssen / LSAR).

    Fits an all-pole model to the audio around the gap and chooses the missing
    samples that make the whole stretch as predictable as possible. Excellent
    for clicks and gaps of a few milliseconds; energy sags for long gaps.
    """
    x = np.asarray(x, dtype=np.float64)
    gap = u1 - u0
    if gap <= 0:
        return x
    if order is None:
        order = int(np.clip(4 * gap, 512, 1024))  # high order: piano partials are narrow
    ctx = max(8 * order, 4 * gap, 512)
    a0, a1 = max(0, u0 - ctx), min(len(x), u1 + ctx)
    order = int(min(order, (u0 - a0), (a1 - u1)))
    if order < 2:
        out = x.copy()
        left = x[u0 - 1] if u0 > 0 else (x[u1] if u1 < len(x) else 0.0)
        right = x[u1] if u1 < len(x) else left
        out[u0:u1] = np.linspace(left, right, gap + 2)[1:-1]
        return out

    known = np.concatenate([x[a0:u0], x[u1:a1]])
    a = lpc(known, order)
    order = len(a) - 1

    z = x[a0:a1].copy()
    g0 = u0 - a0
    z[g0 : g0 + gap] = 0.0
    e = signal.lfilter(a, [1.0], z)
    d = -e[g0 : g0 + gap + order]
    rhs = np.correlate(d, a, mode="valid")
    acf = np.correlate(a, a, mode="full")[order:]
    bands = min(order, gap - 1)
    ab = np.zeros((bands + 1, gap))
    for k in range(bands + 1):
        ab[k, : gap - k] = acf[k]
    ab[0] *= 1.0 + 1e-9
    filled = linalg.solveh_banded(ab, rhs, lower=True)
    out = x.copy()
    out[u0:u1] = filled
    return out


def _covariance_lpc(x: np.ndarray, order: int) -> np.ndarray:
    """Least-squares predictor coefficients c so that x[n] ~ sum(c[k] * x[n-1-k])."""
    rows = np.lib.stride_tricks.sliding_window_view(x, order + 1)
    A, y = rows[:, :-1][:, ::-1], rows[:, -1]
    reg = 1e-6 * np.mean(x ** 2) * len(x)
    return linalg.solve(A.T @ A + reg * np.eye(order), A.T @ y, assume_a="pos")


def _extrapolate(x: np.ndarray, n: int, order: int) -> np.ndarray:
    """Continue 1-D signal x for n samples with a covariance-method linear predictor."""
    order = int(min(order, len(x) // 3))
    if order < 2 or np.max(np.abs(x)) < 1e-9:
        return np.zeros(n)
    peak = np.max(np.abs(x)) * 2.0

    def run(coeffs: np.ndarray) -> np.ndarray:
        # Extrapolating = running the all-pole filter with zero input from its last state.
        a = np.concatenate([[1.0], -coeffs])
        zi = signal.lfiltic([1.0], a, x[::-1][:order])
        out, _ = signal.lfilter([1.0], a, np.zeros(n), zi=zi)
        return out

    out = run(_covariance_lpc(x, order))
    if not np.all(np.isfinite(out)) or np.max(np.abs(out)) > peak:
        out = run(-lpc(x, order)[1:])  # autocorrelation method is always stable
    return out


def ar_bridge(x: np.ndarray, u0: int, u1: int, order: int | None = None) -> np.ndarray:
    """Fill a longer gap by predicting forward from the left and backward from the right."""
    x = np.asarray(x, dtype=np.float64)
    gap = u1 - u0
    if order is None:
        order = 1024
    ctx = 4 * order
    left = x[max(0, u0 - ctx) : u0]
    right = x[u1 : min(len(x), u1 + ctx)]
    fwd = _extrapolate(left, gap, order) if len(left) > 8 else np.zeros(gap)
    bwd = _extrapolate(right[::-1], gap, order)[::-1] if len(right) > 8 else np.zeros(gap)
    if len(left) <= 8:
        mix = bwd
    elif len(right) <= 8:
        mix = fwd
    else:
        t = (np.arange(gap) + 0.5) / gap
        w = 0.5 - 0.5 * np.cos(np.pi * t)
        mix = fwd * (1 - w) + bwd * w
    out = x.copy()
    out[u0:u1] = mix
    return out


def find_clicks(mono: np.ndarray, sr: int, *, threshold=10.0, max_ms=3.0, order=32) -> list[dict]:
    """Find short impulsive clicks as outliers of the linear-prediction residual.

    Note attacks also excite the residual, so a candidate is dropped when the
    level stays up afterwards (something started sounding) or when the burst is
    longer than ``max_ms``.
    """
    mono = np.asarray(mono, dtype=np.float64)
    n = len(mono)
    block = max(4096, nfft_for(sr, 180.0))
    hits: list[dict] = []
    flagged = np.zeros(n, dtype=bool)
    score = np.zeros(n)
    resid_abs = np.zeros(n)
    for b0 in range(0, n, block):
        b1 = min(n, b0 + block)
        s0 = max(0, b0 - order * 4)
        chunk = mono[s0:b1]
        if len(chunk) < order * 4 or np.max(np.abs(chunk)) < 1e-7:
            continue
        resid = signal.lfilter(lpc(chunk, order), [1.0], chunk)[b0 - s0 :]
        sigma = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-9
        z = np.abs(resid) / sigma
        resid_abs[b0:b1] = np.abs(resid)
        score[b0:b1] = z
        flagged[b0:b1] = z > threshold
    if not flagged.any():
        return hits
    merge = max(1, int(1e-3 * sr))
    flagged = ndimage.binary_closing(flagged, structure=np.ones(merge))
    labels, count = ndimage.label(flagged)
    win = int(0.02 * sr)
    guard = int(0.002 * sr)
    near = int(0.008 * sr)
    for sl in ndimage.find_objects(labels)[:5000]:
        s, e = sl[0].start, sl[0].stop
        if (e - s) / sr * 1e3 > max_ms:
            continue
        pre = mono[max(0, s - guard - win) : max(0, s - guard)]
        post = mono[min(n, e + guard) : min(n, e + guard + win)]
        if len(pre) < 16 or len(post) < 16:
            continue
        pre_db = 10 * np.log10(np.mean(pre ** 2) + EPS)
        post_db = 10 * np.log10(np.mean(post ** 2) + EPS)
        if post_db - pre_db > 3.0:
            continue  # level jumped: this is a note starting, not a click
        # A hammer strike leaves the residual noisy for several ms; a click is over at once.
        tail = resid_abs[min(n, e + guard) : min(n, e + guard + near)]
        head = resid_abs[max(0, s - guard - near) : max(0, s - guard)]
        peak = float(np.max(resid_abs[s:e]))
        local = 1.4826 * max(np.median(tail) if len(tail) else 0.0, np.median(head) if len(head) else 0.0)
        if peak < threshold * local:
            continue
        hits.append({
            "start": int(s), "end": int(e),
            "time": (s + e) / (2 * sr),
            "width_ms": round((e - s) / sr * 1e3, 2),
            "strength": round(float(np.max(score[s:e])), 1),
        })
    return hits


def declick(segment, sr, ra, rb, *, clicks: list[tuple[int, int]], pad_ms=0.1):
    """Replace each (start, end) sample span, given in segment coordinates, by AR interpolation."""
    out = segment.copy()
    pad = max(2, int(pad_ms * 1e-3 * sr))
    for s, e in clicks:
        u0, u1 = max(1, s - pad), min(len(out) - 1, e + pad)
        for ch in range(out.shape[1]):
            out[:, ch] = ar_fill(out[:, ch], u0, u1)
    return out
