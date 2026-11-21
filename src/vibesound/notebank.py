"""Note replacement: rebuild a passage from clean single-note recordings.

The idea (Daniel's): a damaged passage still tells you *what was played*. Transcribe
it to notes, then play those notes back with clean recordings of the same piano and
set the result where the damage was. Nothing is guessed.

Three pieces live here:

* :func:`scan_samples` reads long "one note at a time" recordings and finds every
  note in them: where it starts and ends, which key it is, and how loud it is.
* :class:`NoteBank` holds those notes and picks the best one for a requested key,
  loudness and pedal state.
* :func:`render` plays a list of transcribed notes from the bank, and
  :func:`replace_region` sets a rendered passage into a recording.

Transcription itself is outside this module (any tool that yields pitch, start, end
and velocity works; see docs/note-replacement.md).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


@dataclass
class Note:
    """One transcribed note. Times in seconds; velocity 1-127 (MIDI convention)."""

    pitch: int
    start: float
    end: float
    velocity: int = 64


@dataclass
class Sample:
    """One recorded note inside a sample file."""

    path: str
    start: int  # sample index of the attack
    end: int  # sample index where it has faded (or the next note begins)
    pitch: int  # MIDI note number
    level_db: float  # RMS of the first 250 ms, dBFS
    pedal: bool = False
    tag: str = ""


def hz_to_midi(f: float) -> float:
    return 69.0 + 12.0 * np.log2(f / 440.0)


def midi_to_hz(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69.0) / 12.0)


def note_name(m: int) -> str:
    return ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][m % 12] + str(m // 12 - 1)


# ----------------------------------------------------------------- scanning


def _envelope_db(mono: np.ndarray, sr: int, hop: int) -> np.ndarray:
    win = 2 * hop
    pad = np.pad(mono.astype(np.float64) ** 2, (0, win))
    c = np.cumsum(np.insert(pad, 0, 0.0))
    idx = np.arange(0, len(mono), hop)
    return 10 * np.log10((c[idx + win] - c[idx]) / win + 1e-14)


def detect_pitch(mono: np.ndarray, sr: int, fmin: float = 25.0, fmax: float = 4400.0) -> float | None:
    """Fundamental of a single piano note, in Hz, by harmonic summation.

    Looks at up to the first second after the attack. Scores each candidate
    fundamental by the energy at its first harmonics, so a weak or missing
    fundamental (common in the lowest octave) is still found.
    """
    seg = mono[int(0.03 * sr): int(1.0 * sr)]
    if len(seg) < int(0.15 * sr):
        seg = mono
    n = 1 << int(np.ceil(np.log2(max(len(seg), sr))) + 1)
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n))
    spec[: int(20 * n / sr)] = 0.0
    if spec.max() <= 0:
        return None
    spec = spec / spec.max()
    cands = midi_to_hz(np.arange(hz_to_midi(fmin), hz_to_midi(fmax), 0.1))
    best, best_f = -1.0, None
    for f0 in cands:
        score, present = 0.0, 0
        for h in range(1, 9):
            fh = f0 * h
            if fh >= sr / 2:
                break
            k = int(round(fh * n / sr))
            w = max(1, int(0.015 * fh * n / sr))
            v = spec[max(0, k - w): k + w + 1].max()
            score += v / h ** 0.5
            present += v > 0.02
        # Penalise candidates an octave (or more) too low: every other harmonic is missing.
        odd = sum(spec[int(round(f0 * h * n / sr))] for h in (1, 3, 5) if f0 * h < sr / 2)
        even = sum(spec[int(round(f0 * h * n / sr))] for h in (2, 4, 6) if f0 * h < sr / 2)
        if even > 0 and odd < 0.08 * even:
            score *= 0.5
        if present >= 2 and score > best * 1.001:
            best, best_f = score, float(f0)
    if best_f is None:
        return None
    # Refine on the strongest low harmonic.
    for h in (1, 2):
        k0 = best_f * h * n / sr
        lo, hi = int(k0 * 0.97), int(k0 * 1.03) + 1
        k = lo + int(np.argmax(spec[lo:hi]))
        if spec[k] > 0.05:
            return float(k * sr / n / h)
    return best_f


def scan_samples(path: str, pedal: bool | None = None, tag: str = "", rise_db: float = 14.0,
                 min_gap_s: float = 0.35) -> list[Sample]:
    """Find every note in a recording of single notes played one after another.

    ``pedal`` defaults to whether "pedal" appears in the file name.
    """
    audio, sr = sf.read(path, dtype="float64", always_2d=True)
    mono = audio.mean(axis=1)
    hop = int(0.005 * sr)
    env = _envelope_db(mono, sr, hop)
    floor = np.percentile(env, 10)
    loud = env > floor + rise_db
    # An attack: the level jumps well above what it was 60 ms earlier, from a loud-enough frame.
    lag = max(1, int(0.06 * sr / hop))
    jump = np.zeros_like(env)
    jump[lag:] = env[lag:] - env[:-lag]
    cand = np.where(loud & (jump > 9.0))[0]
    onsets: list[int] = []
    for i in cand:
        if not onsets or (i - onsets[-1]) * hop / sr > min_gap_s:
            onsets.append(int(i))
    if pedal is None:
        pedal = "pedal" in Path(path).stem.lower() and "no pedal" not in Path(path).stem.lower()
    out: list[Sample] = []
    for j, i in enumerate(onsets):
        # Back up to the true start of the attack: the first moment the signal leaves the
        # noise floor, found at 1 ms resolution, minus a 2 ms safety margin.
        a0 = max(0, (i - lag) * hop)
        fine = max(1, int(0.001 * sr))
        seg = mono[a0: (i + 2) * hop]
        fenv = _envelope_db(seg, sr, fine)
        up = np.where(fenv > floor + 8.0)[0]
        a = a0 + (int(up[0]) * fine if len(up) else 0)
        a = max(0, a - int(0.002 * sr))
        nxt = onsets[j + 1] * hop - lag * hop if j + 1 < len(onsets) else len(mono)
        quiet = np.where(env[i: nxt // hop] < floor + 3.0)[0]
        b = min(nxt, (i + int(quiet[0])) * hop) if len(quiet) else nxt
        if (b - a) / sr < 0.15:
            continue
        f0 = detect_pitch(mono[a:b], sr)
        if f0 is None:
            continue
        head = mono[a: a + int(0.25 * sr)]
        out.append(Sample(path=str(path), start=int(a), end=int(b), pitch=int(round(hz_to_midi(f0))),
                          level_db=float(10 * np.log10(np.mean(head ** 2) + 1e-14)), pedal=bool(pedal), tag=tag))
    return out


# --------------------------------------------------------------------- bank


class NoteBank:
    def __init__(self, samples: list[Sample]):
        self.samples = list(samples)
        self._cache: dict[str, tuple[np.ndarray, int]] = {}

    # -- persistence
    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps([asdict(s) for s in self.samples], indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "NoteBank":
        return cls([Sample(**d) for d in json.loads(Path(path).read_text(encoding="utf-8"))])

    @classmethod
    def from_files(cls, paths: list[str]) -> "NoteBank":
        samples: list[Sample] = []
        for p in paths:
            samples += scan_samples(p, tag=Path(p).stem)
        return cls(samples)

    # -- summary
    def coverage(self) -> dict:
        pitches = sorted({s.pitch for s in self.samples})
        missing = [note_name(p) for p in range(21, 109) if p not in pitches]
        return {"samples": len(self.samples), "keys": len(pitches),
                "lowest": note_name(pitches[0]) if pitches else None,
                "highest": note_name(pitches[-1]) if pitches else None,
                "missing_keys": missing, "with_pedal": sum(s.pedal for s in self.samples)}

    def levels(self, pitch: int) -> list[float]:
        return sorted(s.level_db for s in self.samples if s.pitch == pitch)

    # -- selection
    def pick(self, pitch: int, level_db: float | None = None, pedal: bool = False, max_shift: int = 2
             ) -> tuple[Sample, int] | None:
        """Best sample for a key. Returns (sample, semitones to shift it by), or None.

        Prefers the exact key, then the right pedal state, then the closest loudness.
        Falls back to a neighbouring key (shifted by resampling) only if the key is missing.
        """
        best, best_cost = None, np.inf
        for s in self.samples:
            shift = pitch - s.pitch
            if abs(shift) > max_shift:
                continue
            cost = 100.0 * abs(shift) + (0.0 if s.pedal == pedal else 6.0)
            if level_db is not None:
                cost += abs(s.level_db - level_db)
            if cost < best_cost:
                best, best_cost = (s, shift), cost
        return best

    def audio(self, s: Sample) -> tuple[np.ndarray, int]:
        if s.path not in self._cache:
            self._cache[s.path] = sf.read(s.path, dtype="float64", always_2d=True)
        data, sr = self._cache[s.path]
        return data[s.start: s.end], sr


# ----------------------------------------------------------------- rendering


def velocity_to_level(velocity: int, soft_db: float, firm_db: float, v_soft: int = 25, v_firm: int = 90) -> float:
    """Map a transcribed velocity onto the bank's loudness range for that key."""
    x = np.clip((velocity - v_soft) / float(v_firm - v_soft), 0.0, 1.0)
    return float(soft_db + x * (firm_db - soft_db))


def render(notes: list[Note], bank: NoteBank, sr: int, t0: float, t1: float, channels: int = 2,
           pedal_spans: list[tuple[float, float]] | None = None, release_ms: float = 120.0) -> tuple[np.ndarray, list[dict]]:
    """Play ``notes`` from the bank into a buffer covering [t0, t1] seconds.

    Notes that begin before ``t0`` but are still sounding are included, so their
    tails are present at the start of the buffer. A note is cut with a short
    release fade at its end unless the sustain pedal is down there.
    Returns (audio, report) where the report lists which sample played each note.
    """
    n = int(round((t1 - t0) * sr))
    out = np.zeros((n, channels))
    report: list[dict] = []
    pedal_spans = pedal_spans or []

    def pedal_at(t: float) -> bool:
        return any(a <= t <= b for a, b in pedal_spans)

    def sounding_until(note: Note) -> float:
        end = note.end
        for a, b in pedal_spans:
            if a <= end <= b:
                end = b
        return end

    for note in sorted(notes, key=lambda x: x.start):
        stop = sounding_until(note)
        if stop <= t0 or note.start >= t1:
            continue
        lv = bank.levels(note.pitch)
        target = velocity_to_level(note.velocity, lv[0], lv[-1]) if lv else None
        choice = bank.pick(note.pitch, target, pedal_at(note.start))
        if choice is None:
            report.append({"note": note_name(note.pitch), "start": note.start, "sample": None})
            continue
        smp, shift = choice
        data, ssr = bank.audio(smp)
        ratio = (sr / ssr) / (2.0 ** (shift / 12.0))
        if abs(ratio - 1.0) > 1e-9:
            data = signal.resample_poly(data, int(round(ratio * 10000)), 10000, axis=0)
        if data.shape[1] != channels:
            data = np.repeat(data.mean(axis=1, keepdims=True), channels, axis=1)
        gain = 1.0 if target is None else 10 ** (np.clip(target - smp.level_db, -6.0, 6.0) / 20.0)
        data = data * gain
        length = min(len(data), int(round((stop - note.start) * sr)) + int(release_ms * 1e-3 * sr))
        data = data[:length].copy()
        rel = min(length, int(release_ms * 1e-3 * sr))
        if rel > 1:
            data[-rel:] *= np.cos(np.linspace(0, np.pi / 2, rel))[:, None] ** 2
        a = int(round((note.start - t0) * sr))
        src0 = max(0, -a)
        a = max(0, a)
        b = min(n, a + length - src0)
        if b > a:
            out[a:b] += data[src0: src0 + (b - a)]
        report.append({"note": note_name(note.pitch), "start": round(note.start, 3), "velocity": note.velocity,
                       "sample": Path(smp.path).name, "sample_level_db": round(smp.level_db, 1),
                       "shift": shift, "gain_db": round(float(20 * np.log10(gain)), 1), "pedal": smp.pedal})
    return out, report


def match_gain_db(original: np.ndarray, rendered: np.ndarray, sr: int, lo: float = 200.0, hi: float = 4000.0) -> float:
    """Gain that brings ``rendered`` to the original's level in a band the damage does not dominate."""
    sos = signal.butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    ro = np.sqrt(np.mean(signal.sosfiltfilt(sos, original, axis=0) ** 2) + 1e-14)
    rr = np.sqrt(np.mean(signal.sosfiltfilt(sos, rendered, axis=0) ** 2) + 1e-14)
    return float(np.clip(20 * np.log10(ro / rr), -12.0, 12.0))


def replace_region(audio: np.ndarray, sr: int, notes: list[Note], bank: NoteBank, start: float, end: float,
                   pedal_spans: list[tuple[float, float]] | None = None, xfade_ms: float = 30.0,
                   auto_gain: bool = True) -> tuple[np.ndarray, dict]:
    """Rebuild [start, end] seconds of ``audio`` from the bank and crossfade it in.

    Choose ``start`` just before a note attack where possible: an attack hides the seam.
    The crossfades are placed just outside the region.
    """
    # The crossfades sit just outside [start, end], so the whole region is purely rebuilt.
    pad = xfade_ms * 1e-3
    r0, r1 = max(0.0, start - pad), min(len(audio) / sr, end + pad)
    rendered, report = render(notes, bank, sr, r0, r1, audio.shape[1], pedal_spans)
    a = int(round(r0 * sr))
    b = min(a + len(rendered), len(audio))
    rendered = rendered[: b - a]
    gain_db = match_gain_db(audio[a:b], rendered, sr) if auto_gain and np.any(rendered) else 0.0
    rendered = rendered * 10 ** (gain_db / 20.0)
    xf = min(int(pad * sr), (b - a) // 2)
    w = np.ones(b - a)
    if xf > 1:
        ramp = np.sin(np.linspace(0, np.pi / 2, xf)) ** 2
        w[:xf], w[-xf:] = ramp, ramp[::-1]
    out = audio.copy()
    out[a:b] = audio[a:b] * (1 - w)[:, None] + rendered * w[:, None]
    missing = [r["note"] for r in report if r["sample"] is None]
    return out, {"notes": report, "gain_db": round(gain_db, 1), "missing": missing}
