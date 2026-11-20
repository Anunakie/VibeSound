"""Synthetic felt-piano-ish test material with known, planted defects."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

SR = 44100


def note(freq: float, dur: float, sr: int = SR, amp: float = 0.2, seed: int = 0) -> np.ndarray:
    """A decaying, slightly inharmonic tone with a soft attack: a crude felted piano note."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * sr)) / sr
    out = np.zeros_like(t)
    for k in range(1, 13):
        f = freq * k * np.sqrt(1 + 0.0004 * k * k)
        if f > sr / 2.2:
            break
        out += (1.0 / k ** 1.6) * np.exp(-t * (0.9 + 0.35 * k)) * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28))
    attack = 0.5 - 0.5 * np.cos(np.pi * np.minimum(1.0, t / 0.012))
    # Hammer-on-felt noise: a short dull burst at the strike, as on a real piano.
    burst = rng.normal(0, 1, int(0.02 * sr)) * np.hanning(int(0.02 * sr))
    burst = np.convolve(burst, np.ones(12) / 12, mode="same")
    out = out * attack
    out[: len(burst)] += 0.08 * burst
    return amp * out


def piano_track(dur: float = 12.0, sr: int = SR, seed: int = 1) -> np.ndarray:
    """Stereo track: a note every 0.5 s from a small scale, plus a low noise floor."""
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    mono = np.zeros(n)
    pitches = [110.0, 146.83, 164.81, 220.0, 261.63, 329.63, 392.0, 523.25]
    for i, start in enumerate(np.arange(0.25, dur - 1.0, 0.5)):
        tone = note(pitches[(i * 3) % len(pitches)], 2.5, sr, seed=i)
        a = int(start * sr)
        b = min(n, a + len(tone))
        mono[a:b] += tone[: b - a]
    stereo = np.stack([mono, 0.9 * np.roll(mono, 7)], axis=1)
    stereo += rng.normal(0, 10 ** (-78 / 20), stereo.shape)
    return stereo


def thump(sr: int = SR, amp: float = 0.25) -> np.ndarray:
    t = np.arange(int(0.12 * sr)) / sr
    return amp * np.exp(-t * 45) * np.sin(2 * np.pi * 55 * t) * np.minimum(1.0, t / 0.004)


@pytest.fixture
def clean() -> np.ndarray:
    return piano_track()


@pytest.fixture
def write_wav(tmp_path):
    def _write(data: np.ndarray, name: str = "take.wav", subtype: str = "PCM_24") -> str:
        path = tmp_path / name
        sf.write(str(path), data, SR, subtype=subtype)
        return str(path)

    return _write
