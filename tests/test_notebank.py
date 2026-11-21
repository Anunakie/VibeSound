from __future__ import annotations

import numpy as np
import soundfile as sf

from vibesound import notebank as nb

from conftest import SR, note, thump


def _sample_file(tmp_path, name, pitches, amp, seed=0):
    """A 'one note at a time' recording: each pitch rings 2.5 s, then 0.6 s of quiet."""
    rng = np.random.default_rng(seed)
    parts = [rng.normal(0, 1e-4, int(0.8 * SR))]
    for i, p in enumerate(pitches):
        tone = note(nb.midi_to_hz(p), 2.5, amp=amp, seed=seed * 100 + i)
        parts += [tone + rng.normal(0, 1e-4, len(tone)), rng.normal(0, 1e-4, int(0.6 * SR))]
    mono = np.concatenate(parts)
    path = tmp_path / name
    sf.write(str(path), np.stack([mono, 0.9 * mono], axis=1), SR, subtype="FLOAT")
    return str(path)


PITCHES = [39, 46, 51, 55, 58, 68, 72]  # D#2 A#2 D#3 G3 A#3 G#4 C5


def test_scan_finds_every_note_with_the_right_key(tmp_path):
    path = _sample_file(tmp_path, "soft.wav", PITCHES, amp=0.05)
    found = nb.scan_samples(path)
    assert [s.pitch for s in found] == PITCHES
    assert all(not s.pedal for s in found)
    assert nb.scan_samples(_sample_file(tmp_path, "soft pedal.wav", [60], 0.05))[0].pedal


def test_bank_picks_closest_loudness_and_reports_coverage(tmp_path):
    bank = nb.NoteBank.from_files([_sample_file(tmp_path, "soft.wav", PITCHES, 0.04),
                                   _sample_file(tmp_path, "firm.wav", PITCHES, 0.25, seed=1)])
    cov = bank.coverage()
    assert cov["samples"] == 14 and cov["keys"] == 7 and cov["lowest"] == "D#2"
    lo, hi = bank.levels(39)
    assert hi - lo > 10
    assert "soft" in bank.pick(39, lo + 1)[0].path and "firm" in bank.pick(39, hi - 1)[0].path
    smp, shift = bank.pick(40, lo)  # E2 was never recorded: borrow D#2, shifted up one
    assert smp.pitch == 39 and shift == 1
    bank.save(str(tmp_path / "bank.json"))
    assert len(nb.NoteBank.load(str(tmp_path / "bank.json")).samples) == 14


def test_replace_region_removes_a_boom_and_keeps_the_notes_in_time(tmp_path):
    bank = nb.NoteBank.from_files([_sample_file(tmp_path, "soft.wav", PITCHES, 0.05),
                                   _sample_file(tmp_path, "medium.wav", PITCHES, 0.12, seed=1)])
    # A "performance" played on different note recordings than the bank holds.
    notes = [nb.Note(p, 1.0 + 0.4 * i, 1.0 + 0.4 * i + 0.8, 50) for i, p in enumerate([39, 46, 51, 55, 58])]
    n = int(5.0 * SR)
    mono = np.zeros(n)
    for i, nt in enumerate(notes):
        tone = note(nb.midi_to_hz(nt.pitch), 0.9, amp=0.1, seed=900 + i)
        a = int(nt.start * SR)
        mono[a: a + len(tone)] += tone
    clean = np.stack([mono, 0.9 * mono], axis=1)
    damaged = clean.copy()
    boom = thump(amp=0.5)
    a = int(1.0 * SR)
    damaged[a: a + len(boom)] += boom[:, None]

    fixed, info = nb.replace_region(damaged, SR, notes, bank, 0.98, 2.2)
    assert not info["missing"]

    def low(x, t0, t1):
        seg = x[int(t0 * SR): int(t1 * SR), 0]
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        f = np.fft.rfftfreq(len(seg), 1 / SR)
        return 10 * np.log10(np.sum(spec[(f > 40) & (f < 70)] ** 2) + 1e-12)

    # The 55 Hz boom is gone: the low band is back to what the clean performance had.
    assert low(damaged, 1.0, 1.2) > low(clean, 1.0, 1.2) + 5
    assert abs(low(fixed, 1.0, 1.2) - low(clean, 1.0, 1.2)) < 4
    # Outside the region nothing changed.
    assert np.array_equal(fixed[: int(0.9 * SR)], damaged[: int(0.9 * SR)])
    assert np.array_equal(fixed[int(2.3 * SR):], damaged[int(2.3 * SR):])

    # Each note attack is where it was played (within 50 ms).
    def attacks(x):
        env = np.abs(x[:, 0])
        k = int(0.01 * SR)
        e = np.convolve(env, np.ones(k) / k, mode="same")
        d = e[k:] - e[:-k]
        return d

    d = attacks(fixed)
    for nt in notes[:3]:
        i = int(nt.start * SR)
        window = d[i - int(0.05 * SR): i + int(0.05 * SR)]
        assert window.max() > 0.3 * d[int(0.98 * SR): int(2.2 * SR)].max(), nb.note_name(nt.pitch)
    assert all(r["shift"] == 0 for r in info["notes"])
