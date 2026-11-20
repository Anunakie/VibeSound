from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
from scipy import signal

from vibesound import analysis, dsp, inpaint
from vibesound.engine import Session
from vibesound.store import Project
from vibesound.timeutil import fmt_time, parse_time

from conftest import SR, note, thump


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)) + 1e-12)


def band_db(x, lo, hi, sr=SR):
    sos = signal.butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    return rms_db(signal.sosfiltfilt(sos, np.asarray(x, dtype=np.float64), axis=0))


# ------------------------------------------------------------------ basics


def test_time_parsing():
    assert parse_time("2:15") == 135.0
    assert parse_time("2:15.5") == 135.5
    assert parse_time("1:02:03") == 3723.0
    assert parse_time(12.5) == 12.5
    assert parse_time("12.5s") == 12.5
    assert fmt_time(135.5) == "2:15.500"
    with pytest.raises(ValueError):
        parse_time("soon")


def test_versions_are_non_destructive(clean, write_wav, tmp_path):
    path = write_wav(clean)
    source_bytes = open(path, "rb").read()
    s = Session()
    info = s.open_project(path)
    assert info["current_version"] == 0 and info["sample_rate"] == SR

    r1 = s.gain("0:03", "0:04", -6)
    r2 = s.gain("0:06", "0:07", -3)
    assert (r1["new_version"], r2["new_version"]) == (1, 2)
    assert open(path, "rb").read() == source_bytes, "source file must never change"

    p = s.project
    v0, v2 = p.audio(0), p.audio(2)
    # Untouched audio is bit-identical; only the two regions (plus fades) differ.
    changed = np.flatnonzero(np.any(v0 != v2, axis=1)) / SR
    assert changed.min() > 2.8 and changed.max() < 7.2
    assert not np.any((changed > 4.2) & (changed < 5.8))
    assert abs((rms_db(v2[int(3.3 * SR):int(3.7 * SR)]) - rms_db(v0[int(3.3 * SR):int(3.7 * SR)])) + 6) < 0.05

    s.undo()
    assert p.head == 1 and np.array_equal(p.audio()[int(6.2 * SR):int(6.8 * SR)], v0[int(6.2 * SR):int(6.8 * SR)])
    r3 = s.gain("0:08", "0:09", 2)  # branch from v1; v2 stays available
    assert r3["based_on"] == 1 and [v["id"] for v in p.chain()] == [0, 1, 3]
    s.checkout(2)
    assert np.array_equal(p.audio(), v2)

    # Reopening resumes, and exporting the original version reproduces the source samples exactly.
    s2 = Session()
    assert s2.open_project(path)["current_version"] == 2
    out = s2.export(str(tmp_path / "orig_out.wav"), version=0)
    a, _ = sf.read(path, dtype="int32")
    b, _ = sf.read(out["written"], dtype="int32")
    assert out["bit_format"] == "PCM_24" and np.array_equal(a, b)
    with pytest.raises(ValueError):
        s2.export(path)


def test_edit_edges_are_seamless(clean, write_wav):
    """A drastic edit must not put a step (click) at its boundaries."""
    s = Session()
    s.open_project(write_wav(clean))
    s.gain("0:05", "0:06", -40)
    y = s.project.audio()[:, 0].astype(np.float64)
    x = s.project.audio(0)[:, 0].astype(np.float64)
    for edge in (5.0, 6.0):
        i = int(edge * SR)
        jump = np.max(np.abs(np.diff(y[i - 2000:i + 2000])))
        assert jump <= np.max(np.abs(np.diff(x[i - 2000:i + 2000]))) * 1.05


# ------------------------------------------------------------------- fixes


def test_reduce_mud_hits_only_the_band_and_region(clean, write_wav):
    muddy = clean.copy()
    a, b = int(4 * SR), int(8 * SR)
    sos = signal.butter(2, [160, 420], btype="band", fs=SR, output="sos")
    muddy[a:b] += 1.2 * signal.sosfiltfilt(sos, clean[a:b], axis=0)  # plant ~+6 dB of low-mid

    s = Session()
    s.open_project(write_wav(muddy))
    found = s.analyze("0:04.5", "0:07.5", reference_start="0:00.5", reference_end="0:03.5")
    balance = found["band_balance_vs_reference_db"]
    assert max(balance, key=balance.get) in ("low 120-250", "low-mid 250-500")

    report = s.reduce_mud("0:04", "0:08", amount_db=6, dynamic=False)
    before, after = s.project.audio(0), s.project.audio()
    inner = slice(int(4.5 * SR), int(7.5 * SR))
    assert -6.6 < band_db(after[inner], 200, 380) - band_db(before[inner], 200, 380) < -5.2
    assert abs(band_db(after[inner], 1500, 6000) - band_db(before[inner], 1500, 6000)) < 0.3
    assert np.array_equal(after[: int(3.5 * SR)], before[: int(3.5 * SR)])
    assert np.array_equal(after[int(8.5 * SR):], before[int(8.5 * SR):])
    assert min(report["band_change_db"].values()) < -3

    s.undo()
    s.reduce_mud("0:04", "0:08", amount_db=6, dynamic=True)
    dyn = s.project.audio()
    change = band_db(dyn[inner], 200, 380) - band_db(before[inner], 200, 380)
    assert -6.5 < change < -1.5


def test_thump_is_found_and_removed_without_touching_notes(clean, write_wav):
    dirty = clean.copy()
    t0 = 5.13
    bump = thump()
    i = int(t0 * SR)
    dirty[i:i + len(bump)] += bump[:, None]

    s = Session()
    s.open_project(write_wav(dirty))
    found = s.find_problems("0:02", "0:10")
    times = [t["seconds"] for t in found["thumps"]]
    assert any(abs(t - t0) < 0.04 for t in times), found
    assert len(times) <= 2, f"note attacks misread as thumps: {found['thumps']}"

    s.remove_thump(min(times, key=lambda t: abs(t - t0)))
    fixed = s.project.audio()
    win = slice(i, i + int(0.1 * SR))
    err_before = rms_db(dirty[win] - clean[win])
    err_after = rms_db(fixed[win] - clean[win])
    assert err_after < err_before - 10, (err_before, err_after)
    assert abs(band_db(fixed[win], 400, 5000) - band_db(clean[win], 400, 5000)) < 0.5


def test_clicks_are_found_and_repaired(clean, write_wav):
    dirty = clean.copy()
    spots = [3.2101, 5.6307, 7.4444]
    for k, t in enumerate(spots):
        i = int(t * SR)
        dirty[i:i + 5] += np.array([0.25, -0.3, 0.2, -0.1, 0.05])[:, None] * (1 if k % 2 else -1)

    s = Session()
    s.open_project(write_wav(dirty))
    found = s.find_problems("0:01", "0:11")
    got = [c["seconds"] for c in found["clicks"]]
    for t in spots:
        assert any(abs(g - t) < 0.002 for g in got), (t, got)
    assert len(got) <= len(spots) + 2

    report = s.remove_clicks("0:01", "0:11")
    assert report["clicks_repaired"] >= 3
    fixed = s.project.audio()
    for t in spots:
        i = int(t * SR)
        win = slice(i - 20, i + 40)
        assert rms_db(fixed[win] - clean[win]) < rms_db(dirty[win] - clean[win]) - 25


def test_spectral_repair_removes_a_squeak_but_keeps_the_music(clean, write_wav):
    dirty = clean.copy()
    i, n = int(6.1 * SR), int(0.15 * SR)
    t = np.arange(n) / SR
    squeak = 0.03 * np.sin(2 * np.pi * (3000 * t + 2000 * t * t)) * np.hanning(n)
    dirty[i:i + n] += squeak[:, None]

    s = Session()
    s.open_project(write_wav(dirty))
    s.spectral_repair(6.08, 6.27, low_hz=2700, high_hz=3900)
    fixed = s.project.audio()
    win = slice(i, i + n)
    assert band_db(fixed[win] - clean[win], 2800, 3800) < band_db(dirty[win] - clean[win], 2800, 3800) - 12
    assert abs(band_db(fixed[win], 80, 1500) - band_db(clean[win], 80, 1500)) < 0.2


def test_denoise_lowers_the_floor_not_the_notes(write_wav):
    rng = np.random.default_rng(3)
    from conftest import piano_track
    track = piano_track(seed=5)
    track[: int(0.24 * SR)] = 0
    noisy = track + rng.normal(0, 10 ** (-52 / 20), track.shape)
    s = Session()
    s.open_project(write_wav(noisy, subtype="FLOAT"))
    s.denoise("0:00", "0:11", amount_db=12, noise_start=0.0, noise_end=0.22)
    out = s.project.audio()
    assert rms_db(out[int(0.05 * SR):int(0.2 * SR)]) < rms_db(noisy[int(0.05 * SR):int(0.2 * SR)]) - 8
    loud = slice(int(2.26 * SR), int(2.5 * SR))
    assert abs(band_db(out[loud], 100, 1500) - band_db(noisy[loud], 100, 1500)) < 0.5
    assert rms_db(out[int(2 * SR):int(9 * SR)] - track[int(2 * SR):int(9 * SR)]) < \
        rms_db(noisy[int(2 * SR):int(9 * SR)] - track[int(2 * SR):int(9 * SR)]) - 3


# --------------------------------------------------------------- inpainting


def sustained(dur=3.0):
    return sum(note(f, dur, seed=k) for k, f in enumerate([130.81, 196.0, 261.63, 329.63]))


def test_ar_inpainting_restores_a_short_gap():
    x = sustained()
    u0, u1 = int(1.0 * SR), int(1.0 * SR) + int(0.012 * SR)
    hurt = x.copy()
    hurt[u0:u1] = 0
    filled = dsp.ar_fill(hurt, u0, u1)
    snr = rms_db(x[u0:u1]) - rms_db(filled[u0:u1] - x[u0:u1])
    assert snr > 25, snr


def test_bridge_inpainting_holds_a_longer_gap():
    x = sustained()
    u0, u1 = int(1.0 * SR), int(1.0 * SR) + int(0.15 * SR)
    hurt = x.copy()
    hurt[u0:u1] = 0
    filled = dsp.ar_bridge(hurt, u0, u1, order=1024)
    snr = rms_db(x[u0:u1]) - rms_db(filled[u0:u1] - x[u0:u1])
    assert snr > 10, snr
    assert abs(rms_db(filled[u0:u1]) - rms_db(x[u0:u1])) < 2.0


def test_inpaint_tool_and_backend_selection(clean, write_wav, monkeypatch):
    monkeypatch.delenv("VIBESOUND_INPAINT_CMD", raising=False)
    monkeypatch.setattr(inpaint.ExternalBackend, "command", staticmethod(lambda: None))
    assert inpaint.choose(10).name == "ar"
    assert inpaint.choose(200).name == "bridge"
    with pytest.raises(ValueError, match="too long"):
        inpaint.choose(3000)

    class Fake:
        name, max_gap_ms = "fake-model", None

        def inpaint(self, audio, sr, a, b):
            out = audio.copy()
            out[a:b] = 0.123
            return out

    monkeypatch.setitem(inpaint.BACKENDS, "fake-model", Fake())
    assert inpaint.choose(3000).name == "fake-model"

    s = Session()
    s.open_project(write_wav(clean))
    report = s.inpaint(5.0, 7.0)
    assert report["backend"] == "fake-model"
    mid = s.project.audio()[int(5.5 * SR):int(6.5 * SR)]
    assert np.allclose(mid, 0.123, atol=1e-6)
    assert s.inpaint(3.300, 3.310, backend="ar")["backend"] == "ar"


def test_external_backend_runs_a_command(clean, write_wav, tmp_path, monkeypatch):
    import sys
    script = tmp_path / "model.py"
    script.write_text(
        "import sys, soundfile as sf\n"
        "x, sr = sf.read(sys.argv[1], always_2d=True)\n"
        "a, b = int(float(sys.argv[3]) * sr), int(float(sys.argv[4]) * sr)\n"
        "x[a:b] = 0.5\n"
        "sf.write(sys.argv[2], x, sr, subtype='FLOAT')\n"
    )
    monkeypatch.setattr(inpaint.ExternalBackend, "command",
                        staticmethod(lambda: [sys.executable, str(script), "{input}", "{output}", "{start}", "{end}"]))
    s = Session()
    s.open_project(write_wav(clean))
    assert s.inpaint(4.0, 5.0)["backend"] == "external"
    assert np.allclose(s.project.audio()[int(4.3 * SR):int(4.7 * SR)], 0.5, atol=1e-6)


def test_copy_from_region_aligns_to_the_destination(write_wav):
    phrase = np.zeros(int(2.5 * SR))
    for k, f in enumerate([220.0, 277.18, 329.63]):
        tone = note(f, 1.5, seed=k)
        phrase[int(k * 0.4 * SR):int(k * 0.4 * SR) + len(tone)] += tone
    gap = np.zeros(int(0.5 * SR) + 137)
    mono = np.concatenate([np.zeros(SR // 2), phrase, gap, phrase, np.zeros(SR // 2)])
    rng = np.random.default_rng(0)
    clean = np.stack([mono, mono], axis=1) + rng.normal(0, 1e-4, (len(mono), 2))
    dirty = clean.copy()
    d0 = 0.5 + 2.5 + len(gap) / SR + 1.0  # one second into the repeat
    i = int(d0 * SR)
    dirty[i:i + int(0.3 * SR)] += rng.normal(0, 0.05, (int(0.3 * SR), 2))

    s = Session()
    s.open_project(write_wav(dirty, subtype="FLOAT"))
    report = s.copy_from_region(source_start=1.5 + 0.004, dest_start=d0, length=0.3)  # deliberately 4 ms off
    fixed = s.project.audio()
    win = slice(i - 2000, i + int(0.3 * SR) + 2000)
    assert abs(report["align_shift_ms"] + 4.0) < 0.05
    assert rms_db(fixed[win] - clean[win]) < -55


# ----------------------------------------------------------------- pictures


def test_pictures_and_listen_page(clean, write_wav):
    s = Session()
    s.open_project(write_wav(clean))
    info, png = s.view("0:02", "0:08", marks=["0:05"], highlight_start=4, highlight_end=6)
    assert png[:3] == b"\xff\xd8\xff" and 20_000 < len(png) < 700_000
    s.reduce_mud(4, 8, 5)
    info, png = s.compare(3, 9)
    assert png[:3] == b"\xff\xd8\xff" and len(png) < 700_000 and info["band_change_b_minus_a_db"]["low-mid 250-500"] < -0.5
    assert not info["identical"]
    out = s.listen(4, 8, open_player=False)
    page = open(out["page"], encoding="utf-8").read()
    assert len(out["clips"]) == 2 and all(c.split("previews")[-1][1:] in page for c in out["clips"])
    assert sf.info(out["clips"][0]).duration == pytest.approx(10.0, abs=0.01)
    assert len(s.history()["versions"]) == 2


def test_swoosh_is_removed_but_partials_stay(clean, write_wav):
    rng = np.random.default_rng(7)
    dirty = clean.copy()
    i, n = int(6.3 * SR), int(0.35 * SR)
    sos = signal.butter(2, 2000, btype="high", fs=SR, output="sos")
    cloud = signal.sosfilt(sos, rng.normal(0, 0.004, (n, 2)), axis=0) * np.hanning(n)[:, None]
    dirty[i:i + n] += cloud

    s = Session()
    s.open_project(write_wav(dirty))
    s.remove_swoosh(6.25, 6.70)
    fixed = s.project.audio()
    win = slice(i + n // 4, i + 3 * n // 4)
    assert band_db(fixed[win] - clean[win], 2500, 12000) < band_db(dirty[win] - clean[win], 2500, 12000) - 4
    assert abs(band_db(fixed[win], 80, 1200) - band_db(clean[win], 80, 1200)) < 0.1
    assert np.array_equal(fixed[: int(6.0 * SR)], s.project.audio(0)[: int(6.0 * SR)])
