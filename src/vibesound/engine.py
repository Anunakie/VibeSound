"""The editing session: everything the MCP tools do, as a plain Python API.

Kept free of MCP imports so it can be tested and reused by a future desktop UI.
Methods return JSON-ready dicts; picture methods return (dict, jpeg_bytes).
"""

from __future__ import annotations

import os
import sys
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal

from . import abpage, analysis, dsp, inpaint
from .store import EditResult, Project
from .timeutil import fmt_time, parse_time

Time = str | float | int


class Session:
    def __init__(self) -> None:
        self.project: Project | None = None

    # ----------------------------------------------------------------- helpers

    def _p(self) -> Project:
        if self.project is None:
            raise RuntimeError("No recording is open yet. Call open_project with the path to a WAV file first.")
        return self.project

    def _range(self, start: Time, end: Time) -> tuple[float, float]:
        p = self._p()
        s, e = parse_time(start), parse_time(end)
        if e <= s:
            raise ValueError(f"end ({fmt_time(e)}) must be after start ({fmt_time(s)})")
        if s >= p.duration:
            raise ValueError(f"start {fmt_time(s)} is past the end of the track ({fmt_time(p.duration)})")
        return s, min(e, p.duration)

    def _version_line(self, v: dict[str, Any]) -> dict[str, Any]:
        line = {"version": v["id"], "parent": v["parent"], "label": v["label"]}
        if "start" in v:
            line["range"] = f"{fmt_time(v['start'])} to {fmt_time(v['end'])}"
        if v.get("note"):
            line["note"] = v["note"]
        return line

    def _report(self, result: EditResult, label: str, extra: dict | None = None) -> dict:
        """What an edit measurably did, so Claude can sanity-check before Daniel listens."""
        p = self._p()
        before = p.clip(result.start, result.end, result.parent)
        after = p.clip(result.start, result.end, result.version)
        b_before, b_after = analysis.band_levels(before, p.sr), analysis.band_levels(after, p.sr)
        diff = np.asarray(after, dtype=np.float64) - before
        removed_db = float(analysis.db(np.sqrt(np.mean(diff ** 2)))) if len(diff) else -200.0
        report = {
            "new_version": result.version,
            "based_on": result.parent,
            "edit": label,
            "range": f"{fmt_time(result.start)} to {fmt_time(result.end)}",
            "band_change_db": {k: v for k, v in analysis.compare_bands(b_after, b_before).items() if abs(v) >= 0.1},
            "level_before_dbfs": round(float(analysis.db(np.sqrt(np.mean(before ** 2)))), 1),
            "level_of_what_changed_dbfs": round(removed_db, 1),
            "peak_after_dbfs": round(float(analysis.db(np.max(np.abs(after)))), 1),
            "next": "Use compare to see before/after, then listen so Daniel can A/B it. undo reverts.",
        }
        if extra:
            report.update(extra)
        return report

    # ----------------------------------------------------------------- project

    def open_project(self, path: str) -> dict:
        self.project = Project.open(path)
        return self.status()

    def status(self) -> dict:
        p = self._p()
        full = p.audio()
        return {
            "name": p.name,
            "project_folder": str(p.folder),
            "duration": fmt_time(p.duration),
            "duration_seconds": round(p.duration, 3),
            "sample_rate": p.sr,
            "channels": p.channels,
            "bit_format": p.meta.get("subtype"),
            "current_version": p.head,
            "version_count": len(p.meta["versions"]),
            "overall": analysis.basic_stats(full[:: max(1, len(full) // 4_000_000)], p.sr)
            | {"peak_dbfs": round(float(analysis.db(np.max(np.abs(full)))), 1)},
            "history": [self._version_line(v) for v in p.chain()][-8:],
        }

    def history(self) -> dict:
        p = self._p()
        on_path = {v["id"] for v in p.chain()}
        rows = []
        for v in p.meta["versions"]:
            row = self._version_line(v)
            row["op"] = v["op"]
            if v["params"]:
                row["settings"] = v["params"]
            row["in_current"] = v["id"] in on_path
            rows.append(row)
        return {"current_version": p.head, "versions": rows}

    def checkout(self, version: int) -> dict:
        p = self._p()
        p.checkout(version)
        return {"current_version": p.head, "includes": [self._version_line(v) for v in p.chain()]}

    def undo(self) -> dict:
        p = self._p()
        parent = p.version()["parent"]
        if parent is None:
            raise RuntimeError("Already at the original recording; nothing to undo.")
        undone = p.head
        p.checkout(parent)
        return {"undone_version": undone, "current_version": p.head,
                "note": f"Version {undone} is kept; checkout {undone} brings it back."}

    def add_note(self, text: str, version: int | None = None) -> dict:
        p = self._p()
        vid = p.head if version is None else version
        p.set_note(vid, text)
        return {"version": vid, "note": text}

    # ---------------------------------------------------------------- looking

    def view(self, start: Time, end: Time, version: int | None = None, fmin: float = 20.0,
             fmax: float = 20000.0, marks: list[Time] | None = None,
             highlight_start: Time | None = None, highlight_end: Time | None = None,
             floor_db: float = -110.0, ceiling_db: float = -20.0) -> tuple[dict, bytes]:
        p = self._p()
        s, e = self._range(start, end)
        vid = p.head if version is None else version
        clip = p.clip(s, e, vid)
        mark_list = [(parse_time(m), fmt_time(parse_time(m))) for m in (marks or [])]
        hl = None
        if highlight_start is not None and highlight_end is not None:
            hl = (parse_time(highlight_start), parse_time(highlight_end))
        picture = analysis.render_spectrogram(
            [(f"{p.name}  v{vid} ({p.version(vid)['label']})  {fmt_time(s)} to {fmt_time(e)}", clip)],
            p.sr, s, fmin=fmin, fmax=fmax, db_range=(floor_db, ceiling_db), marks=mark_list, highlight=hl,
        )
        info = {"version": vid, "range": f"{fmt_time(s)} to {fmt_time(e)}",
                "reading": "Log frequency axis, colour is level in dBFS (mono sum). Top strip is the waveform."}
        return info, picture

    def compare(self, start: Time, end: Time, a: int | None = None, b: int | None = None,
                fmin: float = 20.0, fmax: float = 20000.0) -> tuple[dict, bytes]:
        p = self._p()
        s, e = self._range(start, end)
        vb = p.head if b is None else b
        va = p.version(vb)["parent"] if a is None else a
        if va is None:
            va = 0
        clip_a, clip_b = p.clip(s, e, va), p.clip(s, e, vb)
        picture = analysis.render_spectrogram(
            [(f"A: v{va} ({p.version(va)['label']})", clip_a), (f"B: v{vb} ({p.version(vb)['label']})", clip_b)],
            p.sr, s, fmin=fmin, fmax=fmax, diff=True,
        )
        bands_a, bands_b = analysis.band_levels(clip_a, p.sr), analysis.band_levels(clip_b, p.sr)
        delta = np.asarray(clip_b, dtype=np.float64) - clip_a
        info = {
            "a": va, "b": vb, "range": f"{fmt_time(s)} to {fmt_time(e)}",
            "band_levels_a_dbfs": bands_a,
            "band_change_b_minus_a_db": analysis.compare_bands(bands_b, bands_a),
            "difference_rms_dbfs": round(float(analysis.db(np.sqrt(np.mean(delta ** 2)))), 1),
            "identical": bool(np.max(np.abs(delta)) == 0.0) if len(delta) else True,
        }
        return info, picture

    def analyze(self, start: Time, end: Time, reference_start: Time | None = None,
                reference_end: Time | None = None, version: int | None = None,
                timeline_step: float | None = None) -> dict:
        p = self._p()
        s, e = self._range(start, end)
        clip = p.clip(s, e, version)
        out: dict[str, Any] = {
            "range": f"{fmt_time(s)} to {fmt_time(e)}",
            "stats": analysis.basic_stats(clip, p.sr),
            "band_levels_dbfs": analysis.band_levels(clip, p.sr),
        }
        if reference_start is not None and reference_end is not None:
            rs, re_ = self._range(reference_start, reference_end)
            ref = p.clip(rs, re_, version)
            ref_name = f"{fmt_time(rs)} to {fmt_time(re_)}"
        else:
            # Default reference: the rest of the track, which is "how this piano normally sounds".
            full = p.audio(version)
            a, b = p.sample_range(s, e)
            ref = np.concatenate([full[:a], full[b:]]) if (len(full) - (b - a)) > p.sr else full
            limit = 120 * p.sr
            if len(ref) > limit:  # an even sample of long tracks keeps this fast
                idx = np.linspace(0, len(ref) - p.sr, 120).astype(int)
                ref = np.concatenate([ref[i : i + p.sr] for i in idx])
            ref_name = "rest of the track"
        ref_bands = analysis.band_levels(ref, p.sr)
        region, reference = out["band_levels_dbfs"], ref_bands
        # Compare spectral *shape*: remove the overall loudness difference first.
        # Bands far below the loudest one are mostly noise floor, and say nothing about tone.
        top = max(max(region.values()), max(reference.values()))
        musical = [k for k in region if min(region[k], reference[k]) > top - 45.0] or list(region)
        offset = float(np.median([region[k] - reference[k] for k in musical]))
        out["reference"] = ref_name
        out["reference_band_levels_dbfs"] = ref_bands
        out["loudness_offset_db"] = round(offset, 1)
        out["band_balance_vs_reference_db"] = {k: round(region[k] - reference[k] - offset, 1) for k in musical}
        quiet = [k for k in region if k not in musical]
        if quiet:
            out["bands_too_quiet_to_judge"] = quiet
        out["reading"] = ("band_balance_vs_reference_db is the tonal difference after matching loudness: "
                          "positive means this region has relatively more of that band than the reference.")
        if timeline_step:
            out["timeline"] = analysis.band_timeline(clip, p.sr, s, max(0.1, float(timeline_step)))
        return out

    def find_problems(self, start: Time, end: Time, version: int | None = None,
                      click_sensitivity: float = 10.0, thump_max_hz: float = 150.0,
                      thump_min_rise_db: float = 5.0) -> dict:
        p = self._p()
        s, e = self._range(start, end)
        clip = p.clip(s, e, version)
        thumps = analysis.find_thumps(clip, p.sr, max_hz=thump_max_hz, min_rise_db=thump_min_rise_db)
        for t in thumps:
            t["seconds"] = round(t["seconds"] + s, 3)
            t["time"] = fmt_time(t["seconds"])
        clicks = [
            {"time": fmt_time(c["time"] + s), "seconds": round(c["time"] + s, 4),
             "width_ms": c["width_ms"], "strength": c["strength"]}
            for c in analysis.find_clicks(clip, p.sr, threshold=click_sensitivity)
        ]
        return {
            "range": f"{fmt_time(s)} to {fmt_time(e)}",
            "thumps": thumps[:60], "clicks": clicks[:200],
            "thump_count": len(thumps), "click_count": len(clicks),
            "reading": ("Candidates only. Thumps are low-band jumps with no matching jump in the mids "
                        "(pedal, bench, floor). Clicks are sub-3 ms outliers. Confirm with view before fixing."),
        }

    # ------------------------------------------------------------------- fixes

    def reduce_mud(self, start: Time, end: Time, amount_db: float = 4.0, low_hz: float = 150.0,
                   high_hz: float = 450.0, dynamic: bool = True) -> dict:
        s, e = self._range(start, end)
        params = {"amount_db": amount_db, "low_hz": low_hz, "high_hz": high_hz, "dynamic": dynamic}
        label = f"reduce {low_hz:g}-{high_hz:g} Hz by {abs(amount_db):g} dB" + (" (dynamic)" if dynamic else "")
        result = self._p().apply_edit(
            s, e, lambda seg, sr, ra, rb: dsp.reduce_band(seg, sr, ra, rb, **params),
            op="reduce_mud", params=params, label=label, xfade_ms=250.0, context_s=1.5,
        )
        return self._report(result, label)

    def eq(self, start: Time, end: Time, bands: list[dict]) -> dict:
        s, e = self._range(start, end)
        if not bands:
            raise ValueError("give at least one EQ band, e.g. {'type': 'bell', 'freq': 300, 'gain_db': -3, 'q': 1.2}")
        dsp.eq_response(bands, self._p().sr, np.array([100.0, 1000.0]))  # validate before editing
        label = "EQ " + ", ".join(
            f"{b.get('type', 'bell')} {float(b['freq']):g} Hz {float(b.get('gain_db', 0)):+g} dB" for b in bands)
        result = self._p().apply_edit(
            s, e, lambda seg, sr, ra, rb: dsp.eq(seg, sr, ra, rb, bands=bands),
            op="eq", params={"bands": bands}, label=label, xfade_ms=250.0, context_s=1.5,
        )
        return self._report(result, label)

    def gain(self, start: Time, end: Time, gain_db: float) -> dict:
        s, e = self._range(start, end)
        label = f"gain {gain_db:+g} dB"
        result = self._p().apply_edit(
            s, e, lambda seg, sr, ra, rb: dsp.gain(seg, sr, ra, rb, gain_db=gain_db),
            op="gain", params={"gain_db": gain_db}, label=label, xfade_ms=150.0,
        )
        return self._report(result, label)

    def spectral_repair(self, start: Time, end: Time, low_hz: float = 20.0, high_hz: float = 20000.0,
                        mode: str = "attenuate", strength: float = 1.0) -> dict:
        s, e = self._range(start, end)
        if mode not in ("attenuate", "replace"):
            raise ValueError("mode must be 'attenuate' or 'replace'")
        if e - s > 5.0:
            raise ValueError("spectral_repair interpolates across the region and is meant for events up to a "
                             "few seconds. For longer stretches use reduce_mud, eq or denoise.")
        params = {"low_hz": low_hz, "high_hz": high_hz, "mode": mode, "strength": strength}
        label = f"spectral repair {low_hz:g}-{high_hz:g} Hz ({mode})"
        # Short events need short windows; long ones can afford frequency detail.
        win_ms = float(np.clip((e - s) * 1000 * 0.5, 23.0, 46.0))
        result = self._p().apply_edit(
            s, e, lambda seg, sr, ra, rb: dsp.spectral_repair(seg, sr, ra, rb, win_ms=win_ms, **params),
            op="spectral_repair", params=params, label=label, xfade_ms=12.0, context_s=1.0,
        )
        return self._report(result, label)

    def remove_swoosh(self, start: Time, end: Time, low_hz: float = 1500.0, high_hz: float = 20000.0,
                      max_reduction_db: float = 15.0, protect_db: float = 7.0) -> dict:
        s, e = self._range(start, end)
        if e - s > 4.0:
            raise ValueError("remove_swoosh compares the region with the audio just around it and is meant "
                             "for bursts up to a few seconds. For steady noise use denoise.")
        params = {"low_hz": low_hz, "high_hz": high_hz, "max_reduction_db": max_reduction_db,
                  "protect_db": protect_db}
        label = f"remove swoosh above {low_hz:g} Hz (up to {abs(max_reduction_db):g} dB)"
        result = self._p().apply_edit(
            s, e, lambda seg, sr, ra, rb: dsp.reduce_noise_burst(seg, sr, ra, rb, **params),
            op="remove_swoosh", params=params, label=label, xfade_ms=40.0, context_s=1.0,
        )
        return self._report(result, label)

    def remove_thump(self, time: Time, width_ms: float = 140.0, max_hz: float = 150.0,
                     strength: float = 1.0) -> dict:
        t = parse_time(time)
        half = width_ms / 2000.0
        s, e = self._range(max(0.0, t - half * 0.4), t + half * 1.6)  # a thump rings on after its onset
        params = {"low_hz": 10.0, "high_hz": max_hz, "mode": "attenuate", "strength": strength}
        label = f"remove thump at {fmt_time(t)} (below {max_hz:g} Hz)"
        result = self._p().apply_edit(
            s, e, lambda seg, sr, ra, rb: dsp.spectral_repair(seg, sr, ra, rb, win_ms=46.0,
                                                              context_ms=120.0, shoulder_oct=0.3, **params),
            op="remove_thump", params={"time": t, "width_ms": width_ms, "max_hz": max_hz, "strength": strength},
            label=label, xfade_ms=15.0, context_s=1.0,
        )
        return self._report(result, label)

    def remove_clicks(self, start: Time, end: Time, sensitivity: float = 10.0,
                      times: list[Time] | None = None, width_ms: float = 1.5) -> dict:
        p = self._p()
        s, e = self._range(start, end)
        a, b = p.sample_range(s, e)
        sr = p.sr
        if times:
            half = max(2, int(width_ms * 1e-3 * sr / 2))
            spans = []
            for t in times:
                c = int(round(parse_time(t) * sr))
                if not a <= c < b:
                    raise ValueError(f"click time {fmt_time(c / sr)} is outside {fmt_time(s)} to {fmt_time(e)}")
                spans.append((c - half, c + half))
        else:
            clip = p.clip(s, e)
            spans = [(a + c["start"], a + c["end"]) for c in analysis.find_clicks(clip, sr, threshold=sensitivity)]
        if not spans:
            return {"new_version": None, "clicks_found": 0,
                    "note": "No clicks found at this sensitivity; nothing changed. Lower sensitivity "
                            "(e.g. 7) finds fainter ones, or pass exact times."}
        if len(spans) > 2000:
            raise ValueError(f"{len(spans)} candidates is too many to be clicks; raise sensitivity or narrow the range")

        def fn(seg, sr_, ra, rb):
            shift = a - ra
            return dsp.declick(seg, sr_, ra, rb, clicks=[(u0 - shift, u1 - shift) for u0, u1 in spans])

        label = f"remove {len(spans)} click{'s' if len(spans) != 1 else ''}"
        result = p.apply_edit(s, e, fn, op="remove_clicks", label=label, xfade_ms=5.0, context_s=0.5,
                              params={"sensitivity": sensitivity, "count": len(spans),
                                      "times": [round((u0 + u1) / (2 * sr), 4) for u0, u1 in spans][:200]})
        return self._report(result, label, {"clicks_repaired": len(spans),
                                            "at": [fmt_time((u0 + u1) / (2 * sr)) for u0, u1 in spans][:50]})

    def denoise(self, start: Time, end: Time, amount_db: float = 8.0, noise_start: Time | None = None,
                noise_end: Time | None = None, sensitivity: float = 1.5) -> dict:
        p = self._p()
        s, e = self._range(start, end)
        profile = None
        params: dict[str, Any] = {"amount_db": amount_db, "sensitivity": sensitivity}
        if noise_start is not None and noise_end is not None:
            ns, ne = self._range(noise_start, noise_end)
            if ne - ns < 0.15:
                raise ValueError("the noise sample should be at least 0.15 s of noise with no notes ringing")
            profile = dsp.noise_profile(np.asarray(p.clip(ns, ne), dtype=np.float64), p.sr)
            params |= {"noise_start": ns, "noise_end": ne}
        label = f"denoise up to {abs(amount_db):g} dB" + ("" if profile is not None else " (auto profile)")
        result = p.apply_edit(
            s, e, lambda seg, sr, ra, rb: dsp.denoise(seg, sr, ra, rb, profile=profile,
                                                      amount_db=amount_db, sensitivity=sensitivity),
            op="denoise", params=params, label=label, xfade_ms=200.0, context_s=1.5,
        )
        return self._report(result, label)

    def inpaint(self, start: Time, end: Time, backend: str = "auto") -> dict:
        p = self._p()
        s, e = self._range(start, end)
        gap_ms = (e - s) * 1000.0
        chosen = inpaint.choose(gap_ms, backend)
        label = f"inpaint {gap_ms:.0f} ms ({chosen.name})"
        result = p.apply_edit(
            s, e, lambda seg, sr, ra, rb: chosen.inpaint(seg, sr, ra, rb),
            op="inpaint", params={"backend": chosen.name, "gap_ms": round(gap_ms, 1)}, label=label,
            xfade_ms=float(np.clip(gap_ms * 0.1, 1.0, 20.0)), context_s=6.0 if chosen.name not in ("ar", "bridge") else 1.0,
        )
        return self._report(result, label, {"backend": chosen.name})

    def copy_from_region(self, source_start: Time, dest_start: Time, length: float, align: bool = True) -> dict:
        """Replace [dest_start, dest_start+length] with the same-length audio starting at source_start."""
        p = self._p()
        sr = p.sr
        d0, d1 = self._range(dest_start, parse_time(dest_start) + float(length))
        src0 = parse_time(source_start)
        base = p.audio()
        n = int(round((d1 - d0) * sr))
        da = int(round(d0 * sr))
        sa = int(round(src0 * sr))
        lag = 0
        if align:
            # Line the source up with the destination using the clean audio just outside the region.
            ctx, reach = int(0.2 * sr), int(0.015 * sr)
            score = np.zeros(2 * reach + 1)
            for dst_lo, src_lo in ((da - ctx, sa - ctx), (da + n, sa + n)):
                if dst_lo < 0 or dst_lo + ctx > len(base) or src_lo - reach < 0 or src_lo + ctx + reach > len(base):
                    continue
                dst = base[dst_lo : dst_lo + ctx].mean(axis=1)
                src = base[src_lo - reach : src_lo + ctx + reach].mean(axis=1)
                score += signal.correlate(src, dst, mode="valid")
            if np.any(score):
                lag = int(np.argmax(score)) - reach
        sa += lag
        if sa < 0 or sa + n > p.frames:
            raise ValueError("the source range runs off the track")

        def fn(seg, sr_, ra, rb):
            out = seg.copy()
            off = sa - ra  # segment index 'ra' corresponds to source sample 'sa'
            lo, hi = max(0, -off), min(len(seg), len(base) - off)
            out[lo:hi] = base[lo + off : hi + off]
            return out

        label = f"copy {n / sr:.2f} s from {fmt_time(sa / sr)}"
        result = p.apply_edit(d0, d1, fn, op="copy_from_region", label=label, xfade_ms=40.0, context_s=0.5,
                              params={"source_start": round(sa / sr, 5), "length": round(n / sr, 5),
                                      "align_shift_ms": round(lag / sr * 1000, 2)})
        return self._report(result, label, {"align_shift_ms": round(lag / sr * 1000, 2)})

    # --------------------------------------------------------------- listening

    def listen(self, start: Time, end: Time, a: int | None = None, b: int | None = None,
               preroll: float = 3.0, open_player: bool = True) -> dict:
        """Write an A/B page for a region (with lead-in) and open it in the browser."""
        p = self._p()
        s, e = self._range(start, end)
        vb = p.head if b is None else b
        va = p.version(vb)["parent"] if a is None else a
        s0, e0 = max(0.0, s - preroll), min(p.duration, e + preroll)
        folder = p.folder / "previews"
        folder.mkdir(exist_ok=True)
        stamp = f"{int(s0 * 1000):08d}-{int(e0 * 1000):08d}"
        versions = [v for v in dict.fromkeys([va, vb]) if v is not None]
        clips, panels = [], []
        for key, vid in zip("AB", versions):
            name = f"clip_{stamp}_v{vid}.wav"
            p.export(folder / name, vid, subtype="PCM_24", start=s0, end=e0)
            clips.append({"key": key, "label": f"v{vid} {p.version(vid)['label']}", "file": name})
            panels.append((f"{key}: v{vid} ({p.version(vid)['label']})", p.clip(s0, e0, vid)))
        picture = analysis.render_spectrogram(panels, p.sr, s0, highlight=(s, e), diff=len(panels) == 2,
                                          waveform=False)
        page = folder / f"listen_{stamp}.html"
        page.write_text(abpage.build(f"{p.name}: {fmt_time(s)} to {fmt_time(e)}",
                                     f"Playing {fmt_time(s0)} to {fmt_time(e0)} with lead-in. "
                                     "The edited range is marked in cyan below.",
                                     clips, s0, picture), encoding="utf-8")
        opened = False
        if open_player:
            opened = _open_file(page)
        return {"page": str(page), "opened": opened, "clips": [str(folder / c["file"]) for c in clips],
                "a": versions[0], "b": versions[-1],
                "tell_daniel": "The A/B page is open in the browser: Space plays, A and B (or Tab) switch instantly."
                if opened else f"Open {page} in a browser to A/B."}

    def export(self, path: str | None = None, version: int | None = None, bit_format: str | None = None) -> dict:
        p = self._p()
        vid = p.head if version is None else version
        if path is None:
            path = str(Path(p.meta["source"]).with_name(f"{p.name}_v{vid}.wav"))
        out = Path(path).expanduser()
        if out.resolve() == Path(p.meta["source"]).resolve():
            raise ValueError("refusing to overwrite the source recording; choose another file name")
        written = p.export(out, vid, subtype=bit_format)
        info = sf.info(str(written))
        return {"written": str(written), "version": vid, "sample_rate": info.samplerate,
                "bit_format": info.subtype, "duration": fmt_time(info.duration)}


def _open_file(path: Path) -> bool:
    try:
        if sys.platform == "win32":
            os.startfile(str(path))  # type: ignore[attr-defined]
            return True
        return bool(webbrowser.open(path.as_uri()))
    except Exception:
        return False
