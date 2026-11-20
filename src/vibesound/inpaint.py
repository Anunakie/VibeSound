"""Audio inpainting: regenerate a bad stretch from the clean audio around it.

V1 ships two signal-model backends that need no GPU:

* ``ar``      least-squares autoregressive interpolation, for gaps up to ~30 ms
* ``bridge``  forward/backward linear prediction, for gaps up to ~400 ms

Longer gaps need a generative model. That is the V2 hook: anything that
implements :class:`InpaintBackend` can be registered, and the ``external``
backend already lets a separate program (for example a diffusion model running
through DirectML, or on a rented GPU) do the work without touching this code.
See docs/inpainting-research.md.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf

from . import dsp

AR_MAX_MS = 30.0
BRIDGE_MAX_MS = 400.0


class InpaintBackend(Protocol):
    name: str
    max_gap_ms: float | None

    def inpaint(self, audio: np.ndarray, sr: int, gap_start: int, gap_end: int) -> np.ndarray:
        """Return ``audio`` (n, channels) with samples [gap_start, gap_end) regenerated.

        ``audio`` holds several seconds of context on both sides of the gap.
        Samples outside the gap may be returned changed; only the gap (plus the
        crossfade) is kept.
        """
        ...


class ARBackend:
    name = "ar"
    max_gap_ms: float | None = AR_MAX_MS

    def inpaint(self, audio, sr, gap_start, gap_end):
        out = audio.copy()
        for ch in range(out.shape[1]):
            out[:, ch] = dsp.ar_fill(out[:, ch], gap_start, gap_end)
        return out


class BridgeBackend:
    name = "bridge"
    max_gap_ms: float | None = BRIDGE_MAX_MS

    def inpaint(self, audio, sr, gap_start, gap_end):
        out = audio.copy()
        order = int(np.clip(sr * 0.023, 256, 2048))
        for ch in range(out.shape[1]):
            out[:, ch] = dsp.ar_bridge(out[:, ch], gap_start, gap_end, order=order)
        return out


class ExternalBackend:
    """Hand the job to another program.

    Configure with the environment variable ``VIBESOUND_INPAINT_CMD`` or the
    ``inpaint_command`` key in ``~/.vibesound/config.json``. The command is a
    template run without a shell; these placeholders are filled in:

        {input}   WAV with context + gap (gap samples are the damaged original)
        {output}  WAV the program must write, same length and channels
        {start}   gap start in seconds within the input file
        {end}     gap end in seconds within the input file

    Example: ``python C:\\models\\a2sb_inpaint.py --in {input} --out {output} --start {start} --end {end}``
    """

    name = "external"
    max_gap_ms: float | None = None

    @staticmethod
    def command() -> list[str] | None:
        raw: str | list[str] | None = os.environ.get("VIBESOUND_INPAINT_CMD")
        if not raw:
            cfg = Path.home() / ".vibesound" / "config.json"
            if cfg.exists():
                try:
                    raw = json.loads(cfg.read_text(encoding="utf-8")).get("inpaint_command")
                except (OSError, ValueError):
                    raw = None
        if not raw:
            return None
        if isinstance(raw, list):
            return [str(p) for p in raw]
        import shlex

        return shlex.split(raw, posix=(sys.platform != "win32"))

    def inpaint(self, audio, sr, gap_start, gap_end):
        cmd = self.command()
        if not cmd:
            raise RuntimeError(
                "No external inpainting model is configured. Set VIBESOUND_INPAINT_CMD or "
                "inpaint_command in ~/.vibesound/config.json (see docs/inpainting-research.md)."
            )
        with tempfile.TemporaryDirectory(prefix="vibesound_inpaint_") as tmp:
            src, dst = Path(tmp) / "input.wav", Path(tmp) / "output.wav"
            sf.write(str(src), audio.astype(np.float32), sr, subtype="FLOAT")
            fill = {"input": str(src), "output": str(dst),
                    "start": f"{gap_start / sr:.6f}", "end": f"{gap_end / sr:.6f}"}
            args = [part.format(**fill) for part in cmd]
            done = subprocess.run(args, capture_output=True, text=True, timeout=3600)
            if done.returncode != 0 or not dst.exists():
                raise RuntimeError(f"inpainting command failed ({done.returncode}): {done.stderr[-800:]}")
            out, out_sr = sf.read(str(dst), dtype="float64", always_2d=True)
        if out_sr != sr or out.shape != audio.shape:
            raise RuntimeError(
                f"inpainting command returned {out.shape} at {out_sr} Hz, expected {audio.shape} at {sr} Hz"
            )
        return out


BACKENDS: dict[str, InpaintBackend] = {b.name: b for b in (ARBackend(), BridgeBackend(), ExternalBackend())}


def register(backend: InpaintBackend) -> None:
    """Add or replace a backend (the V2 hook for an in-process model)."""
    BACKENDS[backend.name] = backend


def choose(gap_ms: float, requested: str = "auto") -> InpaintBackend:
    if requested != "auto":
        if requested not in BACKENDS:
            raise ValueError(f"unknown inpainting backend {requested!r}; have {sorted(BACKENDS)}")
        backend = BACKENDS[requested]
        if backend.max_gap_ms is not None and gap_ms > backend.max_gap_ms * 1.5:
            raise ValueError(
                f"the {requested!r} backend only holds up to about {backend.max_gap_ms:.0f} ms; "
                f"this gap is {gap_ms:.0f} ms"
            )
        return backend
    if gap_ms <= AR_MAX_MS:
        return BACKENDS["ar"]
    if ExternalBackend.command():
        return BACKENDS["external"]
    custom = [b for name, b in BACKENDS.items() if name not in ("ar", "bridge", "external")]
    if custom:
        return custom[0]
    if gap_ms <= BRIDGE_MAX_MS:
        return BACKENDS["bridge"]
    raise ValueError(
        f"A {gap_ms:.0f} ms gap is too long for the built-in signal-model inpainting "
        f"(limit about {BRIDGE_MAX_MS:.0f} ms) and no generative model is configured. "
        "Options: narrow the region, use spectral_repair on just the offending frequencies, "
        "or use copy_from_region to borrow a matching passage."
    )
