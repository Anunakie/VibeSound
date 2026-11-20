"""Non-destructive project store.

A project is a folder next to the source WAV (``<name>.vibesound/``) holding a
copy of the original and a tree of versions. Each version is the parent's audio
plus one small "patch": the processed samples for the edited time range, with
the crossfade into the surrounding audio already baked in. The original is
never modified, any version can be rendered at any time, and undo is just
moving the ``head`` pointer.
"""

from __future__ import annotations

import json
import shutil
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

PROJECT_SUFFIX = ".vibesound"
DEFAULT_XFADE_MS = 30.0

ProcessFn = Callable[[np.ndarray, int, int, int], np.ndarray]
"""fn(segment, sr, region_start_in_segment, region_end_in_segment) -> processed segment."""


@dataclass
class EditResult:
    version: int
    parent: int
    start: float
    end: float


class Project:
    """One recording and its version history."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.meta_path = self.folder / "project.json"
        if not self.meta_path.exists():
            raise FileNotFoundError(f"no VibeSound project at {self.folder}")
        self.meta: dict[str, Any] = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    # ------------------------------------------------------------------ create

    @classmethod
    def open(cls, wav_path: str | Path) -> "Project":
        """Open the project for a WAV file (or a project folder), creating it on first use."""
        path = Path(wav_path).expanduser()
        if path.is_dir() and (path / "project.json").exists():
            return cls(path)
        if path.suffix == PROJECT_SUFFIX and (path / "project.json").exists():
            return cls(path)
        if not path.is_file():
            raise FileNotFoundError(f"audio file not found: {path}")
        folder = path.with_name(path.stem + PROJECT_SUFFIX)
        if (folder / "project.json").exists():
            return cls(folder)

        info = sf.info(str(path))
        (folder / "patches").mkdir(parents=True, exist_ok=True)
        (folder / "previews").mkdir(exist_ok=True)
        original = folder / ("original" + path.suffix.lower())
        shutil.copy2(path, original)
        meta = {
            "name": path.stem,
            "source": str(path),
            "original": original.name,
            "sr": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "format": info.format,
            "subtype": info.subtype,
            "head": 0,
            "versions": [
                {
                    "id": 0,
                    "parent": None,
                    "label": "original",
                    "op": "import",
                    "params": {},
                    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            ],
        }
        (folder / "project.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return cls(folder)

    # -------------------------------------------------------------- properties

    @property
    def sr(self) -> int:
        return int(self.meta["sr"])

    @property
    def channels(self) -> int:
        return int(self.meta["channels"])

    @property
    def frames(self) -> int:
        return int(self.meta["frames"])

    @property
    def duration(self) -> float:
        return self.frames / self.sr

    @property
    def head(self) -> int:
        return int(self.meta["head"])

    @property
    def name(self) -> str:
        return str(self.meta["name"])

    def save(self) -> None:
        tmp = self.meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        tmp.replace(self.meta_path)

    def version(self, version: int | None = None) -> dict[str, Any]:
        vid = self.head if version is None else int(version)
        for entry in self.meta["versions"]:
            if entry["id"] == vid:
                return entry
        raise KeyError(f"version {vid} does not exist (have 0..{len(self.meta['versions']) - 1})")

    def chain(self, version: int | None = None) -> list[dict[str, Any]]:
        """Versions from the original up to ``version``, in order."""
        out = []
        entry: dict[str, Any] | None = self.version(version)
        while entry is not None:
            out.append(entry)
            entry = None if entry["parent"] is None else self.version(entry["parent"])
        return out[::-1]

    # ------------------------------------------------------------------- audio

    def _read_original(self) -> np.ndarray:
        data, sr = sf.read(str(self.folder / self.meta["original"]), dtype="float32", always_2d=True)
        if sr != self.sr:
            raise RuntimeError("original file changed sample rate")
        return data

    def audio(self, version: int | None = None) -> np.ndarray:
        """Full audio of a version as float32, shape (frames, channels). Do not modify in place."""
        vid = self.head if version is None else int(version)
        if vid in self._cache:
            self._cache.move_to_end(vid)
            return self._cache[vid]
        entry = self.version(vid)
        if entry["parent"] is None:
            data = self._read_original()
        else:
            data = self.audio(entry["parent"]).copy()
            patch, _ = sf.read(str(self.folder / "patches" / entry["patch"]), dtype="float32", always_2d=True)
            start = int(entry["patch_start"])
            data[start : start + len(patch)] = patch
        data.setflags(write=False)
        self._cache[vid] = data
        while len(self._cache) > 3:
            self._cache.popitem(last=False)
        return data

    def clip(self, start: float, end: float, version: int | None = None) -> np.ndarray:
        a, b = self.sample_range(start, end)
        return self.audio(version)[a:b]

    def sample_range(self, start: float, end: float) -> tuple[int, int]:
        a = int(round(max(0.0, start) * self.sr))
        b = int(round(min(self.duration, end) * self.sr))
        a = min(a, self.frames)
        b = max(a, min(b, self.frames))
        if b - a < 1:
            raise ValueError(
                f"time range {start:.3f}s to {end:.3f}s is empty (track is {self.duration:.3f}s long)"
            )
        return a, b

    # ------------------------------------------------------------------- edits

    def apply_edit(
        self,
        start: float,
        end: float,
        fn: ProcessFn,
        *,
        op: str,
        params: dict[str, Any],
        label: str,
        parent: int | None = None,
        xfade_ms: float = DEFAULT_XFADE_MS,
        context_s: float = 1.0,
    ) -> EditResult:
        """Run ``fn`` on [start, end] of the parent version and store the result as a new version.

        ``fn`` receives the region plus ``context_s`` of real audio on both sides
        so filters and spectral tools have settled before the region begins. The
        processed audio is then crossfaded back into the untouched audio with
        ramps just outside the region edges, so an edit can never leave a click.
        """
        parent_id = self.head if parent is None else int(parent)
        base = self.audio(parent_id)
        a, b = self.sample_range(start, end)
        sr = self.sr

        xf = max(8, int(round(xfade_ms * 1e-3 * sr)))
        ctx = max(int(context_s * sr), xf * 2)
        seg_a = max(0, a - ctx)
        seg_b = min(self.frames, b + ctx)
        segment = np.array(base[seg_a:seg_b], dtype=np.float64)

        processed = np.asarray(fn(segment.copy(), sr, a - seg_a, b - seg_a), dtype=np.float64)
        if processed.shape != segment.shape:
            raise RuntimeError(f"{op} returned shape {processed.shape}, expected {segment.shape}")
        if not np.all(np.isfinite(processed)):
            raise RuntimeError(f"{op} produced non-finite samples; edit discarded")

        # Blend weight: 1 across the whole region, raised-cosine ramps to 0 just outside it,
        # so the region itself is 100% the processed audio (a repair leaves no residue).
        n = seg_b - seg_a
        ra, rb = a - seg_a, b - seg_a
        weight = np.zeros(n)
        weight[ra:rb] = 1.0
        ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(xf) + 0.5) / xf)
        k = min(xf, ra)
        if k:
            weight[ra - k : ra] = ramp[xf - k :] if a - k > 0 else 1.0
        k = min(xf, n - rb)
        if k:
            weight[rb : rb + k] = ramp[::-1][:k] if b + k < self.frames else 1.0
        mixed = segment + weight[:, None] * (processed - segment)

        touched = np.flatnonzero(weight > 0)
        p0, p1 = int(touched[0]), int(touched[-1]) + 1
        patch = mixed[p0:p1].astype(np.float32)

        vid = max(v["id"] for v in self.meta["versions"]) + 1
        patch_name = f"v{vid:04d}.wav"
        sf.write(str(self.folder / "patches" / patch_name), patch, sr, subtype="FLOAT")
        self.meta["versions"].append(
            {
                "id": vid,
                "parent": parent_id,
                "label": label,
                "op": op,
                "params": params,
                "start": round(a / sr, 6),
                "end": round(b / sr, 6),
                "patch": patch_name,
                "patch_start": seg_a + p0,
                "patch_frames": int(p1 - p0),
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        self.meta["head"] = vid
        self.save()
        return EditResult(version=vid, parent=parent_id, start=a / sr, end=b / sr)

    def checkout(self, version: int) -> None:
        self.version(version)  # validates
        self.meta["head"] = int(version)
        self.save()

    def set_note(self, version: int, note: str) -> None:
        self.version(version)["note"] = note
        self.save()

    # ------------------------------------------------------------------ export

    def export(self, path: str | Path, version: int | None = None, subtype: str | None = None,
               start: float | None = None, end: float | None = None) -> Path:
        out = Path(path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        data = self.audio(version)
        if start is not None or end is not None:
            a, b = self.sample_range(start or 0.0, self.duration if end is None else end)
            data = data[a:b]
        fmt = "WAV" if out.suffix.lower() == ".wav" else None
        sub = subtype or self.meta.get("subtype")
        if fmt and not sf.check_format("WAV", sub):
            sub = "PCM_24"
        peak = float(np.max(np.abs(data))) if len(data) else 0.0
        if peak > 1.0 and sub and sub.startswith("PCM"):
            data = data / peak * 0.999  # integer formats cannot hold overs
        sf.write(str(out), data, self.sr, subtype=sub if fmt else None, format=fmt)
        return out
