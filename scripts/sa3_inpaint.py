"""Stable Audio 3 inpainting backend for VibeSound's ``external`` hook.

Usage as VibeSound's external command (see src/vibesound/inpaint.py)::

    VIBESOUND_INPAINT_CMD="python scripts/sa3_inpaint.py --in {input} --out {output} --start {start} --end {end}"

The input WAV holds real audio on both sides of the gap. The model regenerates
the gap from that context, then only the gap (plus short crossfades) is spliced
back into the original, so audio outside the gap stays bit-identical.

Model: by default ``stabilityai/stable-audio-3-small-music-base`` (open, not
gated, CPU-capable). Point ``--model-dir`` at a local download of a Stable
Audio 3 small repo (the folder with model_config.json, model.safetensors and
t5gemma-b-b-ul2/), or set VIBESOUND_SA3_DIR.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_PROMPT = (
    "Solo felted upright-style grand piano, intimate close-mic recording, soft hammers, "
    "warm and muffled, slow emotional ambient neoclassical piano"
)
DEFAULT_NEGATIVE = "drums, percussion, vocals, strings, synth, noise, distortion"


def _patch_t5gemma_path(node, local_dir: str):
    """Rewrite conditioner configs that point at a (possibly gated) HF repo to the local copy."""
    if isinstance(node, dict):
        if "subfolder" in node and str(node.get("subfolder", "")).startswith("t5gemma"):
            node.pop("repo_id", None)
            node["model_path"] = str(Path(local_dir) / node.pop("subfolder"))
        for v in node.values():
            _patch_t5gemma_path(v, local_dir)
    elif isinstance(node, list):
        for v in node:
            _patch_t5gemma_path(v, local_dir)


class SA3Inpainter:
    def __init__(self, model_dir: str, threads: int | None = None):
        import torch
        from stable_audio_3.loading_utils import load_diffusion_cond
        from stable_audio_3.model import StableAudioModel

        if threads:
            torch.set_num_threads(threads)
        model_dir = str(Path(model_dir).expanduser())
        with open(Path(model_dir) / "model_config.json") as f:
            cfg = json.load(f)
        cfg = copy.deepcopy(cfg)
        _patch_t5gemma_path(cfg, model_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        half = device == "cuda"
        model = load_diffusion_cond(cfg, str(Path(model_dir) / "model.safetensors"), device=device, model_half=half)
        model.use_lora = False
        model.lora_names = []
        self.sam = StableAudioModel(model, cfg, device, half)
        self.sr = int(cfg["sample_rate"])
        self.base = "base" in Path(model_dir).name

    def generate(self, audio: np.ndarray, sr: int, start_s: float, end_s: float, *, prompt=DEFAULT_PROMPT,
                 negative=DEFAULT_NEGATIVE, steps=None, cfg_scale=None, seed=-1, guide=None) -> np.ndarray:
        """Return the model's full-length output (channels last, at ``sr``).

        ``guide`` (0-1) keeps the original gap as a blueprint: the model starts from the
        original audio with that much noise mixed in, so notes and timing survive and only
        the sound is re-rendered. None means blind inpainting (the gap is guessed from context).
        """
        import torch
        import torchaudio

        x = torch.from_numpy(np.ascontiguousarray(audio.T)).float()  # (ch, n)
        if x.shape[0] == 1:
            x = x.repeat(2, 1)
        if sr != self.sr:
            x = torchaudio.functional.resample(x, sr, self.sr)
        dur = x.shape[-1] / self.sr
        steps = steps or (50 if self.base else 8)
        cfg_scale = cfg_scale if cfg_scale is not None else 1.0  # guidance above 1 produced noise in CPU trials
        out = self.sam.generate(
            prompt=prompt,
            negative_prompt=negative if cfg_scale > 1.0 else None,
            duration=dur,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            inpaint_audio=(self.sr, x),
            **({"init_audio": (self.sr, x.clone()), "init_noise_level": float(guide)} if guide else {}),
            inpaint_mask_start_seconds=float(start_s),
            inpaint_mask_end_seconds=float(end_s),
        )
        y = out[0].float().cpu()
        if sr != self.sr:
            y = torchaudio.functional.resample(y, self.sr, sr)
        y = y.numpy().T
        n = audio.shape[0]
        if y.shape[0] < n:
            y = np.pad(y, ((0, n - y.shape[0]), (0, 0)))
        y = y[:n]
        if audio.shape[1] == 1:
            y = y.mean(axis=1, keepdims=True)
        return y.astype(np.float64)


def splice(orig: np.ndarray, gen: np.ndarray, sr: int, start_s: float, end_s: float, xfade_ms=25.0,
           match_level=True) -> np.ndarray:
    """Put the generated gap into the original with equal-power crossfades."""
    n = orig.shape[0]
    a, b = int(round(start_s * sr)), int(round(end_s * sr))
    xf = max(1, int(xfade_ms * 1e-3 * sr))
    gen = gen.copy()
    if match_level:
        # Match the model's loudness to the original on the context just outside the gap.
        ctx = int(0.5 * sr)
        idx = np.r_[max(0, a - ctx):a, b:min(n, b + ctx)]
        ro = np.sqrt(np.mean(orig[idx] ** 2) + 1e-12)
        rg = np.sqrt(np.mean(gen[idx] ** 2) + 1e-12)
        gen *= np.clip(ro / rg, 0.5, 2.0)
    w = np.zeros(n)
    w[a:b] = 1.0
    lo, hi = max(0, a - xf), min(n, b + xf)
    t = np.linspace(0, 1, a - lo, endpoint=False)
    w[lo:a] = np.sin(t * np.pi / 2) ** 2
    t = np.linspace(0, 1, hi - b, endpoint=False)
    w[b:hi] = np.cos(t * np.pi / 2) ** 2
    g, o = np.sqrt(w)[:, None], np.sqrt(1 - w)[:, None]
    return orig * o + gen * g


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--model-dir", default=os.environ.get("VIBESOUND_SA3_DIR", "~/models/sa3-small-music-base"))
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--cfg", type=float)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--guide", type=float, default=None,
                    help="0-1: re-render the gap from the original at this noise level (keeps notes and timing). "
                         "Omit for blind inpainting.")
    ap.add_argument("--xfade-ms", type=float, default=25.0)
    ap.add_argument("--threads", type=int)
    args = ap.parse_args(argv)

    audio, sr = sf.read(args.inp, dtype="float64", always_2d=True)
    t0 = time.time()
    inp = SA3Inpainter(args.model_dir, args.threads)
    gen = inp.generate(audio, sr, args.start, args.end, prompt=args.prompt, steps=args.steps,
                       cfg_scale=args.cfg, seed=args.seed, guide=args.guide)
    out = splice(audio, gen, sr, args.start, args.end, args.xfade_ms)
    sf.write(args.out, out.astype(np.float32), sr, subtype="FLOAT")
    print(f"sa3_inpaint: {args.end - args.start:.2f}s gap in {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
