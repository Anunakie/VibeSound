# Inpainting on an AMD RX 6600 XT: what can actually run

Researched 20 November 2026. Goal: regenerate a damaged stretch of a solo felted-piano recording from the clean audio before and after it, on a Windows PC with a Radeon RX 6600 XT (8 GB, no CUDA).

## Short version

- **Best fit to try first: Stable Audio 3 Small.** Open weights, has real inpainting (give it a file and a start/end time), 44.1 kHz stereo, and is designed to run on a CPU with no GPU at all. That sidesteps the AMD problem completely.
- **Best quality candidate for restoration: NVIDIA A2SB.** It is built for exactly this job (music inpainting and bandwidth extension at 44.1 kHz), but it is non-commercial only, mono, CUDA-oriented and 565M parameters. Realistic on this PC only on CPU (slow) or on a rented GPU.
- **The RX 6600 XT is not supported by AMD's PyTorch for Windows.** Plan for CPU or DirectML, and treat the GPU as a bonus.
- The piano-specific risk with every general model is that it will not match *this* felted C7. The likely end state is a general model plus a light fine-tune on Daniel's own recordings, which needs a rented GPU for a few hours, once.

## The GPU situation

AMD now ships official PyTorch for Windows (ROCm 7.2, January 2026; docs list 7.2.1 with PyTorch 2.9, Python 3.12, Windows 11). Supported Radeon cards are the RX 9070 / 9070 XT / 9060 XT, RX 7900 XTX, RX 7700 and the PRO W7900 / R9700, plus Ryzen AI laptop chips. **No RX 6000 series card is listed**, so the 6600 XT cannot use it.

What does work on a 6600 XT:

| Route | Status | Notes |
|---|---|---|
| CPU PyTorch | Works everywhere | Fine for small models and short gaps; slow for 500M+ diffusion models |
| ONNX Runtime + DirectML | Works on any DX12 card including RDNA 2 | Model must be exported to ONNX; diffusion samplers need some porting |
| `torch-directml` | Works, but lags PyTorch releases and misses some ops | Good for experiments, not guaranteed for every model |
| ZLUDA (CUDA shim) | Community project, hit and miss | Not something to build a product on |
| Rented cloud GPU | Always works | A few cents per fix; audio leaves the machine |

## Candidate models

### Stable Audio 3 (Stability AI, May 2026)

- Small (about 433M to 459M parameters, music and SFX variants) and Medium (1.4B) have downloadable weights; Large is API only.
- Inpainting is built in: `stable-audio --inpaint-audio input.wav --inpaint-start 4 --inpaint-end 8`, or `model.generate(inpaint_audio=..., inpaint_mask_start_seconds=..., inpaint_mask_end_seconds=..., prompt=...)`.
- 44.1 kHz stereo. Small handles up to 120 s, Medium up to 380 s.
- Small is stated to need no GPU (peak about 1.7 to 2.4 GB of memory). Medium peaks at about 5 to 6.5 GB of VRAM but needs CUDA and Flash Attention 2, so it is out for this card.
- Weights are under the Stability AI Community License. Check its terms before releasing music commercially.
- Open questions to test: it is a text-prompted generator, so how faithfully it continues *this* piano's tone across the seam, and whether its 44.1 kHz output is acceptable for sources recorded at 48 or 96 kHz (VibeSound would resample only the patched stretch).

### A2SB: Audio-to-Audio Schrödinger Bridges (NVIDIA, 2025)

- Purpose-built restoration model: inpainting and bandwidth extension for music at 44.1 kHz, works directly on waveforms/spectrograms rather than through a lossy codec, so the result stays close to the source.
- Demos fill 1000 ms gaps. Trained on about 3 s windows and extended to long files.
- Checkpoints on Hugging Face (`nvidia/audio_to_audio_schrodinger_bridge`), 565M parameters, PyTorch Lightning.
- **Licence: NVIDIA non-commercial for both code and weights.** Fine for experiments, not for released recordings without a licence from NVIDIA.
- Mono (stereo is listed as future work), so stereo needs mid/side or per-channel processing.
- The published inpainting script works on a dataset manifest, not one file; a small wrapper is needed.
- Training data includes MusicNet and Medley-Solos-DB, so it has heard solo piano.

### Stable Audio Open 1.0 / Open Small (2024 to 2025)

Text-to-audio models; inpainting was not part of the open release. Superseded by Stable Audio 3 for this purpose.

### Not neural, already in V1

`ar` (least-squares autoregressive interpolation) restores gaps up to about 30 ms almost exactly, and `bridge` holds sustained notes for up to about 400 ms. Neither can invent a note attack. `copy_from_region` borrows the same passage from a repeat, which for real music is often the most convincing fix of all.

## Recommended path for V2

1. Wrap **Stable Audio 3 Small** as an external backend and try it on real damaged spots at 0.5 s, 1 s and 3 s. It costs nothing and runs on the CPU.
2. In parallel, try **A2SB** on CPU for short gaps to hear the quality ceiling of a dedicated restoration model.
3. If neither matches the piano closely enough, fine-tune the better one on a few hours of Daniel's own C7 recordings on a rented GPU, then run inference locally.
4. Keep the generated fill honest: VibeSound already crossfades it in, and can level- and tone-match it to the surrounding audio using the same measurements Claude uses to inspect it.

## The hook that is already there

`vibesound/inpaint.py` has an `external` backend. Point it at any program and `inpaint` will use it automatically for gaps longer than 30 ms:

```json
// %USERPROFILE%\.vibesound\config.json
{ "inpaint_command": "C:\\models\\sa3\\python.exe C:\\models\\sa3\\fill.py --in {input} --out {output} --start {start} --end {end}" }
```

VibeSound writes a WAV with six seconds of context on each side, passes the gap position in seconds, reads back a WAV of the same length, and crossfades only the regenerated stretch into the track as a new version. A model can also be registered in-process with `vibesound.inpaint.register(backend)`.

## Sources

- AMD PyTorch on Windows 7.2 release notes: https://www.amd.com/en/resources/support-articles/release-notes/RN-AMDGPU-WINDOWS-PYTORCH-7-2.html
- ROCm Windows compatibility matrix: https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html
- A2SB code: https://github.com/nvidia/diffusion-audio-restoration
- A2SB project page: https://research.nvidia.com/labs/adlr/A2SB
- stable-audio-tools: https://github.com/Stability-AI/stable-audio-tools
- Stable Audio 3 overview: https://artintech.substack.com/p/stable-audio-3-explained-in-5-figures
