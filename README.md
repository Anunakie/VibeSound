# VibeSound

Edit a piano recording by talking about it.

You listen and say what you hear: *"2:15 to 2:45 the bottom half of the piano is muddy"*, *"pedal noise at 2:12"*. Claude looks at the audio, fixes it, and opens an A/B page so you can hear before and after. You say "more", "less" or "that's it", and you go back and forth until it sounds right.

VibeSound is a local [MCP](https://modelcontextprotocol.io) server. It runs on your PC, works on your WAV files where they are, and is driven by Claude Desktop or Claude Code on your normal Claude subscription. No API key, no audio leaves your machine (Claude only sees pictures and measurements of the regions it inspects).

## What it can do today (V1)

| You say | Claude uses |
|---|---|
| "the low end is muddy / boomy here" | `analyze` to find the heavy band, then `reduce_mud` (a dynamic cut that only bites where the lows pile up) or `eq` |
| "pedal thump at 2:12" | `find_problems` to pin the instant, `remove_thump` |
| "there's a click / tick" | `remove_clicks` (finds them itself, or at times you give) |
| "a swoosh when the pedal moves" | `remove_swoosh`: turns the cloud of noise back down to the level around it and keeps the note lines |
| "squeak / creak / hammer noise / chair" | `view` to see it, `spectral_repair` to rebuild just that time-and-frequency box from the audio around it. This is the "paint it away" job from RX, done by Claude instead of by hand |
| "too much hiss / action rustle" | `denoise`, ideally with a sample of the noise alone |
| "this little bit is ruined" | `inpaint` (very short stretches for now, see below) or `copy_from_region` to borrow the same bar from a repeat |
| "go back" / "try it stronger" | `undo`, `checkout`, `history`. Every edit is a version; nothing is ever lost |
| "let me hear it" | `listen` opens an A/B page in your browser; `export` writes a new WAV |

Every edit is crossfaded into the untouched audio around it, and the original file is never modified. Projects live in a `<name>.vibesound` folder beside the recording.

### About inpainting

True generative inpainting (regenerate a second or more of piano from the clean audio around it) needs a neural model, and that is V2. V1 includes two signal-model fills that work without a GPU: up to about 30 ms very accurately, and up to about 400 ms for sustained notes. The hook for a real model is already in place, and [docs/inpainting-research.md](docs/inpainting-research.md) covers which models can run on an AMD card.

## Install on Windows

You need Python 3.10 or newer ([python.org](https://www.python.org/downloads/windows/), tick "Add python.exe to PATH") and [Git](https://git-scm.com/download/win).

Open **PowerShell** and run:

```powershell
cd $HOME
git clone https://github.com/Anunakie/vibesound.git
cd vibesound
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
```

The script creates a private Python environment in the folder, installs VibeSound into it, and adds it to Claude Desktop's settings. Then **quit Claude Desktop completely** (right-click its icon in the system tray, Quit) and start it again. In a new chat, the tools menu should list **vibesound**.

<details>
<summary>Doing it by hand instead</summary>

```powershell
py -m venv .venv
.venv\Scripts\pip install -e .
```

In Claude Desktop: Settings, Developer, Edit Config. Add this to `claude_desktop_config.json` (use your real path, with doubled backslashes):

```json
{
  "mcpServers": {
    "vibesound": {
      "command": "C:\\Users\\YOU\\vibesound\\.venv\\Scripts\\python.exe",
      "args": ["-m", "vibesound.server"]
    }
  }
}
```

For Claude Code instead:

```powershell
claude mcp add vibesound -- "$HOME\vibesound\.venv\Scripts\python.exe" -m vibesound.server
```
</details>

## Using it

Start a chat in Claude Desktop and say something like:

> Open `D:\Recordings\nocturne-take3.wav` in VibeSound. From 2:15 to 2:45 the bottom half of the piano is muddy, can you clean that up?

Claude opens the recording, looks at that stretch, makes a gentle fix and opens the A/B page. On that page **Space** plays, **A** and **B** (or **Tab**) switch instantly between before and after, and it loops. Then just tell Claude what you think. When you are happy: "export it".

Tips:

- Times can be rough. Claude zooms in to find the exact spot.
- Describing the sound helps as much as the time: "boomy", "boxy", "a thud", "a tick", "papery".
- Point at a passage you like ("0:40 to 1:00 sounds right") and Claude will use it as the reference for tone.

## How it is built

```
src/vibesound/
  store.py     project folder, version tree, crossfaded patches (non-destructive)
  dsp.py       the processing: band cut, EQ, spectral repair, denoise, declick, AR fills
  analysis.py  what Claude "hears" with: band levels, thump/click detection, spectrograms
  inpaint.py   inpainting backends and the hook for a generative model
  engine.py    the editing session as a plain Python API (used by the server and tests)
  abpage.py    the A/B listening page
  server.py    MCP tool definitions and the instructions Claude follows
```

Run the tests with `pip install -e .[dev]` then `pytest`.

## Roadmap

1. **V1 (this):** MCP core, repair tools, version history, A/B page.
2. **Desktop UI (Tauri):** waveform and spectrogram view, drag to select a region, A/B and version list in one window, talking to the same engine.
3. **V2 inpainting:** a generative backend behind the existing hook.
4. **Later:** piano timbre swap and blends (for example 80% original, 20% another felted piano).

MIT licensed.
