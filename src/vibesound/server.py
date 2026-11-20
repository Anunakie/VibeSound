"""VibeSound MCP server: the tools Claude Desktop / Claude Code call.

Run with ``python -m vibesound.server`` (stdio transport).
"""

from __future__ import annotations

import functools
import json
from typing import Any

try:  # MCP Python SDK 2.x
    from mcp.server.mcpserver import Image, MCPServer as _Server
except ImportError:  # MCP Python SDK 1.x
    from mcp.server.fastmcp import FastMCP as _Server, Image

from .engine import Session

INSTRUCTIONS = """\
VibeSound lets you edit a piano recording by conversation. The person (Daniel) listens and
describes what he hears in plain words, with rough times ("2:15 to 2:45 the bottom half is muddy",
"pedal noise at 2:12"). You cannot hear, so you look and measure instead, then fix, then he listens.

The loop for every request:
1. LOOK first. `view` the area (widen a few seconds around the times he gave; his times are rough),
   and use `analyze` (compare with a nearby passage that sounds fine) or `find_problems` to pin down
   what and exactly where. Zoom: a second `view` of 1-3 s with a tighter fmin/fmax shows detail.
2. FIX conservatively with the narrowest tool: reduce_mud / eq for tone over a passage,
   remove_thump for pedal and bench thumps, remove_clicks for ticks, spectral_repair to rebuild a
   specific time/frequency box (squeaks, creaks, hammer/action noise), denoise for steady noise,
   inpaint to regenerate a very short stretch, copy_from_region to borrow a matching passage.
   Start gentle (3-4 dB tone moves). It is easier for him to ask for more than to un-hear damage.
3. CHECK with `compare`: confirm the change landed where intended and nothing else moved.
4. LET HIM LISTEN: call `listen` on the region so he can A/B, and tell him in one or two plain
   sentences what you changed. Then wait for his reaction.
5. ITERATE. "More"/"less"/"too thin now" means `undo` and redo with new settings rather than
   stacking fixes on fixes. Record what he said with `add_note`. Nothing is ever lost: every edit
   is a version, `history` lists them, `checkout` jumps to any of them.

Times can be "m:ss", "m:ss.s" or seconds. Every edit is crossfaded into the surrounding audio.
The original file is never modified; `export` writes a new WAV when he is happy.
Speak to him as a musician, not an engineer: say "the low notes between 150 and 400 Hz" rather than
quoting parameter names, and never claim something sounds better. Only he can judge that.
"""

mcp = _Server("vibesound", instructions=INSTRUCTIONS)
session = Session()

Time = str | float


def _text(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=1, ensure_ascii=False)


def _with_image(result: tuple[dict, bytes]) -> list:
    info, picture = result
    return [_text(info), Image(data=picture, format="jpeg")]


def tool(fn):
    """Register an MCP tool whose failures come back as a readable message.

    Claude needs the reason ("end must be after start") to correct itself, and SDK versions
    differ in whether they pass exception text through.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - every failure should reach the model as text
            message = f"Could not do that: {exc}"
            wants_list = fn.__annotations__.get("return") in (list, "list")
            return [message] if wants_list else message

    return mcp.tool()(wrapper)


# ------------------------------------------------------------------- project


@tool
def open_project(path: str) -> str:
    """Open a WAV recording for editing (full path to the file).

    The first time, this creates a project folder next to it ("<name>.vibesound") holding a copy
    of the original and all versions. Opening it again resumes where you left off.
    """
    return _text(session.open_project(path))


@tool
def status() -> str:
    """Summary of the open recording: length, format, levels, current version, recent edits."""
    return _text(session.status())


@tool
def history() -> str:
    """List every version with what it changed, where, and any notes about how it sounded."""
    return _text(session.history())


@tool
def undo() -> str:
    """Step back to the version before the current one. The undone version is kept."""
    return _text(session.undo())


@tool
def checkout(version: int) -> str:
    """Make any version the current one (0 is the untouched original). New edits build on it."""
    return _text(session.checkout(version))


@tool
def add_note(text: str, version: int | None = None) -> str:
    """Attach a note to a version, such as Daniel's reaction ("better, but still boomy at 2:31")."""
    return _text(session.add_note(text, version))


# ------------------------------------------------------------------- looking


@tool
def view(start: Time, end: Time, version: int | None = None, fmin: float = 20.0, fmax: float = 20000.0,
         marks: list[Time] | None = None, highlight_start: Time | None = None,
         highlight_end: Time | None = None, floor_db: float = -110.0, ceiling_db: float = -20.0) -> list:
    """Spectrogram picture (log frequency, dBFS colour) plus waveform of a time range.

    This is how you "listen". Use the whole track for an overview, 5-30 s to find an event, and
    0.5-3 s to see its exact time and frequency extent. fmin/fmax zoom the frequency axis (for
    pedal thumps try fmax=500). marks draws dashed lines at times; highlight shades a range.
    Raise floor_db (e.g. -90) if quiet noise clutters the picture; lower it to see the noise floor.
    """
    return _with_image(session.view(start, end, version, fmin, fmax, marks, highlight_start,
                                    highlight_end, floor_db, ceiling_db))


@tool
def compare(start: Time, end: Time, a: int | None = None, b: int | None = None,
            fmin: float = 20.0, fmax: float = 20000.0) -> list:
    """Before/after picture of two versions with a third panel showing exactly what changed.

    Defaults: b is the current version and a is the one before it. Also returns the level change
    per frequency band. Use after every fix to confirm it did what was intended and nothing more.
    """
    return _with_image(session.compare(start, end, a, b, fmin, fmax))


@tool
def analyze(start: Time, end: Time, reference_start: Time | None = None, reference_end: Time | None = None,
            version: int | None = None, timeline_step: float | None = None) -> str:
    """Measure a region: level, peak, and energy in nine frequency bands, compared with a reference.

    The reference defaults to the rest of the track; better, pass a nearby passage Daniel is happy
    with. "band_balance_vs_reference_db" is the tonal difference after matching loudness, e.g.
    +4 in low-mid 250-500 means that band is 4 dB heavier here: that is what "muddy" looks like.
    timeline_step (seconds) adds band levels over time to find when a band builds up.
    """
    return _text(session.analyze(start, end, reference_start, reference_end, version, timeline_step))


@tool
def find_problems(start: Time, end: Time, version: int | None = None, click_sensitivity: float = 10.0,
                  thump_max_hz: float = 150.0, thump_min_rise_db: float = 5.0) -> str:
    """Scan a range for candidate pedal/bench thumps and clicks, with exact times.

    Thumps: low-frequency jumps with no matching jump in the mids (so not a note being struck).
    Clicks: impulses shorter than 3 ms. These are candidates, so confirm with view before fixing.
    Lower click_sensitivity (7) or thump_min_rise_db (3) to find fainter ones.
    """
    return _text(session.find_problems(start, end, version, click_sensitivity, thump_max_hz, thump_min_rise_db))


# --------------------------------------------------------------------- fixes


@tool
def reduce_mud(start: Time, end: Time, amount_db: float = 4.0, low_hz: float = 150.0,
               high_hz: float = 450.0, dynamic: bool = True) -> str:
    """Clean up a muddy or boomy passage by turning down a low band, with smooth shoulders.

    dynamic=True cuts only as much as the band is built up at each moment (full cut where low
    notes pile up, little where it is already clear), which keeps the piano's warmth. Typical:
    3-6 dB. Mud usually sits around 150-450 Hz, boom 60-150 Hz, boxiness 400-800 Hz; use analyze
    to see which band is actually heavy before choosing. Creates a new version.
    """
    return _text(session.reduce_mud(start, end, amount_db, low_hz, high_hz, dynamic))


@tool
def eq(start: Time, end: Time, bands: list[dict]) -> str:
    """Parametric EQ over a passage (zero phase). Creates a new version.

    bands: list of {"type": "bell"|"low_shelf"|"high_shelf"|"highpass"|"lowpass", "freq": Hz,
    "gain_db": dB, "q": width}. Example: [{"type": "bell", "freq": 280, "gain_db": -3, "q": 1.2},
    {"type": "high_shelf", "freq": 6000, "gain_db": 1.5}]. gain_db is ignored for the two pass types.
    """
    return _text(session.eq(start, end, bands))


@tool
def gain(start: Time, end: Time, gain_db: float) -> str:
    """Make a passage louder or quieter by gain_db, with smooth edges. Creates a new version."""
    return _text(session.gain(start, end, gain_db))


@tool
def remove_thump(time: Time, width_ms: float = 140.0, max_hz: float = 150.0, strength: float = 1.0) -> str:
    """Remove a pedal thump or other low knock at a given time. Creates a new version.

    Everything below max_hz around that moment is pulled down to the level it has just before and
    after, so bass notes ringing through are kept and only the bump goes. Get the time from
    find_problems or a zoomed view. Raise max_hz if the picture shows the thump reaching higher.
    """
    return _text(session.remove_thump(time, width_ms, max_hz, strength))


@tool
def remove_clicks(start: Time, end: Time, sensitivity: float = 10.0, times: list[Time] | None = None,
                  width_ms: float = 1.5) -> str:
    """Remove clicks and ticks in a range by re-synthesising a few milliseconds each.

    By default it finds them itself (lower sensitivity = more aggressive, 7 is strong). Pass
    times to repair specific instants instead, each width_ms wide. Creates a new version.
    """
    return _text(session.remove_clicks(start, end, sensitivity, times, width_ms))


@tool
def spectral_repair(start: Time, end: Time, low_hz: float = 20.0, high_hz: float = 20000.0,
                    mode: str = "attenuate", strength: float = 1.0) -> str:
    """"Paint away" a noise: rebuild one time/frequency box from the audio just before and after.

    For squeaks, creaks, hammer or action noise, a cough, a chair, anything you can see as a blob
    in view. Give the tightest box that covers it. mode="attenuate" only turns things down to the
    surrounding level (safe, keeps notes that ring through); "replace" overwrites the box entirely.
    Up to 5 s; works best under about 1 s. Creates a new version.
    """
    return _text(session.spectral_repair(start, end, low_hz, high_hz, mode, strength))


@tool
def denoise(start: Time, end: Time, amount_db: float = 8.0, noise_start: Time | None = None,
            noise_end: Time | None = None, sensitivity: float = 1.5) -> str:
    """Reduce steady background noise (hiss, room tone, felt/action rustle) by up to amount_db.

    Best with a noise sample: noise_start/noise_end marking at least 0.15 s where nothing is
    playing (before the first note, or a long rest). Keep amount_db modest (6-10); too much dulls
    the felt tone. Creates a new version.
    """
    return _text(session.denoise(start, end, amount_db, noise_start, noise_end, sensitivity))


@tool
def inpaint(start: Time, end: Time, backend: str = "auto") -> str:
    """Regenerate a damaged stretch entirely from the clean audio on both sides of it.

    Built in: "ar" (up to ~30 ms) and "bridge" (up to ~400 ms, sustained notes only; it cannot
    invent a new note attack). "external" runs a generative model if one is configured. "auto"
    picks. For anything longer without a model, prefer spectral_repair on the offending
    frequencies or copy_from_region. Creates a new version.
    """
    return _text(session.inpaint(start, end, backend))


@tool
def copy_from_region(source_start: Time, dest_start: Time, length: float, align: bool = True) -> str:
    """Replace a damaged passage with the same passage played elsewhere (a repeat, another bar).

    Copies `length` seconds starting at source_start over the audio at dest_start, fine-aligned
    and crossfaded. Only sensible when the music is the same in both places. Creates a new version.
    """
    return _text(session.copy_from_region(source_start, dest_start, length, align))


# ------------------------------------------------------------------ listening


@tool
def listen(start: Time, end: Time, a: int | None = None, b: int | None = None, preroll: float = 3.0) -> str:
    """Open an A/B listening page in Daniel's browser for a region, with a few seconds of lead-in.

    Defaults: B is the current version, A is the one before it. Both play in sync and he switches
    instantly with the A and B keys. Call this after a fix so he can judge it.
    """
    return _text(session.listen(start, end, a, b, preroll))


@tool
def export(path: str | None = None, version: int | None = None, bit_format: str | None = None) -> str:
    """Write a version to a new WAV file (never over the source). Defaults: current version,
    saved beside the source as "<name>_v<N>.wav" in the original bit depth. bit_format can be
    PCM_16, PCM_24 or FLOAT."""
    return _text(session.export(path, version, bit_format))


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
