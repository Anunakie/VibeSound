"""A tiny self-contained A/B listening page, until the desktop UI exists.

Both clips play in sync and one is muted, so switching is instant and gapless.
"""

from __future__ import annotations

import base64
import html
import json

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>__TITLE__</title>
<style>
  :root { color-scheme: dark; }
  body { background:#14161a; color:#e8e6e1; font:15px/1.5 "Segoe UI",system-ui,sans-serif; margin:0; padding:24px; }
  h1 { font-size:18px; margin:0 0 4px; font-weight:600; }
  p.sub { margin:0 0 18px; color:#9aa0a8; }
  .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:14px; }
  button { font:inherit; color:inherit; background:#22262d; border:1px solid #3a404a; border-radius:8px;
           padding:10px 18px; cursor:pointer; min-width:64px; }
  button:hover { background:#2b3039; }
  button.on { background:#e0a43a; border-color:#e0a43a; color:#14161a; font-weight:600; }
  button:focus-visible { outline:2px solid #7cc4ff; outline-offset:2px; }
  #time { font-variant-numeric:tabular-nums; color:#9aa0a8; margin-left:6px; }
  #pic { border-radius:8px; overflow:hidden; background:#fff; }
  #pic img { display:block; width:100%; height:auto; }
  input[type=range] { flex:1; min-width:200px; accent-color:#e0a43a; }
  kbd { background:#22262d; border:1px solid #3a404a; border-radius:4px; padding:1px 6px; font:13px ui-monospace,Consolas,monospace; }
  .keys { color:#9aa0a8; font-size:13px; margin-top:14px; }
</style></head><body>
<h1>__TITLE__</h1>
<p class="sub">__SUB__</p>
<div class="row">
  <button id="play">Play</button>
  <span id="choices"></span>
  <label><input type="checkbox" id="loop" checked> loop</label>
  <span id="time">0:00.0</span>
</div>
<div class="row"><input type="range" id="seek" min="0" max="1000" value="0" aria-label="position"></div>
__IMAGE__
<p class="keys"><kbd>Space</kbd> play or pause &nbsp; <kbd>A</kbd> / <kbd>B</kbd> or <kbd>Tab</kbd> switch version &nbsp; <kbd>&larr;</kbd> <kbd>&rarr;</kbd> skip 2 s</p>
<script>
const clips = __CLIPS__, offset = __OFFSET__;
const audios = clips.map(c => { const a = new Audio(c.file); a.preload = "auto"; return a; });
let current = clips.length - 1;
const $ = id => document.getElementById(id);
const choices = $("choices");
clips.forEach((c, i) => {
  const b = document.createElement("button");
  b.textContent = c.key + ": " + c.label; b.onclick = () => pick(i); choices.appendChild(b);
  choices.appendChild(document.createTextNode(" "));
});
function pick(i) {
  current = i;
  audios.forEach((a, j) => { a.muted = j !== i; });
  [...choices.querySelectorAll("button")].forEach((b, j) => b.classList.toggle("on", j === i));
}
function playing() { return !audios[0].paused; }
function toggle() {
  if (playing()) audios.forEach(a => a.pause());
  else { const t = audios[0].currentTime; audios.forEach(a => { a.currentTime = t; a.play(); }); }
  $("play").textContent = playing() ? "Pause" : "Play";
}
function seek(t) { t = Math.max(0, Math.min(t, audios[0].duration || 0)); audios.forEach(a => a.currentTime = t); }
function fmt(t) { t += offset; const m = Math.floor(t / 60); return m + ":" + (t - m * 60).toFixed(1).padStart(4, "0"); }
$("play").onclick = toggle;
$("seek").oninput = e => seek(e.target.value / 1000 * (audios[0].duration || 0));
audios[0].addEventListener("ended", () => {
  if ($("loop").checked) { audios.forEach(a => { a.currentTime = 0; a.play(); }); }
  else $("play").textContent = "Play";
});
setInterval(() => {
  const a = audios[0], d = a.duration || 0;
  if (playing()) for (const o of audios.slice(1)) if (Math.abs(o.currentTime - a.currentTime) > 0.03) o.currentTime = a.currentTime;
  $("time").textContent = fmt(a.currentTime) + " / " + fmt(d);
  if (document.activeElement !== $("seek")) $("seek").value = d ? a.currentTime / d * 1000 : 0;
}, 50);
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" && e.target.type !== "range" && e.target.type !== "checkbox") return;
  const k = e.key.toLowerCase();
  if (k === " ") { e.preventDefault(); toggle(); }
  else if (k === "tab") { e.preventDefault(); pick((current + 1) % clips.length); }
  else if (k === "arrowleft") seek(audios[0].currentTime - 2);
  else if (k === "arrowright") seek(audios[0].currentTime + 2);
  else { const i = clips.findIndex(c => c.key.toLowerCase() === k); if (i >= 0) pick(i); }
});
pick(current);
</script></body></html>
"""


def build(title: str, subtitle: str, clips: list[dict], offset_s: float, image_bytes: bytes | None) -> str:
    """``clips`` are dicts with key ("A"), label and file (relative path to a WAV)."""
    image = ""
    if image_bytes:
        data = base64.b64encode(image_bytes).decode("ascii")
        image = f'<div id="pic"><img alt="Spectrogram of the clips" src="data:image/jpeg;base64,{data}"></div>'
    return (
        PAGE.replace("__TITLE__", html.escape(title))
        .replace("__SUB__", html.escape(subtitle))
        .replace("__IMAGE__", image)
        .replace("__CLIPS__", json.dumps(clips).replace("</", "<\\/"))
        .replace("__OFFSET__", repr(float(offset_s)))
    )
