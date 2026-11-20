"""Parsing and formatting of times the way a person says them ("2:15", "1:02.5", 135)."""

from __future__ import annotations


def parse_time(value: str | float | int) -> float:
    """Return seconds for "m:ss", "m:ss.s", "h:mm:ss", "12.5", "12.5s" or a number."""
    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        text = str(value).strip().lower().removesuffix("s").strip()
        if not text:
            raise ValueError("empty time")
        parts = text.split(":")
        if len(parts) > 3:
            raise ValueError(f"cannot read time {value!r}")
        try:
            seconds = 0.0
            for part in parts:
                seconds = seconds * 60.0 + float(part)
        except ValueError:
            raise ValueError(f"cannot read time {value!r}; use m:ss or seconds") from None
    if seconds < 0:
        raise ValueError(f"time {value!r} is negative")
    return seconds


def fmt_time(seconds: float) -> str:
    """Format seconds as m:ss.mmm (the form Daniel reads off a player)."""
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    if rest >= 59.9995:
        minutes, rest = minutes + 1, 0.0
    return f"{minutes}:{rest:06.3f}"
