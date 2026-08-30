"""
time_policy.py — Configurable work-schedule and late-arrival engine.

Classifies attendance marks as:
    on_time   — arrived at or before WORK_START
    late      — arrived within grace period (WORK_START+1 … WORK_START+GRACE_MINUTES)
    very_late — arrived after grace period ends
    absent    — no attendance recorded by WORK_END

Configuration is loaded from config.json if present, falling back to defaults.
The Admin dashboard writes changes back to config.json.
"""

import json
import os
from datetime import date, datetime, time, timedelta

_BASE        = os.path.dirname(__file__)
CONFIG_PATH  = os.path.join(_BASE, "config.json")

# ── Defaults ──────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "work_start":    "09:00",   # HH:MM — on-time threshold
    "grace_minutes": 15,        # minutes after work_start = late (not very_late)
    "work_end":      "17:00",   # HH:MM — standard end-of-day (for absent calc)
    "work_end_ext":  "18:00",   # HH:MM — extended end-of-day (optional toggle)
    "use_extended":  False,     # True → use work_end_ext for absent calculation
}


# ── Config I/O ────────────────────────────────────────────────────────────────

def load_config() -> dict:
    """Load config from JSON file, filling gaps with defaults."""
    cfg = dict(_DEFAULTS)
    if os.path.isfile(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                cfg.update(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass
    return cfg


def save_config(updates: dict) -> None:
    """Persist config updates to config.json (merges with existing)."""
    cfg = load_config()
    cfg.update(updates)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


# ── Time helpers ──────────────────────────────────────────────────────────────

def _parse_hhmm(s: str) -> time:
    """Parse 'HH:MM' string → datetime.time."""
    h, m = map(int, s.split(":"))
    return time(h, m)


def _time_from_str(s: str) -> time:
    """Parse 'HH:MM:SS' or 'HH:MM' string → datetime.time."""
    parts = s.split(":")
    return time(int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0)


# ── Core classification ───────────────────────────────────────────────────────

def classify_arrival(arrival_time_str: str, cfg: dict | None = None) -> tuple[str, int]:
    """
    Classify an arrival time string ('HH:MM:SS' or 'HH:MM').

    Returns (status: str, late_minutes: int) where:
        status        — 'on_time' | 'late' | 'very_late'
        late_minutes  — minutes after WORK_START (0 if on-time)

    Config used if provided, else loaded from file.
    """
    if cfg is None:
        cfg = load_config()

    work_start    = _parse_hhmm(cfg["work_start"])
    grace_minutes = int(cfg["grace_minutes"])
    grace_end     = (datetime.combine(date.today(), work_start)
                     + timedelta(minutes=grace_minutes)).time()

    arrival = _time_from_str(arrival_time_str)

    # Minutes after work_start (negative if early)
    start_dt   = datetime.combine(date.today(), work_start)
    arrival_dt = datetime.combine(date.today(), arrival)
    delta_mins = int((arrival_dt - start_dt).total_seconds() / 60)

    if arrival <= work_start:
        return "on_time", 0
    elif arrival <= grace_end:
        return "late", delta_mins
    else:
        return "very_late", delta_mins


def get_schedule_summary(cfg: dict | None = None) -> dict:
    """
    Return human-readable schedule info for the UI.
    e.g. { 'work_start': '09:00', 'grace_end': '09:15', 'work_end': '17:00', ... }
    """
    if cfg is None:
        cfg = load_config()
    work_start    = _parse_hhmm(cfg["work_start"])
    grace_minutes = int(cfg["grace_minutes"])
    grace_end     = (datetime.combine(date.today(), work_start)
                     + timedelta(minutes=grace_minutes)).time()
    work_end_key  = "work_end_ext" if cfg.get("use_extended") else "work_end"
    return {
        "work_start":    cfg["work_start"],
        "grace_minutes": grace_minutes,
        "grace_end":     grace_end.strftime("%H:%M"),
        "work_end":      cfg[work_end_key],
        "use_extended":  cfg.get("use_extended", False),
        "work_end_std":  cfg["work_end"],
        "work_end_ext":  cfg["work_end_ext"],
    }


def status_label(status: str) -> str:
    """Human-readable label for an attendance status."""
    return {
        "on_time":   "On Time",
        "late":      "Late",
        "very_late": "Very Late",
        "Present":   "On Time",   # legacy value compat
        "absent":    "Absent",
    }.get(status, status.replace("_", " ").title())


def status_badge_class(status: str) -> str:
    """CSS badge class for a status value."""
    return {
        "on_time":   "badge-green",
        "late":      "badge-amber",
        "very_late": "badge-red",
        "Present":   "badge-green",  # legacy
        "absent":    "badge-muted",
    }.get(status, "badge-muted")
