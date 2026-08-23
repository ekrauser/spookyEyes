"""Shared value types — the contract between behavior, renderer, inputs, and app.

Every module communicates through these types only. See DESIGN.md for the full
architecture and per-module responsibilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

SIZE = 240  # panel resolution: 240x240 (GC9A01A round TFT)


class Mode(str, Enum):
    IDLE = "idle"    # autonomous wander / blink / dilation
    SCARE = "scare"  # startled: wide lids, constricted pupils, fast saccades; auto-returns to IDLE
    STARE = "stare"  # locked forward, rare blinks
    SLEEP = "sleep"  # lids closed, occasional twitch


@dataclass
class EyeState:
    """Renderable state of one eye at one instant. Produced by BehaviorEngine,
    consumed by EyeRenderer. All fields are already clamped to their ranges."""

    gaze_x: float = 0.0      # -1..1, positive = viewer's right
    gaze_y: float = 0.0      # -1..1, positive = up
    pupil: float = 0.5       # 0 = fully constricted .. 1 = fully dilated
    openness: float = 1.0    # 0 = closed, 1 = relaxed open, up to 1.25 = startled-wide
    brightness: float = 1.0  # 0..1 software dim applied to the final frame


@dataclass
class Event:
    """Input event, queued by inputs/* threads, drained by the app main loop.

    kind        value
    ----        -----
    "mode"      Mode or mode string ("idle" | "scare" | "stare" | "sleep")
    "theme"     theme name string (handled by the app: reloads renderers + motion)
    "brightness" float 0..1
    "blink"     None (blink once now)
    "motion"    None (PIR edge; behavior decides whether to startle)
    "quit"      None (shut down cleanly)
    """

    kind: str
    value: object = None


@dataclass
class MotionParams:
    """Per-theme animation tuning. Parsed from theme.json "motion" by themes.py,
    consumed by BehaviorEngine (via set_motion on theme switch)."""

    saccade_interval: tuple[float, float] = (0.6, 3.0)   # s between gaze jumps (uniform)
    saccade_duration: tuple[float, float] = (0.09, 0.22)  # s per gaze jump (uniform)
    wander: float = 0.35        # 0..1 amplitude of gaze targets around center
    crazy: float = 0.1          # 0 = conjugate eyes .. 1 = fully independent
    blink_interval: tuple[float, float] = (2.0, 8.0)      # s between blinks (uniform)
    blink_speed: float = 1.0    # multiplier on lid close/open speed
    pupil_speed: float = 0.5    # rate of autonomous pupil drift
    drift: float = 0.0          # continuous slow sinusoidal gaze drift (ghost themes)
    flicker: float = 0.0        # 0..1 brightness flicker amplitude (ghost themes)
