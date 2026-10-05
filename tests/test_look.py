"""Tests for the look (aim-and-hold) feature of BehaviorEngine."""

from __future__ import annotations

import random

import pytest

from spookyeyes.behavior import BehaviorEngine, look_presets
from spookyeyes.model import Event, Mode, MotionParams

FAST = dict(saccade_interval=(0.2, 0.4), saccade_duration=(0.05, 0.1), wander=0.8)


def make_engine(seed: int = 7, presets=None, **kw) -> BehaviorEngine:
    return BehaviorEngine(MotionParams(**{**FAST, **kw}), random.Random(seed), presets=presets)


def run(eng: BehaviorEngine, seconds: float, dt: float = 1 / 60):
    left = right = None
    for _ in range(int(seconds / dt)):
        left, right = eng.step(dt)
    return left, right


def gaze(eye) -> tuple[float, float]:
    return (round(eye.gaze_x, 3), round(eye.gaze_y, 3))


# --- presets -------------------------------------------------------------------


def test_presets_visitor_pov_right_is_positive_x() -> None:
    p = look_presets(0.5, (0.6, -0.2))
    assert p["right"] == (0.5, 0.0) and p["left"] == (-0.5, 0.0)
    assert p["up"] == (0.0, 0.5) and p["down"] == (0.0, -0.5)
    assert p["center"] == (0.0, 0.0)
    assert p["doorbell"] == (0.6, -0.2)


def test_presets_are_clamped() -> None:
    p = look_presets(3.0, (2.0, -9.0))
    assert p["right"] == (1.0, 0.0)
    assert p["doorbell"] == (1.0, -1.0)


# --- idle: hold and release ----------------------------------------------------


def test_look_in_idle_moves_both_eyes_and_holds() -> None:
    eng = make_engine()
    eng.handle(Event("look", "right"))
    assert eng.look == "right"
    for _ in range(600):  # 10 s, many wander intervals
        left, right = eng.step(1 / 60)
        if _ > 30:  # after the saccade settled
            assert abs(left.gaze_x - 0.7) < 0.01, "left eye drifted off the look"
            assert abs(right.gaze_x - 0.7) < 0.01, "right eye drifted off the look"
            assert abs(left.gaze_y) < 0.01


def test_center_releases_wander() -> None:
    eng = make_engine()
    eng.handle(Event("look", "down"))
    run(eng, 1.0)
    eng.handle(Event("look", "center"))
    assert eng.look == "center"
    moved = False
    for _ in range(600):
        left, _r = eng.step(1 / 60)
        if _ > 60 and abs(left.gaze_x) > 0.05:
            moved = True
            break
    assert moved, "wander did not resume after center"


def test_look_uses_saccade_not_teleport() -> None:
    eng = make_engine(saccade_duration=(0.2, 0.2))
    eng.handle(Event("look", "left"))
    left, _ = eng.step(0.05)
    assert -0.7 < left.gaze_x < 0.0  # in flight


# --- stare + look --------------------------------------------------------------


def test_stare_plus_doorbell_aims_at_doorbell() -> None:
    eng = make_engine(presets={"doorbell": (0.55, -0.25)})
    eng.handle(Event("mode", "stare"))
    eng.handle(Event("look", "doorbell"))
    left, right = run(eng, 2.0)
    assert gaze(left) == (0.55, -0.25)
    assert gaze(right) == (0.55, -0.25)
    assert eng.look == "doorbell" and eng.mode is Mode.STARE


def test_look_then_stare_keeps_look_offset() -> None:
    eng = make_engine()
    eng.handle(Event("look", "up"))
    eng.handle(Event("mode", "stare"))
    left, _ = run(eng, 2.0)
    assert gaze(left) == (0.0, 0.7)


def test_stare_center_returns_to_origin() -> None:
    eng = make_engine()
    eng.handle(Event("mode", "stare"))
    eng.handle(Event("look", "right"))
    run(eng, 1.0)
    eng.handle(Event("look", "center"))
    left, _ = run(eng, 1.0)
    assert gaze(left) == (0.0, 0.0)


# --- resets --------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["scare", "sleep"])
def test_mode_change_to_scare_or_sleep_resets_look(mode: str) -> None:
    eng = make_engine()
    eng.handle(Event("look", "doorbell"))
    eng.handle(Event("mode", mode))
    assert eng.look == "center"
    assert eng.look_xy == (0.0, 0.0)


def test_pir_motion_scare_resets_look() -> None:
    eng = make_engine()
    eng.handle(Event("look", "left"))
    eng.handle(Event("motion"))
    assert eng.mode is Mode.SCARE
    assert eng.look == "center"


def test_scare_auto_return_leaves_look_centered() -> None:
    eng = make_engine()
    eng.handle(Event("look", "left"))
    eng.handle(Event("mode", "scare"))
    run(eng, 7.0)
    assert eng.mode is Mode.IDLE
    assert eng.look == "center"


def test_look_while_asleep_is_honoured_on_waking() -> None:
    eng = make_engine()
    eng.handle(Event("mode", "sleep"))
    eng.handle(Event("look", "right"))
    assert eng.look == "right"
    eng.handle(Event("mode", "idle"))
    left, _ = run(eng, 2.0)
    assert gaze(left) == (0.7, 0.0)


def test_stare_does_not_reset_look() -> None:
    eng = make_engine()
    eng.handle(Event("look", "doorbell"))
    eng.handle(Event("mode", "stare"))
    assert eng.look == "doorbell"


# --- continuous aiming ---------------------------------------------------------


def test_xy_tuple_maps_to_nearest_option() -> None:
    eng = make_engine()
    eng.handle(Event("look", (0.65, 0.05)))
    assert eng.look == "right"
    assert eng.look_xy == (0.65, 0.05)
    eng.handle(Event("look", {"x": 0.0, "y": 0.1}))
    assert eng.look == "center"
    eng.handle(Event("look", (0.7, -0.3)))
    assert eng.look == "doorbell"


def test_xy_clamped_and_applied() -> None:
    eng = make_engine()
    eng.handle(Event("look", (4.0, -4.0)))
    assert eng.look_xy == (1.0, -1.0)
    left, _ = run(eng, 2.0)
    assert gaze(left) == (1.0, -1.0)


@pytest.mark.parametrize(
    "bad", ["sideways", "", None, 3, (1.0,), ("a", "b"), {"x": 1}, (float("nan"), 0.0)]
)
def test_invalid_look_ignored(bad) -> None:
    eng = make_engine()
    eng.handle(Event("look", "up"))
    eng.handle(Event("look", bad))
    assert eng.look == "up"
    assert eng.look_xy == (0.0, 0.7)


def test_look_is_deterministic() -> None:
    a, b = make_engine(seed=3), make_engine(seed=3)
    for i in range(300):
        if i == 20:
            a.handle(Event("look", "right")); b.handle(Event("look", "right"))
        if i == 120:
            a.handle(Event("look", "center")); b.handle(Event("look", "center"))
        assert a.step(1 / 60) == b.step(1 / 60)
