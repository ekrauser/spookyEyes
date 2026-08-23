"""Tests for spookyeyes.behavior.BehaviorEngine (agent BEHAVIOR).

Self-contained: builds MotionParams directly, no theme assets needed.
"""

from __future__ import annotations

import math
import random

import pytest

from spookyeyes.behavior import BehaviorEngine
from spookyeyes.model import Event, EyeState, Mode, MotionParams


def make_engine(seed: int = 1234, **kw: object) -> BehaviorEngine:
    return BehaviorEngine(MotionParams(**kw), random.Random(seed))  # type: ignore[arg-type]


def gaze_dist(a: EyeState, b: EyeState) -> float:
    return math.hypot(a.gaze_x - b.gaze_x, a.gaze_y - b.gaze_y)


STRESS = dict(
    saccade_interval=(0.05, 0.3),
    saccade_duration=(0.05, 0.1),
    wander=1.0,
    crazy=1.0,
    blink_interval=(0.3, 1.0),
    blink_speed=2.0,
    pupil_speed=3.0,
    drift=1.0,
    flicker=0.9,
)


# --- determinism ---------------------------------------------------------------


def test_determinism_same_seed_same_dts_same_events() -> None:
    events = {
        40: Event("mode", "stare"),
        110: Event("mode", Mode.IDLE),
        170: Event("motion"),
        250: Event("blink"),
        300: Event("brightness", 0.5),
        380: Event("mode", "sleep"),
        450: Event("mode", "idle"),
    }
    a = make_engine(seed=77, drift=0.4, flicker=0.2, crazy=0.5)
    b = make_engine(seed=77, drift=0.4, flicker=0.2, crazy=0.5)
    dt_rng = random.Random(5)
    dts = [dt_rng.uniform(0.005, 0.05) for _ in range(500)]
    for i, dt in enumerate(dts):
        if i in events:
            a.handle(events[i])
            b.handle(events[i])
        sa = a.step(dt)
        sb = b.step(dt)
        assert sa == sb, f"states diverged at step {i}"
        assert a.mode is b.mode


# --- clamping ------------------------------------------------------------------


@pytest.mark.parametrize("mode", list(Mode))
def test_ranges_always_clamped_2000_random_dt_steps(mode: Mode) -> None:
    eng = make_engine(seed=42, **STRESS)
    eng.handle(Event("mode", mode.value))
    dt_rng = random.Random(999)
    for i in range(2000):
        left, right = eng.step(dt_rng.uniform(0.001, 0.08))
        for eye in (left, right):
            assert -1.0 <= eye.gaze_x <= 1.0, f"gaze_x out of range at step {i}"
            assert -1.0 <= eye.gaze_y <= 1.0, f"gaze_y out of range at step {i}"
            assert 0.0 <= eye.pupil <= 1.0, f"pupil out of range at step {i}"
            assert 0.0 <= eye.openness <= 1.25, f"openness out of range at step {i}"
            assert 0.0 <= eye.brightness <= 1.0, f"brightness out of range at step {i}"


# --- blinking ------------------------------------------------------------------


def test_blink_occurs_within_max_interval_and_dips_low() -> None:
    eng = make_engine(seed=3)
    interval_max = eng.motion.blink_interval[1]
    dt = 0.01
    min_open = 1.0
    max_open = 0.0
    first_dip_t: float | None = None
    steps = int((interval_max + 1.5) / dt)
    for i in range(steps):
        left, _ = eng.step(dt)
        min_open = min(min_open, left.openness)
        max_open = max(max_open, left.openness)
        if first_dip_t is None and left.openness < 0.5:
            first_dip_t = (i + 1) * dt
    assert min_open < 0.15, "blink never closed the lids far enough"
    assert max_open > 0.9, "eyes never settled open"
    assert first_dip_t is not None, "no blink observed at all"
    assert first_dip_t <= interval_max + 0.2, "first blink came later than max interval"


def test_blink_event_triggers_immediate_blink() -> None:
    eng = make_engine(seed=8, blink_interval=(50.0, 60.0))
    for _ in range(50):
        eng.step(0.01)
    eng.handle(Event("blink"))
    min_open = 1.0
    for _ in range(40):  # 0.4 s covers close (0.08 s) + open (0.15 s)
        left, _ = eng.step(0.01)
        min_open = min(min_open, left.openness)
    assert min_open < 0.15


# --- SCARE ---------------------------------------------------------------------


def test_scare_widens_lids_constricts_pupil_then_returns_to_idle() -> None:
    eng = make_engine(seed=11)
    for _ in range(50):  # settle in IDLE
        eng.step(0.02)
    eng.handle(Event("motion"))
    assert eng.mode is Mode.SCARE

    dt = 0.02
    max_open = 0.0
    min_pupil = 1.0
    for _ in range(50):  # first second of the scare
        left, _ = eng.step(dt)
        max_open = max(max_open, left.openness)
        min_pupil = min(min_pupil, left.pupil)
    assert max_open > 1.15, "SCARE did not widen the lids"
    assert min_pupil < 0.2, "SCARE did not constrict the pupil"

    elapsed = 1.0
    while eng.mode is Mode.SCARE and elapsed < 10.0:
        eng.step(dt)
        elapsed += dt
    assert eng.mode is Mode.IDLE
    assert elapsed <= 8.0, f"SCARE did not auto-return within 8 s (took {elapsed:.2f})"


def test_scare_retrigger_restarts_timer() -> None:
    eng = make_engine(seed=12)
    eng.handle(Event("motion"))
    dt = 0.02
    for _ in range(int(4.0 / dt)):  # 4 s into the scare
        eng.step(dt)
    eng.handle(Event("motion"))  # re-trigger: timer restarts at ~6 s
    assert eng.mode is Mode.SCARE
    for _ in range(int(3.0 / dt)):  # t = 7 s > original 6 s deadline
        eng.step(dt)
    assert eng.mode is Mode.SCARE, "re-trigger did not restart the auto-return timer"
    elapsed = 0.0
    while eng.mode is Mode.SCARE and elapsed < 10.0:
        eng.step(dt)
        elapsed += dt
    assert eng.mode is Mode.IDLE


def test_motion_enters_scare_from_idle_only() -> None:
    eng = make_engine(seed=13)
    assert eng.mode is Mode.IDLE
    eng.handle(Event("motion"))
    assert eng.mode is Mode.SCARE

    asleep = make_engine(seed=13)
    asleep.handle(Event("mode", "sleep"))
    for _ in range(50):
        asleep.step(0.02)
    asleep.handle(Event("motion"))
    assert asleep.mode is Mode.SLEEP, "motion must be ignored while sleeping"

    staring = make_engine(seed=13)
    staring.handle(Event("mode", "stare"))
    staring.handle(Event("motion"))
    assert staring.mode is Mode.STARE, "motion must be ignored while staring"


# --- STARE ---------------------------------------------------------------------


def test_stare_converges_gaze_to_center() -> None:
    eng = make_engine(seed=21, wander=0.6, crazy=0.4)
    for _ in range(100):  # wander away from center first
        eng.step(0.02)
    eng.handle(Event("mode", "stare"))
    states: list[tuple[EyeState, EyeState]] = []
    for _ in range(150):  # 3 s
        states.append(eng.step(0.02))
    for left, right in states[-20:]:
        for eye in (left, right):
            assert abs(eye.gaze_x) < 0.02
            assert abs(eye.gaze_y) < 0.02


# --- SLEEP ---------------------------------------------------------------------


def test_sleep_closes_then_idle_reopens() -> None:
    eng = make_engine(seed=31)
    eng.handle(Event("mode", "sleep"))
    left, _ = eng.step(0.02)
    for _ in range(200):  # 4 s, before the first twitch (>= 8 s)
        left, _ = eng.step(0.02)
    assert left.openness < 0.05, "SLEEP did not close the lids"

    eng.handle(Event("mode", "idle"))
    max_open = 0.0
    for _ in range(150):  # 3 s
        left, _ = eng.step(0.02)
        max_open = max(max_open, left.openness)
    assert max_open > 0.9, "eyes did not reopen after waking"


def test_sleep_twitches_but_stays_nearly_closed() -> None:
    eng = make_engine(seed=32)
    eng.handle(Event("mode", "sleep"))
    for _ in range(150):  # 3 s: converge closed
        eng.step(0.02)
    max_open = 0.0
    for _ in range(650):  # 13 s: covers a twitch (every ~8-12 s)
        left, _ = eng.step(0.02)
        max_open = max(max_open, left.openness)
    assert max_open > 0.03, "no sleep twitch observed"
    assert max_open <= 0.2, "sleep twitch opened the lids too far"


# --- crazy factor --------------------------------------------------------------


def test_crazy_zero_gives_identical_eyes() -> None:
    eng = make_engine(seed=41, crazy=0.0, wander=0.5, drift=0.3)
    for _ in range(1500):
        left, right = eng.step(0.02)
        assert left.gaze_x == right.gaze_x
        assert left.gaze_y == right.gaze_y


def test_crazy_one_gives_divergent_eyes() -> None:
    eng = make_engine(seed=41, crazy=1.0, wander=0.5)
    max_diff = 0.0
    for _ in range(1500):
        left, right = eng.step(0.02)
        max_diff = max(max_diff, gaze_dist(left, right))
    assert max_diff > 0.05, "crazy=1 eyes never measurably diverged"


# --- brightness + flicker ------------------------------------------------------


def test_brightness_event_applies_and_clamps() -> None:
    eng = make_engine(seed=51, flicker=0.0)
    eng.handle(Event("brightness", 0.4))
    left, right = eng.step(0.02)
    assert left.brightness == pytest.approx(0.4)
    assert right.brightness == pytest.approx(0.4)

    eng.handle(Event("brightness", 1.5))
    left, _ = eng.step(0.02)
    assert left.brightness == pytest.approx(1.0)

    eng.handle(Event("brightness", -2.0))
    left, _ = eng.step(0.02)
    assert left.brightness == pytest.approx(0.0)


def test_flicker_stays_within_band() -> None:
    flicker = 0.3
    eng = make_engine(seed=52, flicker=flicker)
    lo = 1.0
    hi = 0.0
    for _ in range(800):
        left, _ = eng.step(0.02)
        lo = min(lo, left.brightness)
        hi = max(hi, left.brightness)
    assert hi <= 1.0
    assert lo >= 1.0 - flicker - 1e-9
    assert lo < 0.85, "flicker never actually dimmed the frame"


# --- set_motion ----------------------------------------------------------------


def test_set_motion_keeps_pose_no_gaze_jump() -> None:
    slow = dict(
        saccade_interval=(0.3, 0.8),
        saccade_duration=(0.5, 0.8),  # slow saccades -> small per-step motion
        wander=0.35,
        crazy=0.0,
        drift=0.0,
        flicker=0.0,
    )
    eng = make_engine(seed=61, **slow)
    dt = 0.01
    prev_l, prev_r = eng.step(dt)
    for _ in range(299):
        left, right = eng.step(dt)
        assert gaze_dist(prev_l, left) < 0.12
        assert gaze_dist(prev_r, right) < 0.12
        prev_l, prev_r = left, right

    eng.set_motion(
        MotionParams(
            saccade_interval=(0.1, 0.25),
            saccade_duration=(0.5, 0.8),
            wander=0.1,
            crazy=0.3,
            drift=1.0,
            flicker=0.5,
        )
    )
    left, right = eng.step(dt)  # the step straddling the theme switch
    assert gaze_dist(prev_l, left) < 0.06, "set_motion caused a gaze jump"
    assert gaze_dist(prev_r, right) < 0.06, "set_motion caused a gaze jump"
    prev_l, prev_r = left, right
    for _ in range(300):
        left, right = eng.step(dt)
        assert gaze_dist(prev_l, left) < 0.12
        assert gaze_dist(prev_r, right) < 0.12
        prev_l, prev_r = left, right


# --- robustness ----------------------------------------------------------------


def test_invalid_events_are_ignored_without_raising() -> None:
    eng = make_engine(seed=71)
    eng.handle(Event("mode", "bogus"))
    assert eng.mode is Mode.IDLE
    eng.handle(Event("mode", 42))
    assert eng.mode is Mode.IDLE
    eng.handle(Event("brightness", "not-a-float"))
    eng.handle(Event("brightness", None))
    eng.handle(Event("frobnicate", object()))
    left, right = eng.step(0.02)
    assert left.brightness == pytest.approx(1.0)
    assert isinstance(right, EyeState)


def test_mode_event_accepts_enum_and_padded_strings() -> None:
    eng = make_engine(seed=72)
    eng.handle(Event("mode", Mode.STARE))
    assert eng.mode is Mode.STARE
    eng.handle(Event("mode", "  SLEEP \n"))
    assert eng.mode is Mode.SLEEP
    eng.handle(Event("mode", "idle"))
    assert eng.mode is Mode.IDLE


def test_zero_and_huge_dt_do_not_break_state() -> None:
    eng = make_engine(seed=73, **STRESS)
    for dt in (0.0, 100.0, 0.0, 5.0, 0.016):
        left, right = eng.step(dt)
        for eye in (left, right):
            assert -1.0 <= eye.gaze_x <= 1.0
            assert -1.0 <= eye.gaze_y <= 1.0
            assert 0.0 <= eye.pupil <= 1.0
            assert 0.0 <= eye.openness <= 1.25
            assert 0.0 <= eye.brightness <= 1.0
