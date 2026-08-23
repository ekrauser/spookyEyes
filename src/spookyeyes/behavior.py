"""Behavior state machine: turns time + input events into per-eye EyeState.

Implements the DESIGN.md contract:

- IDLE: wandering gaze targets with ease-out-cubic saccades, blinks (10%
  double-blink), a bounded random-walk pupil drift, slow sinusoidal gaze drift,
  and brightness flicker.
- SCARE: lids snap wide (1.25), pupils constrict to ~0.1, gaze snaps to center
  then fires 2-3 rapid saccades; auto-returns to IDLE after ~6 s. Re-triggering
  restarts the timer.
- STARE: gaze eases to (0, 0) and holds; blink interval x4.
- SLEEP: lids ease closed; every ~10 s a small twitch; motion events ignored;
  on leaving, the eyes reopen smoothly.

All randomness flows through ``self.rng`` so a seeded engine stepped with a
fixed dt sequence is fully deterministic. Every field of the returned
``EyeState`` objects is clamped to its documented range.
"""

from __future__ import annotations

import logging
import math
import random

from .model import Event, EyeState, Mode, MotionParams

log = logging.getLogger("spookyeyes.behavior")

# --- tuning constants (seconds / dimensionless gaze units) ---------------------
SCARE_DURATION = 6.0        # auto-return to IDLE after this long in SCARE
SCARE_OPENNESS = 1.25       # startled-wide lid target
SCARE_PUPIL = 0.1           # constricted pupil target in SCARE
SCARE_OPEN_RATE = 18.0      # 1/s exponential rate for lids snapping wide
SCARE_PUPIL_RATE = 15.0     # 1/s exponential rate for pupil constriction
SCARE_SNAP_DUR = 0.06       # duration of the initial snap-to-center saccade
SCARE_SACCADE_DUR = (0.05, 0.09)    # rapid-burst saccade durations
SCARE_SACCADE_GAP = (0.15, 0.35)    # gap between rapid-burst saccades
SCARE_FIRST_GAP = (0.25, 0.5)       # gap between the snap and the first burst

BLINK_CLOSE_S = 0.08        # lid close time at blink_speed == 1
BLINK_OPEN_S = 0.15         # lid open time at blink_speed == 1
DOUBLE_BLINK_P = 0.1        # chance a scheduled blink is a double-blink
STARE_BLINK_MULT = 4.0      # blink interval multiplier while staring

OPEN_RATE = 6.0             # 1/s lid easing toward relaxed-open (IDLE/STARE)
SLEEP_CLOSE_RATE = 4.0      # 1/s lid easing toward closed in SLEEP
TWITCH_INTERVAL = (8.0, 12.0)   # s between sleep twitches (~10 s)
TWITCH_DUR = 0.35           # s, one half-sine lid twitch
TWITCH_AMP = 0.12           # peak extra openness of a sleep twitch

PUPIL_WALK_LO = 0.15        # bounds of the autonomous pupil random walk
PUPIL_WALK_HI = 0.85
PUPIL_WALK_GAIN = 3.0       # walk step scale on pupil_speed
PUPIL_EASE_RATE = 3.0       # 1/s pupil easing toward its walk target

DRIFT_GAIN = 0.12           # gaze units of sway per unit MotionParams.drift
DRIFT_EASE_RATE = 1.5       # 1/s easing of the applied drift amplitude
DRIFT_W1 = 0.9              # rad/s sinusoid frequencies (incommensurate)
DRIFT_W2 = 0.63

MAX_DT = 0.25               # clamp a stalled frame so state cannot teleport
MIN_SACCADE_DUR = 1e-3


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _ease(dt: float, rate: float) -> float:
    """Frame-rate-independent exponential easing coefficient in [0, 1]."""
    return 1.0 - math.exp(-rate * dt)


class BehaviorEngine:
    """Advances eye animation state; consumed by the app main loop.

    Construct with the theme's :class:`MotionParams`; pass a seeded
    ``random.Random`` for deterministic output (tests, demos, ``--seed``).
    """

    def __init__(self, motion: MotionParams, rng: random.Random | None = None) -> None:
        self.motion = motion
        self.rng = rng if rng is not None else random.Random()
        self._mode = Mode.IDLE
        self._brightness = 1.0

        # gaze: current per-eye position (drift excluded) + active saccade
        self._gaze_l: tuple[float, float] = (0.0, 0.0)
        self._gaze_r: tuple[float, float] = (0.0, 0.0)
        self._sacc_active = False
        self._sacc_from_l: tuple[float, float] = (0.0, 0.0)
        self._sacc_from_r: tuple[float, float] = (0.0, 0.0)
        self._sacc_to_l: tuple[float, float] = (0.0, 0.0)
        self._sacc_to_r: tuple[float, float] = (0.0, 0.0)
        self._sacc_t = 0.0
        self._sacc_dur = MIN_SACCADE_DUR
        self._sacc_timer = self._uniform(motion.saccade_interval)

        # blink envelope
        self._blink_timer = self._uniform(motion.blink_interval)
        self._blink_t: float | None = None
        self._blink_close = BLINK_CLOSE_S
        self._blink_open = BLINK_OPEN_S
        self._double_pending = False

        # pupil random walk
        self._pupil = 0.5
        self._pupil_target = 0.5

        # lids
        self._open_base = 1.0

        # sinusoidal drift
        self._drift_amp = 0.0
        self._drift_phase = 0.0
        self._drift_p0 = self.rng.uniform(0.0, math.tau)

        # sleep twitch
        self._twitch_timer = self._uniform(TWITCH_INTERVAL)
        self._twitch_t: float | None = None

        # scare
        self._scare_timer = 0.0
        self._scare_burst_left = 0
        self._scare_next = 0.0

    # -- public API ------------------------------------------------------------

    @property
    def mode(self) -> Mode:
        return self._mode

    def set_motion(self, motion: MotionParams) -> None:
        """Swap animation tuning (theme switch) without disturbing the pose.

        Current gaze/pupil/lid state and any in-flight saccade or blink keep
        running to completion; only future samples use the new parameters.
        Pending timers are capped to the new maxima so a short-interval theme
        takes effect promptly.
        """
        self.motion = motion
        self._sacc_timer = min(self._sacc_timer, max(motion.saccade_interval))
        blink_cap = max(motion.blink_interval)
        if self._mode is Mode.STARE:
            blink_cap *= STARE_BLINK_MULT
        self._blink_timer = min(self._blink_timer, blink_cap)

    def handle(self, event: Event) -> None:
        """Apply one input event. Never raises on malformed values."""
        kind = event.kind
        if kind == "mode":
            value = event.value
            if isinstance(value, str):
                value = value.strip().lower()
            try:
                target = Mode(value)
            except ValueError:
                log.warning("ignoring mode event with invalid value %r", event.value)
                return
            self._transition(target)
        elif kind == "brightness":
            try:
                level = float(event.value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                log.warning("ignoring brightness event with invalid value %r", event.value)
                return
            self._brightness = _clamp(level, 0.0, 1.0)
        elif kind == "blink":
            self._start_blink()
        elif kind == "motion":
            if self._mode is Mode.IDLE:
                self._transition(Mode.SCARE)
            elif self._mode is Mode.SCARE:
                self._enter_scare()  # restart the timer (and the drama)
            else:
                log.debug("motion ignored in mode %s", self._mode.value)
        else:
            log.debug("behavior ignoring event kind %r", kind)

    def step(self, dt: float) -> tuple[EyeState, EyeState]:
        """Advance the animation by dt seconds and return (left, right)."""
        dt = _clamp(dt, 0.0, MAX_DT)
        m = self.motion

        # SCARE lifecycle: countdown + rapid saccade burst
        if self._mode is Mode.SCARE:
            self._scare_timer -= dt
            if self._scare_timer <= 0.0:
                self._transition(Mode.IDLE)
            elif self._scare_burst_left > 0:
                self._scare_next -= dt
                if self._scare_next <= 0.0:
                    to_l, to_r = self._pick_targets()
                    self._start_saccade(to_l, to_r, self._uniform(SCARE_SACCADE_DUR))
                    self._scare_burst_left -= 1
                    self._scare_next = self._uniform(SCARE_SACCADE_GAP)

        # IDLE wander scheduling
        if self._mode is Mode.IDLE:
            self._sacc_timer -= dt
            if self._sacc_timer <= 0.0:
                to_l, to_r = self._pick_targets()
                self._start_saccade(to_l, to_r, self._uniform(m.saccade_duration))
                self._sacc_timer = self._uniform(m.saccade_interval)

        # saccade interpolation (ease-out cubic)
        if self._sacc_active:
            self._sacc_t += dt
            u = min(1.0, self._sacc_t / self._sacc_dur)
            e = 1.0 - (1.0 - u) ** 3
            self._gaze_l = (
                self._sacc_from_l[0] + (self._sacc_to_l[0] - self._sacc_from_l[0]) * e,
                self._sacc_from_l[1] + (self._sacc_to_l[1] - self._sacc_from_l[1]) * e,
            )
            self._gaze_r = (
                self._sacc_from_r[0] + (self._sacc_to_r[0] - self._sacc_from_r[0]) * e,
                self._sacc_from_r[1] + (self._sacc_to_r[1] - self._sacc_from_r[1]) * e,
            )
            if u >= 1.0:
                self._sacc_active = False

        # blink scheduling (not while sleeping; the lids are already shut)
        if self._mode is not Mode.SLEEP:
            self._blink_timer -= dt
            if self._blink_timer <= 0.0:
                # Only roll for a double when a blink actually starts; if the
                # timer expires mid-blink, the in-flight blink must not be
                # retroactively promoted/demoted.
                if self._start_blink():
                    self._double_pending = self.rng.random() < DOUBLE_BLINK_P
                mult = STARE_BLINK_MULT if self._mode is Mode.STARE else 1.0
                self._blink_timer = self._uniform(m.blink_interval) * mult

        # blink envelope (multiplies the mode's base openness)
        blink_env = 1.0
        if self._blink_t is not None:
            self._blink_t += dt
            t = self._blink_t
            if t < self._blink_close:
                blink_env = 1.0 - t / self._blink_close
            elif t < self._blink_close + self._blink_open:
                blink_env = (t - self._blink_close) / self._blink_open
            elif self._double_pending:
                self._double_pending = False
                self._blink_t = 0.0
            else:
                self._blink_t = None

        # pupil
        if self._mode is Mode.SCARE:
            self._pupil += (SCARE_PUPIL - self._pupil) * _ease(dt, SCARE_PUPIL_RATE)
            self._pupil_target = self._pupil  # walk resumes from here afterwards
        elif self._mode is not Mode.SLEEP:
            walk = self.rng.uniform(-1.0, 1.0) * m.pupil_speed * PUPIL_WALK_GAIN * dt
            self._pupil_target = _clamp(
                self._pupil_target + walk, PUPIL_WALK_LO, PUPIL_WALK_HI
            )
            self._pupil += (self._pupil_target - self._pupil) * _ease(dt, PUPIL_EASE_RATE)

        # lids: ease the base openness toward the mode's target
        if self._mode is Mode.SCARE:
            target, rate = SCARE_OPENNESS, SCARE_OPEN_RATE
        elif self._mode is Mode.SLEEP:
            target, rate = 0.0, SLEEP_CLOSE_RATE
        else:
            target, rate = 1.0, OPEN_RATE
        self._open_base += (target - self._open_base) * _ease(dt, rate)

        # sleep twitch
        twitch = 0.0
        if self._mode is Mode.SLEEP:
            if self._twitch_t is None:
                self._twitch_timer -= dt
                if self._twitch_timer <= 0.0:
                    self._twitch_t = 0.0
                    self._twitch_timer = self._uniform(TWITCH_INTERVAL)
            if self._twitch_t is not None:
                self._twitch_t += dt
                if self._twitch_t >= TWITCH_DUR:
                    self._twitch_t = None
                else:
                    twitch = TWITCH_AMP * math.sin(math.pi * self._twitch_t / TWITCH_DUR)
        else:
            self._twitch_t = None

        # sinusoidal drift: amplitude eases so mode/theme changes never jump
        amp_target = m.drift if self._mode is Mode.IDLE else 0.0
        self._drift_amp += (amp_target - self._drift_amp) * _ease(dt, DRIFT_EASE_RATE)
        self._drift_phase += dt
        drift_x = self._drift_amp * DRIFT_GAIN * math.sin(
            DRIFT_W1 * self._drift_phase + self._drift_p0
        )
        drift_y = self._drift_amp * DRIFT_GAIN * math.sin(
            DRIFT_W2 * self._drift_phase + self._drift_p0 + 1.7
        )

        # flicker: multiplies brightness, always within [1 - flicker, 1]
        noise = self.rng.uniform(-1.0, 1.0)
        flicker = _clamp(m.flicker, 0.0, 1.0)
        flicker_mult = 1.0 - flicker * abs(noise)

        openness = _clamp(self._open_base * blink_env + twitch, 0.0, 1.25)
        pupil = _clamp(self._pupil, 0.0, 1.0)
        brightness = _clamp(self._brightness * flicker_mult, 0.0, 1.0)
        left = EyeState(
            gaze_x=_clamp(self._gaze_l[0] + drift_x, -1.0, 1.0),
            gaze_y=_clamp(self._gaze_l[1] + drift_y, -1.0, 1.0),
            pupil=pupil,
            openness=openness,
            brightness=brightness,
        )
        right = EyeState(
            gaze_x=_clamp(self._gaze_r[0] + drift_x, -1.0, 1.0),
            gaze_y=_clamp(self._gaze_r[1] + drift_y, -1.0, 1.0),
            pupil=pupil,
            openness=openness,
            brightness=brightness,
        )
        return left, right

    # -- internals -------------------------------------------------------------

    def _uniform(self, span: tuple[float, float]) -> float:
        return self.rng.uniform(span[0], span[1])

    def _transition(self, target: Mode) -> None:
        if target is self._mode:
            if target is Mode.SCARE:
                self._enter_scare()  # re-trigger: restart timer + burst
            return
        old = self._mode
        self._mode = target
        log.info("mode %s -> %s", old.value, target.value)
        if target is Mode.SCARE:
            self._enter_scare()
        elif target is Mode.STARE:
            self._start_saccade(
                (0.0, 0.0), (0.0, 0.0), self._uniform(self.motion.saccade_duration)
            )
            self._blink_timer = (
                self._uniform(self.motion.blink_interval) * STARE_BLINK_MULT
            )
        elif target is Mode.SLEEP:
            self._twitch_timer = self._uniform(TWITCH_INTERVAL)
            self._sacc_active = False  # gaze freezes behind closed lids
        else:  # IDLE (mode command, or SCARE auto-return / waking up)
            self._sacc_timer = self._uniform(self.motion.saccade_interval)
            self._blink_timer = self._uniform(self.motion.blink_interval)
            self._pupil_target = self._pupil  # walk resumes without a jump

    def _enter_scare(self) -> None:
        self._mode = Mode.SCARE
        self._scare_timer = SCARE_DURATION
        self._start_saccade((0.0, 0.0), (0.0, 0.0), SCARE_SNAP_DUR)
        self._scare_burst_left = self.rng.randint(2, 3)
        self._scare_next = self._uniform(SCARE_FIRST_GAP)

    def _pick_targets(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Left target uniform in the wander disc; right = left + crazy offset."""
        m = self.motion
        r = m.wander * math.sqrt(self.rng.random())
        a = self.rng.uniform(0.0, math.tau)
        left = (r * math.cos(a), r * math.sin(a))
        # independent offset, scaled by crazy (exactly zero when crazy == 0)
        r2 = m.wander * math.sqrt(self.rng.random())
        a2 = self.rng.uniform(0.0, math.tau)
        right = (
            _clamp(left[0] + m.crazy * r2 * math.cos(a2), -1.0, 1.0),
            _clamp(left[1] + m.crazy * r2 * math.sin(a2), -1.0, 1.0),
        )
        return left, right

    def _start_saccade(
        self,
        to_l: tuple[float, float],
        to_r: tuple[float, float],
        duration: float,
    ) -> None:
        self._sacc_from_l = self._gaze_l
        self._sacc_from_r = self._gaze_r
        self._sacc_to_l = (_clamp(to_l[0], -1.0, 1.0), _clamp(to_l[1], -1.0, 1.0))
        self._sacc_to_r = (_clamp(to_r[0], -1.0, 1.0), _clamp(to_r[1], -1.0, 1.0))
        self._sacc_t = 0.0
        self._sacc_dur = max(duration, MIN_SACCADE_DUR)
        self._sacc_active = True

    def _start_blink(self) -> bool:
        if self._blink_t is not None:
            return False  # already mid-blink
        speed = max(self.motion.blink_speed, 1e-3)
        self._blink_close = BLINK_CLOSE_S / speed
        self._blink_open = BLINK_OPEN_S / speed
        self._blink_t = 0.0
        return True
