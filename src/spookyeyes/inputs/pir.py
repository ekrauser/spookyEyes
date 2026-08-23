"""PIR motion sensor input via gpiozero, with cooldown rate limiting.

gpiozero is an optional dependency and is imported lazily inside the default
sensor factory; on non-Pi hosts (or with the pin unavailable) construction
raises :class:`PirUnavailable`, which the app treats as a warning, not a crash.
"""

from __future__ import annotations

import logging
import queue
import time
from typing import Any, Callable

from ..config import PirConfig
from ..model import Event

log = logging.getLogger("spookyeyes.inputs.pir")


class PirUnavailable(RuntimeError):
    """Raised when the PIR sensor cannot be set up (no gpiozero, no GPIO)."""


class PirInput:
    """Wraps a gpiozero MotionSensor; forwards Event("motion") with cooldown.

    gpiozero fires ``when_motion`` from its own callback thread; the handler
    only touches the thread-safe queue and this instance's timestamp (a single
    attribute write, safe under the GIL for our at-most-once-per-cooldown use).
    """

    def __init__(
        self,
        cfg: PirConfig,
        events: queue.Queue[Event],
        sensor_factory: Callable[[int], Any] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cfg = cfg
        self._events = events
        self._monotonic = monotonic
        self._last_forwarded: float | None = None
        factory = sensor_factory or self._default_sensor_factory
        try:
            self._sensor: Any | None = factory(cfg.pin)
        except ImportError as exc:
            raise PirUnavailable(
                "gpiozero is not installed (pip install spookyeyes[pir])"
            ) from exc
        except Exception as exc:  # gpiozero pin/backend errors on non-Pi hosts
            raise PirUnavailable(
                f"PIR sensor on BCM pin {cfg.pin} unavailable: {exc}"
            ) from exc
        self._sensor.when_motion = self._on_motion
        log.info("pir: watching BCM pin %d (cooldown %.1fs)", cfg.pin, cfg.cooldown)

    @staticmethod
    def _default_sensor_factory(pin: int) -> Any:
        from gpiozero import MotionSensor  # lazy: optional dependency

        return MotionSensor(pin)

    def _on_motion(self) -> None:
        now = self._monotonic()
        if (
            self._last_forwarded is not None
            and now - self._last_forwarded < self._cfg.cooldown
        ):
            log.debug("pir: motion suppressed by cooldown")
            return
        self._last_forwarded = now
        log.info("pir: motion detected")
        self._events.put(Event("motion"))

    def close(self) -> None:
        """Detach the callback and release the GPIO pin. Idempotent."""
        sensor = self._sensor
        if sensor is None:
            return
        self._sensor = None
        try:
            sensor.when_motion = None
            sensor.close()
        except Exception:
            log.exception("pir: error during close")
