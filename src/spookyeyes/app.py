"""Application entry point: CLI parsing, wiring, and the fixed-timestep loop.

Wires config -> theme -> renderers -> behavior -> inputs -> output, then runs
the render loop at ``display.fps`` with monotonic-clock pacing. All input
threads talk to the loop only via the event queue; the app itself handles
``"theme"`` (reload renderers + motion) and ``"quit"`` events and forwards the
rest to the behavior engine.
"""

from __future__ import annotations

import argparse
import logging
import queue
import random
import signal
import threading
import time
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from spookyeyes import behavior as behavior_mod
from spookyeyes import eye as eye_mod
from spookyeyes import themes as themes_mod
from spookyeyes.config import AppConfig, ConfigError, MqttConfig, PirConfig
from spookyeyes.model import Event, Mode
from spookyeyes.outputs import make_output

if TYPE_CHECKING:
    from spookyeyes.outputs import Output

log = logging.getLogger("spookyeyes.app")

FPS_LOG_INTERVAL = 5.0  # seconds between measured-FPS log lines


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spookyeyes",
        description="Dual round-display animated Halloween eyes.",
    )
    p.add_argument("--config", metavar="PATH", help="TOML config file")
    p.add_argument("--theme", metavar="NAME", help="theme name (overrides config)")
    p.add_argument(
        "--output",
        choices=("preview", "fb", "record", "null"),
        help="output backend (overrides config)",
    )
    p.add_argument(
        "--frames", type=int, metavar="N", help="exit after N frames (tests/benchmarks)"
    )
    p.add_argument(
        "--record-dir", metavar="DIR", help="directory for record output PNGs"
    )
    p.add_argument(
        "--gif", metavar="PATH", help="also write an animated GIF (record output)"
    )
    p.add_argument("--fps", type=int, metavar="N", help="frame rate (overrides config)")
    p.add_argument(
        "--seed", type=int, metavar="N", help="RNG seed for deterministic behavior"
    )
    p.add_argument("--verbose", action="store_true", help="debug logging")
    return p


def _start_mqtt(
    cfg: MqttConfig,
    events: queue.Queue[Event],
    theme_options: list[str] | None = None,
) -> object | None:
    """Start the MQTT input if enabled. Missing paho-mqtt or a broker problem
    is a warning, never a crash — the prop must keep animating."""
    if not cfg.enabled:
        return None
    from spookyeyes.inputs.mqtt import MqttInput  # stdlib-only module import

    try:
        mqtt = MqttInput(cfg, events, theme_options=theme_options)
        mqtt.start()
        return mqtt
    except ImportError as e:
        # paho is imported lazily inside start(); this is where its absence shows.
        log.warning("MQTT enabled but paho-mqtt is not installed (%s); disabling", e)
        return None
    except Exception:
        log.warning("MQTT input failed to start; continuing without it", exc_info=True)
        return None


def _make_renderers(theme: object) -> tuple[object, object]:
    """Left and right renderers; the right eye counter-rotates spinning
    irises when the theme asks for it (iris_spin_mirror)."""
    right_dir = -1 if getattr(theme, "iris_spin_mirror", False) else 1
    return eye_mod.EyeRenderer(theme), eye_mod.EyeRenderer(theme, spin_dir=right_dir)


def _list_themes(themes_dir: Path) -> list[str]:
    """Names of all loadable themes on disk (for the HA discovery select)."""
    try:
        return sorted(
            d.name for d in themes_dir.iterdir() if (d / "theme.json").is_file()
        )
    except OSError:
        return []


def _start_pir(cfg: PirConfig, events: queue.Queue[Event]) -> object | None:
    """Start the PIR input if enabled. Missing gpiozero or a non-Pi host
    (PirUnavailable) is a warning, never a crash."""
    if not cfg.enabled:
        return None
    try:
        from spookyeyes.inputs.pir import PirInput
    except ImportError as e:
        log.warning("PIR enabled but gpiozero is not installed (%s); disabling", e)
        return None
    try:
        return PirInput(cfg, events)
    except Exception as e:  # PirUnavailable on non-Pi hosts, wiring errors, ...
        log.warning("PIR input unavailable (%s); continuing without it", e)
        return None


def _stop_input(inp: object | None, label: str) -> None:
    if inp is None:
        return
    for meth in ("stop", "close"):
        fn = getattr(inp, meth, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                log.debug("error stopping %s input", label, exc_info=True)
            return


def main(argv: list[str] | None = None, events: queue.Queue[Event] | None = None) -> int:
    """Run the app; returns a process exit code.

    ``events`` lets tests inject a pre-filled event queue; normally the app
    creates its own and shares it with the input threads and the preview
    window.
    """
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    try:
        cfg = AppConfig.load(args.config)
    except (FileNotFoundError, tomllib.TOMLDecodeError, ConfigError) as e:
        log.error("cannot load config: %s", e)
        return 1

    # CLI overrides config.
    if args.theme:
        cfg.theme.name = args.theme
    if args.output:
        cfg.display.output = args.output
    if args.fps is not None:
        cfg.display.fps = args.fps
    if args.record_dir:
        cfg.display.record_dir = args.record_dir

    fps = cfg.display.fps
    if fps <= 0:
        log.error("fps must be positive, got %s", fps)
        return 2

    themes_dir = Path(cfg.theme.dir)
    try:
        theme = themes_mod.load_theme(themes_dir, cfg.theme.name)
    except (themes_mod.ThemeError, OSError) as e:
        log.error("cannot load theme %r from %s: %s", cfg.theme.name, themes_dir, e)
        return 1
    theme_name = cfg.theme.name

    left_renderer, right_renderer = _make_renderers(theme)
    rng = random.Random(args.seed)  # Random(None) seeds from the OS
    engine = behavior_mod.BehaviorEngine(theme.motion, rng=rng)

    if events is None:
        events = queue.Queue()

    mqtt_input = _start_mqtt(cfg.mqtt, events, theme_options=_list_themes(themes_dir) or None)
    pir_input = _start_pir(cfg.pir, events)

    brightness = 1.0

    def _current_mode_str() -> str:
        mode = getattr(engine, "mode", Mode.IDLE)
        return mode.value if isinstance(mode, Mode) else str(mode)

    def _publish_state() -> None:
        if mqtt_input is None:
            return
        try:
            mqtt_input.publish_state(theme_name, _current_mode_str(), brightness)
        except Exception:
            log.warning("mqtt publish_state failed", exc_info=True)

    if mqtt_input is not None:
        # Republished on every (re)connect so retained state survives restarts.
        mqtt_input.state_provider = lambda: (theme_name, _current_mode_str(), brightness)

    output: Output | None = None
    old_handlers: dict[signal.Signals, object] = {}
    try:
        try:
            output = make_output(cfg.display, events=events, gif_path=args.gif)
        except Exception as e:
            log.error("cannot open output %r: %s", cfg.display.output, e)
            return 1

        # The handler must only set a flag: it runs in the main thread at an
        # arbitrary bytecode boundary, and calling events.put() (or logging)
        # there can self-deadlock on a lock the interrupted code already holds.
        stop_signum: int | None = None

        def _on_signal(signum: int, _frame: object) -> None:
            nonlocal stop_signum
            stop_signum = signum

        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    old_handlers[sig] = signal.signal(sig, _on_signal)
                except (ValueError, OSError):  # pragma: no cover - defensive
                    pass

        log.info(
            "running: theme=%s output=%s fps=%d mirror=(%s, %s)",
            theme_name,
            cfg.display.output,
            fps,
            cfg.display.mirror_left,
            cfg.display.mirror_right,
        )

        period = 1.0 / fps
        dt = period  # fixed timestep, independent of wall-clock jitter
        frames_done = 0
        fps_count = 0
        running = True
        next_t = time.monotonic()
        fps_t0 = next_t

        last_pub_mode: object = None

        while running:
            if stop_signum is not None:
                log.info(
                    "received %s, shutting down", signal.Signals(stop_signum).name
                )
                break
            if args.frames is not None and frames_done >= args.frames:
                log.info("rendered %d frames, exiting (--frames)", frames_done)
                break

            # Drain every queued event before stepping.
            while True:
                try:
                    ev = events.get_nowait()
                except queue.Empty:
                    break
                if ev.kind == "quit":
                    log.info("quit event received")
                    running = False
                elif ev.kind == "theme":
                    name = str(ev.value)
                    try:
                        new_theme = themes_mod.load_theme(themes_dir, name)
                    except (themes_mod.ThemeError, OSError) as e:
                        log.warning("theme switch to %r failed: %s", name, e)
                        continue
                    theme = new_theme
                    theme_name = name
                    left_renderer, right_renderer = _make_renderers(theme)
                    engine.set_motion(theme.motion)
                    log.info("switched to theme %r", theme_name)
                    _publish_state()
                else:
                    try:
                        engine.handle(ev)
                    except Exception:
                        log.warning("behavior rejected event %r", ev, exc_info=True)
                        continue
                    if ev.kind == "brightness":
                        try:
                            brightness = min(1.0, max(0.0, float(ev.value)))
                        except (TypeError, ValueError):
                            pass
                        _publish_state()
                    elif ev.kind == "mode":
                        _publish_state()
            if not running:
                break

            left_state, right_state = engine.step(dt)

            # The engine changes mode on its own (PIR startle, SCARE's timed
            # return to IDLE) — publish whenever the observed mode changes so
            # the retained MQTT state tracks reality, not just cmd topics.
            if mqtt_input is not None:
                mode_now = getattr(engine, "mode", None)
                if mode_now is not last_pub_mode:
                    last_pub_mode = mode_now
                    _publish_state()

            t_anim = frames_done * dt  # deterministic animation clock (spin)
            left_frame = left_renderer.render(left_state, t_anim)
            right_frame = right_renderer.render(right_state, t_anim)
            if cfg.display.mirror_left:
                left_frame = np.fliplr(left_frame)
            if cfg.display.mirror_right:
                right_frame = np.fliplr(right_frame)
            output.show(left_frame, right_frame)
            frames_done += 1
            fps_count += 1

            now = time.monotonic()
            if now - fps_t0 >= FPS_LOG_INTERVAL:
                log.info(
                    "rendering at %.1f fps (target %d)", fps_count / (now - fps_t0), fps
                )
                fps_count = 0
                fps_t0 = now

            # Monotonic pacing: sleep to the next slot; if we fell behind,
            # re-anchor instead of spiralling into a sleepless catch-up.
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()

        return 0
    finally:
        for sig, handler in old_handlers.items():
            try:
                signal.signal(sig, handler)  # type: ignore[arg-type]
            except (ValueError, OSError, TypeError):  # pragma: no cover
                pass
        if output is not None:
            try:
                output.close()
            except Exception:
                log.warning("error closing output", exc_info=True)
        _stop_input(mqtt_input, "mqtt")
        _stop_input(pir_input, "pir")
        log.info("shutdown complete")


if __name__ == "__main__":
    raise SystemExit(main())
