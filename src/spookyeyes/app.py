"""Application entry point: CLI parsing, wiring, and the fixed-timestep loop.

Wires config -> theme -> renderers -> behavior -> inputs -> output, then runs
the render loop at ``display.fps`` with monotonic-clock pacing. All input
threads talk to the loop only via the event queue; the app itself handles
``"theme"`` (reload renderers + motion) and ``"quit"`` events and forwards the
rest to the behavior engine.
"""

from __future__ import annotations

import argparse
import json
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
from spookyeyes.config import AppConfig, ConfigError, MqttConfig
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


def _exposed_themes(on_disk: list[str], expose: list[str]) -> list[str]:
    """Themes offered to HA: the `[theme] expose` allowlist in its own order,
    minus names not found on disk (warned), or everything when it is empty."""
    if not expose:
        return list(on_disk)
    missing = [t for t in expose if t not in on_disk]
    if missing:
        log.warning("[theme] expose lists themes not on disk, skipped: %s", ", ".join(missing))
    return [t for t in expose if t in on_disk]


def _settings_path(cfg: AppConfig, config_path: str | None) -> Path:
    p = Path(cfg.settings.file)
    if not p.is_absolute() and config_path:
        p = Path(config_path).resolve().parent / p
    return p


def _load_settings(path: Path) -> dict:
    """Runtime settings saved from HA; an unreadable file is a warning."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        log.warning("ignoring unreadable settings file %s: %s", path, e)
        return {}
    if not isinstance(data, dict):
        log.warning("ignoring settings file %s: not a JSON object", path)
        return {}
    return data


def _save_settings(path: Path, settings: dict) -> None:
    """Atomic write; a failure is logged, the live value still applies."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)
    except OSError as e:
        log.warning("cannot save settings to %s: %s", path, e)


def _settings_doorbell(settings: dict, default: tuple[float, float]) -> tuple[float, float]:
    try:
        d = settings["doorbell"]
        x, y = float(d["x"]), float(d["y"])
    except (KeyError, TypeError, ValueError):
        return default
    return (min(1.0, max(-1.0, x)), min(1.0, max(-1.0, y)))


def _settings_bool(settings: dict, key: str, default: bool) -> bool:
    v = settings.get(key, default)
    return v if isinstance(v, bool) else default


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
    on_disk = _list_themes(themes_dir)
    exposed = _exposed_themes(on_disk, cfg.theme.expose)

    # Runtime settings saved from Home Assistant override the config defaults.
    settings_path = _settings_path(cfg, args.config)
    settings = _load_settings(settings_path)
    default_theme = str(settings.get("default_theme", cfg.theme.name))
    if default_theme not in on_disk and on_disk:
        log.warning("saved default theme %r not on disk, using %r", default_theme, cfg.theme.name)
        default_theme = cfg.theme.name
    mirror_left = _settings_bool(settings, "mirror_left", cfg.display.mirror_left)
    mirror_right = _settings_bool(settings, "mirror_right", cfg.display.mirror_right)
    doorbell = _settings_doorbell(settings, (cfg.look.doorbell_x, cfg.look.doorbell_y))

    def _persist() -> None:
        settings.update(
            doorbell={"x": engine.preset("doorbell")[0], "y": engine.preset("doorbell")[1]},
            mirror_left=mirror_left,
            mirror_right=mirror_right,
            default_theme=default_theme,
        )
        _save_settings(settings_path, settings)

    # CLI --theme wins, then the saved default, then [theme] name.
    theme_name = args.theme or default_theme
    try:
        theme = themes_mod.load_theme(themes_dir, theme_name)
    except (themes_mod.ThemeError, OSError) as e:
        if theme_name != cfg.theme.name:
            log.warning("cannot load theme %r (%s); falling back to %r",
                        theme_name, e, cfg.theme.name)
            theme_name = cfg.theme.name
            try:
                theme = themes_mod.load_theme(themes_dir, theme_name)
            except (themes_mod.ThemeError, OSError) as e2:
                log.error("cannot load theme %r from %s: %s", theme_name, themes_dir, e2)
                return 1
        else:
            log.error("cannot load theme %r from %s: %s", theme_name, themes_dir, e)
            return 1

    left_renderer, right_renderer = _make_renderers(theme)
    rng = random.Random(args.seed)  # Random(None) seeds from the OS
    presets = behavior_mod.look_presets(cfg.look.amplitude, doorbell)
    engine = behavior_mod.BehaviorEngine(theme.motion, rng=rng, presets=presets)

    if events is None:
        events = queue.Queue()

    mqtt_input = _start_mqtt(cfg.mqtt, events, theme_options=exposed or None)

    brightness = 1.0

    def _current_mode_str() -> str:
        mode = getattr(engine, "mode", Mode.IDLE)
        return mode.value if isinstance(mode, Mode) else str(mode)

    def _current_look() -> str:
        return str(getattr(engine, "look", "center"))

    def _state() -> dict:
        return {
            "theme": theme_name,
            "mode": _current_mode_str(),
            "brightness": brightness,
            "look": _current_look(),
            "doorbell": engine.preset("doorbell"),
            "mirror": (mirror_left, mirror_right),
            "default_theme": default_theme,
        }

    def _publish_state() -> None:
        if mqtt_input is None:
            return
        try:
            mqtt_input.publish_state(**_state())
        except Exception:
            log.warning("mqtt publish_state failed", exc_info=True)

    if mqtt_input is not None:
        # Republished on every (re)connect so retained state survives restarts,
        # and after a rejected command so HA's entities snap back.
        mqtt_input.state_provider = _state

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
            "running: theme=%s (default %s) output=%s fps=%d mirror=(%s, %s) exposed=%d/%d themes",
            theme_name,
            default_theme,
            cfg.display.output,
            fps,
            mirror_left,
            mirror_right,
            len(exposed),
            len(on_disk),
        )

        period = 1.0 / fps
        dt = period  # fixed timestep, independent of wall-clock jitter
        frames_done = 0
        fps_count = 0
        running = True
        next_t = time.monotonic()
        fps_t0 = next_t

        last_pub: tuple[object, object] | None = None

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
                elif ev.kind == "doorbell":
                    # Calibration from HA: merge the axis into the preset,
                    # aim at it so the tuner sees the result, persist.
                    try:
                        x, y = engine.preset("doorbell")
                        upd = dict(ev.value)  # type: ignore[arg-type]
                        new_xy = (float(upd.get("x", x)), float(upd.get("y", y)))
                        engine.set_preset("doorbell", new_xy)
                        engine.handle(Event("look", "doorbell"))
                    except (TypeError, ValueError, AttributeError) as e:
                        log.warning("ignoring bad doorbell calibration %r: %s", ev.value, e)
                        continue
                    _persist()
                    log.info("doorbell preset calibrated to x=%.2f y=%.2f", *new_xy)
                    _publish_state()
                elif ev.kind == "mirror":
                    upd = ev.value if isinstance(ev.value, dict) else {}
                    if "left" in upd:
                        mirror_left = bool(upd["left"])
                    if "right" in upd:
                        mirror_right = bool(upd["right"])
                    _persist()
                    log.info("mirror set to left=%s right=%s", mirror_left, mirror_right)
                    _publish_state()
                elif ev.kind == "default_theme":
                    name = str(ev.value)
                    if name not in (exposed or on_disk):
                        log.warning("default theme %r is not an exposed theme, ignored", name)
                        _publish_state()
                        continue
                    default_theme = name
                    _persist()
                    log.info("default theme set to %r", default_theme)
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
                    elif ev.kind in ("mode", "look"):
                        _publish_state()
            if not running:
                break

            left_state, right_state = engine.step(dt)

            # The engine changes mode and look on its own (SCARE's timed
            # return to IDLE, look reset on scare/sleep) — publish
            # whenever either changes so the retained MQTT state tracks
            # reality, not just cmd topics.
            if mqtt_input is not None:
                now_state = (getattr(engine, "mode", None), getattr(engine, "look", None))
                if now_state != last_pub:
                    last_pub = now_state
                    _publish_state()

            t_anim = frames_done * dt  # deterministic animation clock (spin)
            left_frame = left_renderer.render(left_state, t_anim)
            right_frame = right_renderer.render(right_state, t_anim)
            if mirror_left:
                left_frame = np.fliplr(left_frame)
            if mirror_right:
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
        log.info("shutdown complete")


if __name__ == "__main__":
    raise SystemExit(main())
