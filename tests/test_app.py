"""End-to-end tests for app.main().

These exercise the full loop headlessly and therefore need the renderer and
behavior modules, which are developed separately; the whole module skips
cleanly while they are absent so collection never breaks.
"""

from __future__ import annotations

import json
import queue

import numpy as np
import pytest

pytest.importorskip("spookyeyes.themes")
pytest.importorskip("spookyeyes.eye")
pytest.importorskip("spookyeyes.behavior")

from PIL import Image, ImageDraw

from spookyeyes.model import SIZE, Event

IRIS_FRAC = 0.62


def _theme_json(name: str) -> dict:
    return {
        "name": name,
        "background": [10, 5, 5],
        "layers": {
            "sclera": "sclera.png",
            "iris": "iris.png",
            "highlight": "highlight.png",
        },
        "pupil": {
            "shape": "round",
            "color": [5, 0, 0],
            "min_frac": 0.22,
            "max_frac": 0.55,
            "slit_width_frac": 0.14,
        },
        "iris_frac": IRIS_FRAC,
        "gaze_range_px": 52,
        "sclera_parallax": 0.35,
        "eyelids": {"color": [8, 4, 4], "style": "curved", "upper_bias": 0.6},
        "motion": {
            "saccade_interval": [0.6, 3.0],
            "saccade_duration": [0.09, 0.22],
            "wander": 0.35,
            "crazy": 0.1,
            "blink_interval": [2.0, 8.0],
            "blink_speed": 1.0,
            "pupil_speed": 0.5,
            "drift": 0.0,
            "flicker": 0.0,
        },
    }


def _write_theme(themes_dir, name: str, tint: tuple[int, int, int]) -> None:
    d = themes_dir / name
    d.mkdir(parents=True)
    sclera = Image.new("RGB", (320, 320), tint)
    sclera.save(d / "sclera.png")

    iris_d = int(round(IRIS_FRAC * SIZE))
    iris = Image.new("RGBA", (iris_d, iris_d), (0, 0, 0, 0))
    ImageDraw.Draw(iris).ellipse(
        (0, 0, iris_d - 1, iris_d - 1), fill=(120, 70, 30, 255)
    )
    iris.save(d / "iris.png")

    hl = Image.new("RGBA", (24, 24), (0, 0, 0, 0))
    ImageDraw.Draw(hl).ellipse((0, 0, 23, 23), fill=(255, 255, 255, 180))
    hl.save(d / "highlight.png")

    (d / "theme.json").write_text(json.dumps(_theme_json(name)))


@pytest.fixture
def theme_setup(tmp_path):
    """Two minimal on-disk themes + a config file pointing at them."""
    themes_dir = tmp_path / "themes"
    _write_theme(themes_dir, "apptest", (200, 200, 190))
    _write_theme(themes_dir, "second", (60, 10, 10))
    config = tmp_path / "config.toml"
    config.write_text(
        "[theme]\n"
        'name = "apptest"\n'
        f"dir = {json.dumps(str(themes_dir))}\n"
    )
    return config


def test_main_null_output_renders_frames(theme_setup) -> None:
    from spookyeyes import app

    rc = app.main(
        [
            "--config", str(theme_setup),
            "--theme", "apptest",
            "--output", "null",
            "--frames", "30",
            "--seed", "1",
            "--fps", "120",
        ]
    )
    assert rc == 0


def test_main_quit_event_exits_before_rendering(theme_setup, tmp_path) -> None:
    from spookyeyes import app

    events: queue.Queue = queue.Queue()
    events.put(Event("quit"))
    rec_dir = tmp_path / "quit_rec"
    rc = app.main(
        [
            "--config", str(theme_setup),
            "--output", "record",
            "--record-dir", str(rec_dir),
            "--frames", "30",
            "--seed", "1",
            "--fps", "120",
        ],
        events=events,
    )
    assert rc == 0
    assert sorted(rec_dir.glob("*.png")) == []  # quit drained before any frame


def test_main_record_output_writes_round_frames(theme_setup, tmp_path) -> None:
    from spookyeyes import app

    rec_dir = tmp_path / "rec"
    gif = tmp_path / "demo.gif"
    rc = app.main(
        [
            "--config", str(theme_setup),
            "--output", "record",
            "--record-dir", str(rec_dir),
            "--gif", str(gif),
            "--frames", "5",
            "--seed", "1",
            "--fps", "120",
        ]
    )
    assert rc == 0
    pngs = sorted(rec_dir.glob("*.png"))
    assert [p.name for p in pngs] == [f"frame_{i:05d}.png" for i in range(5)]
    assert gif.exists()
    with Image.open(pngs[0]) as im:
        arr = np.asarray(im.convert("RGB"))
    assert arr.shape[0] == SIZE
    # The renderer's round vignette leaves the frame corners black.
    assert not arr[0, 0].any()


def test_main_handles_theme_switch_and_input_events(theme_setup) -> None:
    from spookyeyes import app

    events: queue.Queue = queue.Queue()
    events.put(Event("theme", "second"))
    events.put(Event("theme", "no-such-theme"))  # warning, keeps running
    events.put(Event("brightness", 0.5))
    events.put(Event("blink"))
    events.put(Event("look", "doorbell"))
    events.put(Event("mode", "stare"))
    rc = app.main(
        [
            "--config", str(theme_setup),
            "--output", "null",
            "--frames", "10",
            "--seed", "1",
            "--fps", "120",
        ],
        events=events,
    )
    assert rc == 0
    assert events.empty()  # everything was drained


def test_main_mirror_config_flips_frames(theme_setup, tmp_path) -> None:
    """mirror_left flips only the left half of the recorded composite."""
    from spookyeyes import app

    plain_dir = tmp_path / "plain"
    mirrored_dir = tmp_path / "mirrored"
    mirrored_cfg = tmp_path / "mirrored.toml"
    mirrored_cfg.write_text(
        theme_setup.read_text()
        + "\n[display]\nmirror_left = true\n"
    )
    args = ["--output", "record", "--frames", "1", "--seed", "1", "--fps", "120"]
    assert app.main(["--config", str(theme_setup), "--record-dir", str(plain_dir), *args]) == 0
    assert app.main(["--config", str(mirrored_cfg), "--record-dir", str(mirrored_dir), *args]) == 0

    with Image.open(plain_dir / "frame_00000.png") as im:
        plain = np.asarray(im.convert("RGB"))
    with Image.open(mirrored_dir / "frame_00000.png") as im:
        mirrored = np.asarray(im.convert("RGB"))
    assert np.array_equal(mirrored[:, :SIZE], np.fliplr(plain[:, :SIZE]))
    assert np.array_equal(mirrored[:, SIZE:], plain[:, SIZE:])


def test_main_missing_theme_is_an_error(theme_setup) -> None:
    from spookyeyes import app

    rc = app.main(
        [
            "--config", str(theme_setup),
            "--theme", "does-not-exist",
            "--output", "null",
            "--frames", "1",
        ]
    )
    assert rc == 1


def test_main_missing_config_is_an_error(tmp_path) -> None:
    from spookyeyes import app

    rc = app.main(["--config", str(tmp_path / "nope.toml"), "--output", "null"])
    assert rc == 1


def test_main_rejects_unknown_output() -> None:
    from spookyeyes import app

    with pytest.raises(SystemExit):
        app.main(["--output", "hologram"])


def test_main_publishes_look_state_and_scare_reset(theme_setup, tmp_path, monkeypatch) -> None:
    """End to end over a fake broker: a look command is published on the
    state topic, and a scare resets it to center and publishes that too."""
    from spookyeyes import app
    from spookyeyes.inputs import mqtt as mqtt_mod

    published: list[tuple[str, object]] = []

    class FakeClient:
        def __init__(self) -> None:
            self.on_connect = None
            self.on_message = None

        def username_pw_set(self, *a, **k): pass
        def will_set(self, *a, **k): pass
        def reconnect_delay_set(self, *a, **k): pass
        def connect_async(self, *a, **k): pass
        def loop_start(self): pass
        def loop_stop(self): pass
        def disconnect(self): pass
        def subscribe(self, *a, **k): pass

        def publish(self, topic, payload=None, qos=0, retain=False):
            published.append((topic, payload))

    monkeypatch.setattr(mqtt_mod.MqttInput, "_default_client_factory", lambda self: FakeClient())

    config = tmp_path / "config_mqtt.toml"
    config.write_text(
        theme_setup.read_text()
        + "\n[mqtt]\nenabled = true\n"
        + "\n[look]\ndoorbell_x = 0.5\ndoorbell_y = -0.1\n"
    )
    events: queue.Queue = queue.Queue()
    events.put(Event("look", "doorbell"))
    events.put(Event("mode", "stare"))
    events.put(Event("mode", "scare"))
    rc = app.main(
        ["--config", str(config), "--output", "null", "--frames", "5", "--seed", "1", "--fps", "120"],
        events=events,
    )
    assert rc == 0
    looks = [p for t, p in published if t == "spookyeyes/state/look"]
    assert "doorbell" in looks           # after the look command
    assert looks[-1] == "center"         # reset by the scare, published
    assert looks.index("doorbell") < len(looks) - 1
    modes = [p for t, p in published if t == "spookyeyes/state/mode"]
    assert modes[-1] == "scare"


def _fake_broker(monkeypatch) -> list[tuple[str, object]]:
    """Patch MqttInput to a recording fake client; returns the publish log."""
    from spookyeyes.inputs import mqtt as mqtt_mod

    published: list[tuple[str, object]] = []

    class FakeClient:
        def __init__(self) -> None:
            self.on_connect = None
            self.on_message = None

        def username_pw_set(self, *a, **k): pass
        def will_set(self, *a, **k): pass
        def reconnect_delay_set(self, *a, **k): pass
        def connect_async(self, *a, **k): pass
        def loop_start(self): pass
        def loop_stop(self): pass
        def disconnect(self): pass
        def subscribe(self, *a, **k): pass

        def publish(self, topic, payload=None, qos=0, retain=False):
            published.append((topic, payload))

    monkeypatch.setattr(mqtt_mod.MqttInput, "_default_client_factory", lambda self: FakeClient())
    return published


def _run(config, events, frames=3):
    from spookyeyes import app

    return app.main(
        ["--config", str(config), "--output", "null", "--frames", str(frames), "--seed", "1", "--fps", "120"],
        events=events,
    )


def _last(published, topic):
    vals = [p for t, p in published if t == topic]
    return vals[-1] if vals else None


def test_main_settings_persist_and_reload(theme_setup, tmp_path, monkeypatch) -> None:
    """Doorbell, mirror and default-theme events are applied, published, and
    saved to settings.json next to config.toml; a fresh start reads them back."""
    from spookyeyes import app

    published = _fake_broker(monkeypatch)
    settings = theme_setup.parent / "settings.json"
    config = tmp_path / "config_mqtt.toml"
    config.write_text(theme_setup.read_text() + "\n[mqtt]\nenabled = true\n")

    events: queue.Queue = queue.Queue()
    events.put(Event("doorbell", {"x": 0.55}))
    events.put(Event("doorbell", {"y": -0.15}))
    events.put(Event("mirror", {"left": True}))
    events.put(Event("default_theme", "second"))
    events.put(Event("default_theme", "no-such-theme"))  # ignored, state republished
    assert _run(config, events) == 0

    saved = json.loads(settings.read_text())
    assert saved["doorbell"] == {"x": 0.55, "y": -0.15}
    assert saved["mirror_left"] is True and saved["mirror_right"] is False
    assert saved["default_theme"] == "second"
    assert _last(published, "spookyeyes/state/doorbell_x") == "0.55"
    assert _last(published, "spookyeyes/state/mirror_left") == "ON"
    assert _last(published, "spookyeyes/state/default_theme") == "second"
    assert _last(published, "spookyeyes/state/theme") == "apptest"  # current theme untouched

    # second start: saved default theme is the startup theme, settings republished
    published.clear()
    assert _run(config, queue.Queue()) == 0
    assert _last(published, "spookyeyes/state/theme") == "second"
    assert _last(published, "spookyeyes/state/default_theme") == "second"
    assert _last(published, "spookyeyes/state/mirror_left") == "ON"
    assert _last(published, "spookyeyes/state/doorbell_y") == "-0.15"

    cfg = app.AppConfig.load(str(config))
    assert app._settings_path(cfg, str(config)) == settings


def test_main_cli_theme_overrides_saved_default(theme_setup, monkeypatch) -> None:
    from spookyeyes import app

    published = _fake_broker(monkeypatch)
    (theme_setup.parent / "settings.json").write_text(json.dumps({"default_theme": "second"}))
    config = theme_setup.parent / "config_mqtt.toml"
    config.write_text(theme_setup.read_text() + "\n[mqtt]\nenabled = true\n")
    rc = app.main(
        ["--config", str(config), "--theme", "apptest", "--output", "null", "--frames", "2", "--fps", "120"],
        events=queue.Queue(),
    )
    assert rc == 0
    assert _last(published, "spookyeyes/state/theme") == "apptest"


def test_main_saved_default_theme_missing_falls_back(theme_setup, monkeypatch) -> None:
    published = _fake_broker(monkeypatch)
    (theme_setup.parent / "settings.json").write_text(json.dumps({"default_theme": "gone"}))
    config = theme_setup.parent / "config_mqtt.toml"
    config.write_text(theme_setup.read_text() + "\n[mqtt]\nenabled = true\n")
    assert _run(config, queue.Queue()) == 0
    assert _last(published, "spookyeyes/state/theme") == "apptest"
    assert _last(published, "spookyeyes/state/default_theme") == "apptest"


def test_main_theme_expose_filters_discovery_options(theme_setup, monkeypatch) -> None:
    published = _fake_broker(monkeypatch)
    config = theme_setup.parent / "config_mqtt.toml"
    config.write_text(
        theme_setup.read_text()
        + 'expose = ["second", "not-there"]\n'
        + "\n[mqtt]\nenabled = true\n"
    )
    events: queue.Queue = queue.Queue()
    events.put(Event("default_theme", "second"))   # exposed -> accepted
    events.put(Event("default_theme", "apptest"))  # on disk but not exposed -> ignored
    assert _run(config, events) == 0
    assert _last(published, "spookyeyes/state/default_theme") == "second"
    # Discovery options come from on_connect, which the fake never fires, so
    # the option list itself is checked through the pure helper.
    from spookyeyes import app

    assert app._exposed_themes(["apptest", "second"], ["second", "not-there"]) == ["second"]
    assert app._exposed_themes(["a", "b"], []) == ["a", "b"]


def test_settings_bad_file_is_ignored(tmp_path) -> None:
    from spookyeyes import app

    p = tmp_path / "settings.json"
    assert app._load_settings(p) == {}
    p.write_text("not json")
    assert app._load_settings(p) == {}
    p.write_text("[1, 2]")
    assert app._load_settings(p) == {}
    assert app._settings_doorbell({"doorbell": {"x": 9, "y": -9}}, (0.7, -0.3)) == (1.0, -1.0)
    assert app._settings_doorbell({"doorbell": "x"}, (0.7, -0.3)) == (0.7, -0.3)
    assert app._settings_bool({"mirror_left": "yes"}, "mirror_left", False) is False
