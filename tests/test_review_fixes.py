"""Regression tests for the code-review fixes (path traversal, config typing,
UTF-8, MQTT theme options / state republish / offline flush)."""

from __future__ import annotations

import json
import queue
from types import SimpleNamespace

import pytest

from spookyeyes.config import AppConfig, ConfigError
from spookyeyes.inputs.mqtt import MqttInput
from spookyeyes.model import Event
from spookyeyes.themes import ThemeError, load_theme


# -- theme name validation -------------------------------------------------


@pytest.mark.parametrize("name", ["../evil", "..", "/etc", "a/b", "a\\b", ".hidden", ""])
def test_load_theme_rejects_path_escapes(tmp_path, name):
    with pytest.raises(ThemeError, match="invalid theme name"):
        load_theme(tmp_path, name)


def test_load_theme_traversal_does_not_touch_target(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "theme.json").write_text("{}")
    themes_dir = tmp_path / "themes"
    themes_dir.mkdir()
    with pytest.raises(ThemeError, match="invalid theme name"):
        load_theme(themes_dir, "../outside")


# -- config type validation + encoding -------------------------------------


def _load_toml(tmp_path, text: str) -> AppConfig:
    p = tmp_path / "config.toml"
    p.write_text(text, encoding="utf-8")
    return AppConfig.load(p)


def test_config_rejects_string_for_int(tmp_path):
    with pytest.raises(ConfigError, match=r"\[display\] fps"):
        _load_toml(tmp_path, '[display]\nfps = "30"\n')


def test_config_rejects_string_for_float(tmp_path):
    with pytest.raises(ConfigError, match=r"\[pir\] cooldown"):
        _load_toml(tmp_path, '[pir]\ncooldown = "10"\n')


def test_config_rejects_int_for_bool(tmp_path):
    with pytest.raises(ConfigError, match=r"\[mqtt\] enabled"):
        _load_toml(tmp_path, "[mqtt]\nenabled = 1\n")


def test_config_rejects_int_for_str(tmp_path):
    with pytest.raises(ConfigError, match=r"\[theme\] name"):
        _load_toml(tmp_path, "[theme]\nname = 5\n")


def test_config_accepts_int_for_float_field(tmp_path):
    cfg = _load_toml(tmp_path, "[pir]\ncooldown = 10\n")
    assert cfg.pir.cooldown == 10.0
    assert isinstance(cfg.pir.cooldown, float)


def test_config_utf8_content_ok(tmp_path):
    cfg = _load_toml(tmp_path, '[theme]\nname = "demon"\ndir = "thèmes"\n')
    assert cfg.theme.dir == "thèmes"


def test_config_non_utf8_raises_config_error(tmp_path):
    p = tmp_path / "config.toml"
    p.write_bytes('[theme]\ndir = "th\xe8mes"\n'.encode("latin-1"))
    with pytest.raises(ConfigError, match="UTF-8"):
        AppConfig.load(p)


# -- MQTT: theme options, state republish on connect, offline flush ---------


class FakeMsgInfo:
    def __init__(self):
        self.waited = False

    def wait_for_publish(self, timeout=None):
        self.waited = True


class FakeClient:
    def __init__(self):
        self.published: list[tuple[str, object, int, bool]] = []
        self.subscribed: list[str] = []
        self.disconnected = False
        self.last_info: FakeMsgInfo | None = None
        self.on_connect = None
        self.on_message = None

    def username_pw_set(self, u, p):
        pass

    def will_set(self, topic, payload, qos=0, retain=False):
        pass

    def reconnect_delay_set(self, min_delay=1, max_delay=30):
        pass

    def connect_async(self, host, port, keepalive=60):
        pass

    def loop_start(self):
        pass

    def loop_stop(self):
        pass

    def disconnect(self):
        self.disconnected = True

    def subscribe(self, topic, qos=0):
        self.subscribed.append(topic)

    def publish(self, topic, payload=None, qos=0, retain=False):
        self.published.append((topic, payload, qos, retain))
        self.last_info = FakeMsgInfo()
        return self.last_info


def _connected_input(**kwargs):
    from spookyeyes.config import MqttConfig

    client = FakeClient()
    inp = MqttInput(
        MqttConfig(enabled=True, discovery=True),
        queue.Queue(),
        client_factory=lambda: client,
        **kwargs,
    )
    inp.start()
    client.on_connect(client, None, None, SimpleNamespace(is_failure=False))
    return inp, client


def _discovery_payload(client, topic):
    for t, payload, _qos, _retain in client.published:
        if t == topic:
            return json.loads(payload)
    raise AssertionError(f"{topic} never published")


def test_theme_options_override_lands_in_discovery():
    _inp, client = _connected_input(theme_options=["human", "vampire"])
    cfg = _discovery_payload(client, "homeassistant/select/spookyeyes_theme/config")
    assert cfg["options"] == ["human", "vampire"]


def test_theme_options_default_unchanged():
    _inp, client = _connected_input()
    cfg = _discovery_payload(client, "homeassistant/select/spookyeyes_theme/config")
    assert cfg["options"] == ["human", "demon", "ghost"]


def test_state_provider_republished_on_connect():
    inp, client = _connected_input()
    client.published.clear()
    inp.state_provider = lambda: ("demon", "idle", 0.5)
    client.on_connect(client, None, None, SimpleNamespace(is_failure=False))
    topics = {t: p for t, p, _q, _r in client.published}
    assert topics.get("spookyeyes/state/theme") == "demon"
    assert topics.get("spookyeyes/state/mode") == "idle"
    assert topics.get("spookyeyes/state/brightness") == "0.5"


def test_close_waits_for_offline_publish():
    inp, client = _connected_input()
    inp.close()
    topic, payload, qos, retain = client.published[-1]
    assert (topic, payload, retain) == ("spookyeyes/availability", "offline", True)
    assert qos == 1
    assert client.last_info is not None and client.last_info.waited
    assert client.disconnected


# -- framebuffer resolution by stable SPI address ---------------------------


def _fake_sysfs(tmp_path, mapping):
    """Build a fake /sys/class/graphics: {fb_name: spi_addr}."""
    root = tmp_path / "graphics"
    for fb_name, spi_addr in mapping.items():
        spi_dir = tmp_path / "devices" / spi_addr
        spi_dir.mkdir(parents=True)
        fb_dir = root / fb_name
        fb_dir.mkdir(parents=True)
        (fb_dir / "device").symlink_to(spi_dir)
    return root


def test_resolve_fb_passes_plain_paths_through(tmp_path):
    from spookyeyes.outputs.fb import resolve_fb

    assert resolve_fb("/dev/fb1", sysfs_root=tmp_path) == "/dev/fb1"
    assert resolve_fb(str(tmp_path / "fbfile"), sysfs_root=tmp_path) == str(
        tmp_path / "fbfile"
    )


def test_resolve_fb_maps_spi_address_regardless_of_probe_order(tmp_path):
    from spookyeyes.outputs.fb import resolve_fb

    # Reversed probe order, as observed on real hardware: spi0.1 got fb1.
    root = _fake_sysfs(tmp_path, {"fb1": "spi0.1", "fb2": "spi0.0"})
    assert resolve_fb("spi0.0", sysfs_root=root) == "/dev/fb2"
    assert resolve_fb("spi0.1", sysfs_root=root) == "/dev/fb1"


def test_resolve_fb_unbound_address_raises(tmp_path):
    from spookyeyes.outputs.fb import resolve_fb

    root = _fake_sysfs(tmp_path, {"fb1": "spi0.1"})
    with pytest.raises(ValueError, match="spi0.0"):
        resolve_fb("spi0.0", sysfs_root=root)
