"""Tests for inputs/mqtt.py and inputs/pir.py using injected fakes."""

from __future__ import annotations

import json
import queue
from types import SimpleNamespace

import pytest

from spookyeyes.config import MqttConfig, PirConfig
from spookyeyes.inputs.mqtt import MqttInput
from spookyeyes.inputs.pir import PirInput, PirUnavailable
from spookyeyes.model import Event, Mode

# ---------------------------------------------------------------------------
# MQTT fakes
# ---------------------------------------------------------------------------


class FakeClient:
    """Records everything MqttInput does; lets tests fire paho callbacks."""

    def __init__(self) -> None:
        self.subscriptions: list[tuple[str, int]] = []
        self.published: list[tuple[str, object, int, bool]] = []
        self.will: tuple[str, object, int, bool] | None = None
        self.credentials: tuple[str, str] | None = None
        self.reconnect_delays: tuple[int, int] | None = None
        self.connect_args: tuple[str, int, int] | None = None
        self.loop_started = False
        self.loop_stopped = False
        self.disconnected = False
        self.on_connect = None
        self.on_message = None

    # -- configuration calls
    def username_pw_set(self, username: str, password: str = "") -> None:
        self.credentials = (username, password)

    def will_set(self, topic, payload=None, qos=0, retain=False) -> None:
        self.will = (topic, payload, qos, retain)

    def reconnect_delay_set(self, min_delay=1, max_delay=120) -> None:
        self.reconnect_delays = (min_delay, max_delay)

    # -- network lifecycle
    def connect_async(self, host, port=1883, keepalive=60) -> None:
        self.connect_args = (host, port, keepalive)

    def loop_start(self) -> None:
        self.loop_started = True

    def loop_stop(self) -> None:
        self.loop_stopped = True

    def disconnect(self) -> None:
        self.disconnected = True

    # -- traffic
    def subscribe(self, topic, qos=0) -> None:
        self.subscriptions.append((topic, qos))

    def publish(self, topic, payload=None, qos=0, retain=False) -> None:
        self.published.append((topic, payload, qos, retain))

    # -- test helpers
    def simulate_connect(self, failure: bool = False) -> None:
        rc = SimpleNamespace(is_failure=failure)
        self.on_connect(self, None, {}, rc, None)

    def simulate_message(self, topic: str, payload: bytes) -> None:
        self.on_message(self, None, SimpleNamespace(topic=topic, payload=payload))


@pytest.fixture
def mqtt_setup():
    cfg = MqttConfig(enabled=True, username="u", password="p")
    events: queue.Queue[Event] = queue.Queue()
    fake = FakeClient()
    inp = MqttInput(cfg, events, client_factory=lambda: fake)
    inp.start()
    return cfg, events, fake, inp


def drain(q: queue.Queue) -> list[Event]:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


# ---------------------------------------------------------------------------
# MQTT: lifecycle, subscriptions, availability
# ---------------------------------------------------------------------------


def test_start_configures_client(mqtt_setup):
    cfg, _, fake, _ = mqtt_setup
    assert fake.connect_args[0] == cfg.host
    assert fake.connect_args[1] == cfg.port
    assert fake.loop_started
    assert fake.credentials == ("u", "p")
    assert fake.reconnect_delays is not None


def test_lwt_offline_retained(mqtt_setup):
    _, _, fake, _ = mqtt_setup
    topic, payload, _qos, retain = fake.will
    assert topic == "spookyeyes/availability"
    assert payload == "offline"
    assert retain is True


def test_on_connect_subscribes_all_cmd_topics(mqtt_setup):
    _, _, fake, _ = mqtt_setup
    fake.simulate_connect()
    assert {t for t, _ in fake.subscriptions} == {
        "spookyeyes/cmd/theme",
        "spookyeyes/cmd/mode",
        "spookyeyes/cmd/brightness",
        "spookyeyes/cmd/blink",
    }


def test_on_connect_publishes_online_retained(mqtt_setup):
    _, _, fake, _ = mqtt_setup
    fake.simulate_connect()
    avail = [p for p in fake.published if p[0] == "spookyeyes/availability"]
    assert avail == [("spookyeyes/availability", "online", 0, True)]


def test_failed_connect_does_nothing(mqtt_setup):
    _, _, fake, _ = mqtt_setup
    fake.simulate_connect(failure=True)
    assert fake.subscriptions == []
    assert fake.published == []


def test_reconnect_resubscribes_and_republishes_discovery(mqtt_setup):
    _, _, fake, _ = mqtt_setup
    fake.simulate_connect()
    n_subs, n_pubs = len(fake.subscriptions), len(fake.published)
    fake.simulate_connect()  # broker bounce -> paho reconnect
    assert len(fake.subscriptions) == 2 * n_subs
    assert len(fake.published) == 2 * n_pubs


def test_close_publishes_offline_and_stops(mqtt_setup):
    _, _, fake, inp = mqtt_setup
    inp.close()
    # qos 1 + wait_for_publish so the retained "offline" actually reaches the
    # broker before the graceful disconnect (which suppresses the LWT).
    assert ("spookyeyes/availability", "offline", 1, True) in fake.published
    assert fake.disconnected
    assert fake.loop_stopped
    inp.close()  # idempotent


# ---------------------------------------------------------------------------
# MQTT: command topic -> Event mapping
# ---------------------------------------------------------------------------


def test_theme_command(mqtt_setup):
    _, events, fake, _ = mqtt_setup
    fake.simulate_message("spookyeyes/cmd/theme", b"demon")
    assert drain(events) == [Event("theme", "demon")]


def test_theme_empty_payload_dropped(mqtt_setup):
    _, events, fake, _ = mqtt_setup
    fake.simulate_message("spookyeyes/cmd/theme", b"   ")
    assert drain(events) == []


@pytest.mark.parametrize("mode", ["idle", "scare", "stare", "sleep"])
def test_mode_command_valid(mqtt_setup, mode):
    _, events, fake, _ = mqtt_setup
    fake.simulate_message("spookyeyes/cmd/mode", mode.encode())
    (event,) = drain(events)
    assert event.kind == "mode"
    assert event.value == Mode(mode)


def test_mode_command_invalid_dropped(mqtt_setup):
    _, events, fake, _ = mqtt_setup
    fake.simulate_message("spookyeyes/cmd/mode", b"banana")
    assert drain(events) == []


def test_brightness_command(mqtt_setup):
    _, events, fake, _ = mqtt_setup
    fake.simulate_message("spookyeyes/cmd/brightness", b"0.55")
    (event,) = drain(events)
    assert event == Event("brightness", 0.55)


@pytest.mark.parametrize("payload,expected", [(b"1.7", 1.0), (b"-0.2", 0.0)])
def test_brightness_clamped(mqtt_setup, payload, expected):
    _, events, fake, _ = mqtt_setup
    fake.simulate_message("spookyeyes/cmd/brightness", payload)
    (event,) = drain(events)
    assert event.value == expected


@pytest.mark.parametrize("payload", [b"bright", b"", b"nan", b"inf"])
def test_brightness_invalid_dropped(mqtt_setup, payload):
    _, events, fake, _ = mqtt_setup
    fake.simulate_message("spookyeyes/cmd/brightness", payload)
    assert drain(events) == []


def test_blink_command_any_payload(mqtt_setup):
    _, events, fake, _ = mqtt_setup
    fake.simulate_message("spookyeyes/cmd/blink", b"PRESS")
    fake.simulate_message("spookyeyes/cmd/blink", b"")
    assert drain(events) == [Event("blink"), Event("blink")]


def test_non_utf8_payload_dropped_without_exception(mqtt_setup):
    _, events, fake, _ = mqtt_setup
    fake.simulate_message("spookyeyes/cmd/theme", b"\xff\xfe\x80")
    assert drain(events) == []


def test_unknown_topic_ignored(mqtt_setup):
    _, events, fake, _ = mqtt_setup
    fake.simulate_message("spookyeyes/other", b"x")
    assert drain(events) == []


# ---------------------------------------------------------------------------
# MQTT: Home Assistant discovery
# ---------------------------------------------------------------------------

DISCOVERY_TOPICS = {
    "homeassistant/select/spookyeyes_theme/config",
    "homeassistant/select/spookyeyes_mode/config",
    "homeassistant/number/spookyeyes_brightness/config",
    "homeassistant/button/spookyeyes_blink/config",
}


def discovery_payloads(fake: FakeClient) -> dict[str, dict]:
    out = {}
    for topic, payload, _qos, retain in fake.published:
        if topic in DISCOVERY_TOPICS:
            assert retain is True, f"discovery config {topic} must be retained"
            out[topic] = json.loads(payload)
    return out


def test_discovery_published_on_connect(mqtt_setup):
    _, _, fake, _ = mqtt_setup
    fake.simulate_connect()
    payloads = discovery_payloads(fake)
    assert set(payloads) == DISCOVERY_TOPICS


def test_discovery_common_keys_and_device_block(mqtt_setup):
    _, _, fake, _ = mqtt_setup
    fake.simulate_connect()
    unique_ids = set()
    for topic, payload in discovery_payloads(fake).items():
        assert payload["command_topic"].startswith("spookyeyes/cmd/"), topic
        assert payload["availability_topic"] == "spookyeyes/availability", topic
        assert payload["device"]["identifiers"] == ["spookyeyes"], topic
        assert payload["device"]["name"] == "Spooky Eyes", topic
        unique_ids.add(payload["unique_id"])
    assert len(unique_ids) == 4, "unique_id must differ per entity"


def test_discovery_entity_specifics(mqtt_setup):
    _, _, fake, _ = mqtt_setup
    fake.simulate_connect()
    payloads = discovery_payloads(fake)
    theme = payloads["homeassistant/select/spookyeyes_theme/config"]
    assert theme["options"] == ["human", "demon", "ghost"]
    assert theme["command_topic"] == "spookyeyes/cmd/theme"
    assert theme["state_topic"] == "spookyeyes/state/theme"
    mode = payloads["homeassistant/select/spookyeyes_mode/config"]
    assert set(mode["options"]) == {"idle", "scare", "stare", "sleep"}
    number = payloads["homeassistant/number/spookyeyes_brightness/config"]
    assert number["min"] == 0.0
    assert number["max"] == 1.0
    assert number["step"] == 0.05
    button = payloads["homeassistant/button/spookyeyes_blink/config"]
    assert button["command_topic"] == "spookyeyes/cmd/blink"


def test_discovery_disabled():
    cfg = MqttConfig(discovery=False)
    fake = FakeClient()
    inp = MqttInput(cfg, queue.Queue(), client_factory=lambda: fake)
    inp.start()
    fake.simulate_connect()
    assert discovery_payloads(fake) == {}
    # availability + subscriptions still happen
    assert len(fake.subscriptions) == 4
    assert ("spookyeyes/availability", "online", 0, True) in fake.published


# ---------------------------------------------------------------------------
# MQTT: publish_state
# ---------------------------------------------------------------------------


def test_publish_state_three_retained_topics(mqtt_setup):
    _, _, fake, inp = mqtt_setup
    inp.publish_state("ghost", "scare", 0.75)
    assert fake.published == [
        ("spookyeyes/state/theme", "ghost", 0, True),
        ("spookyeyes/state/mode", "scare", 0, True),
        ("spookyeyes/state/brightness", "0.75", 0, True),
    ]


def test_publish_state_accepts_mode_enum(mqtt_setup):
    _, _, fake, inp = mqtt_setup
    inp.publish_state("human", Mode.STARE, 1.0)
    assert ("spookyeyes/state/mode", "stare", 0, True) in fake.published


def test_publish_state_before_start_is_noop():
    inp = MqttInput(MqttConfig(), queue.Queue(), client_factory=FakeClient)
    inp.publish_state("human", "idle", 1.0)  # must not raise


def test_custom_base_topic():
    cfg = MqttConfig(base_topic="props/eyes")
    events: queue.Queue[Event] = queue.Queue()
    fake = FakeClient()
    inp = MqttInput(cfg, events, client_factory=lambda: fake)
    inp.start()
    fake.simulate_connect()
    assert ("props/eyes/cmd/mode", 0) in fake.subscriptions
    assert fake.will[0] == "props/eyes/availability"
    fake.simulate_message("props/eyes/cmd/mode", b"sleep")
    assert drain(events) == [Event("mode", Mode.SLEEP)]
    payloads = discovery_payloads(fake)
    theme = payloads["homeassistant/select/spookyeyes_theme/config"]
    assert theme["command_topic"] == "props/eyes/cmd/theme"
    assert theme["availability_topic"] == "props/eyes/availability"


# ---------------------------------------------------------------------------
# PIR
# ---------------------------------------------------------------------------


class FakeSensor:
    def __init__(self, pin: int) -> None:
        self.pin = pin
        self.when_motion = None
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


@pytest.fixture
def pir_setup():
    cfg = PirConfig(enabled=True, pin=17, cooldown=10.0)
    events: queue.Queue[Event] = queue.Queue()
    sensors: list[FakeSensor] = []

    def factory(pin: int) -> FakeSensor:
        sensors.append(FakeSensor(pin))
        return sensors[-1]

    clock = FakeClock()
    inp = PirInput(cfg, events, sensor_factory=factory, monotonic=clock)
    return events, sensors[0], clock, inp


def test_pir_wires_when_motion_to_configured_pin(pir_setup):
    _, sensor, _, _ = pir_setup
    assert sensor.pin == 17
    assert callable(sensor.when_motion)


def test_pir_first_motion_forwarded(pir_setup):
    events, sensor, _, _ = pir_setup
    sensor.when_motion()
    assert drain(events) == [Event("motion")]


def test_pir_cooldown_suppresses_second_event(pir_setup):
    events, sensor, clock, _ = pir_setup
    sensor.when_motion()
    clock.advance(9.99)
    sensor.when_motion()
    assert drain(events) == [Event("motion")]


def test_pir_after_cooldown_forwards_again(pir_setup):
    events, sensor, clock, _ = pir_setup
    sensor.when_motion()
    clock.advance(5.0)
    sensor.when_motion()  # suppressed; must NOT reset the cooldown window
    clock.advance(5.0)  # 10.0s since last *forwarded* event
    sensor.when_motion()
    assert drain(events) == [Event("motion"), Event("motion")]


def test_pir_close_releases_sensor(pir_setup):
    events, sensor, _, inp = pir_setup
    inp.close()
    assert sensor.closed
    assert sensor.when_motion is None
    inp.close()  # idempotent


def test_pir_unavailable_on_import_error():
    def factory(pin: int):
        raise ImportError("No module named 'gpiozero'")

    with pytest.raises(PirUnavailable):
        PirInput(PirConfig(), queue.Queue(), sensor_factory=factory)


def test_pir_unavailable_on_pin_error():
    def factory(pin: int):
        raise RuntimeError("BadPinFactory: no default pin factory")

    with pytest.raises(PirUnavailable):
        PirInput(PirConfig(), queue.Queue(), sensor_factory=factory)
