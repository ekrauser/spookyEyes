"""MQTT input: maps broker command topics to model.Event and integrates with
Home Assistant via MQTT discovery.

Uses the paho-mqtt v2 callback API (CallbackAPIVersion.VERSION2). paho is an
optional dependency and is imported lazily inside the default client factory,
so ``import spookyeyes.inputs.mqtt`` works without it (tests inject a fake
client via ``client_factory``).
"""

from __future__ import annotations

import json
import logging
import math
import queue
from typing import Any, Callable

from ..config import MqttConfig
from ..model import Event, Mode

log = logging.getLogger("spookyeyes.inputs.mqtt")

THEME_OPTIONS = ["human", "demon", "ghost"]


class MqttInput:
    """Background MQTT client pushing Events onto the app queue.

    Threading: paho runs its network loop in its own thread (``loop_start``);
    the callbacks below only ever touch the thread-safe event queue. State is
    published from the app thread via :meth:`publish_state`.
    """

    def __init__(
        self,
        cfg: MqttConfig,
        events: queue.Queue[Event],
        client_factory: Callable[[], Any] | None = None,
        theme_options: list[str] | None = None,
    ) -> None:
        self._cfg = cfg
        self._events = events
        self._client_factory = client_factory or self._default_client_factory
        self._client: Any | None = None
        self._theme_options = list(theme_options) if theme_options else list(THEME_OPTIONS)
        # Optional callback returning (theme, mode, brightness); when set, the
        # current state is (re)published on every connect so retained state
        # topics reflect reality after restarts and broker reconnects.
        self.state_provider: Callable[[], tuple[str, str, float]] | None = None
        base = cfg.base_topic
        self.availability_topic = f"{base}/availability"
        self._cmd_handlers: dict[str, Callable[[str], None]] = {
            f"{base}/cmd/theme": self._handle_theme,
            f"{base}/cmd/mode": self._handle_mode,
            f"{base}/cmd/brightness": self._handle_brightness,
            f"{base}/cmd/blink": self._handle_blink,
        }

    # -- lifecycle ---------------------------------------------------------

    def _default_client_factory(self) -> Any:
        import paho.mqtt.client as paho  # lazy: optional dependency

        return paho.Client(
            callback_api_version=paho.CallbackAPIVersion.VERSION2,
            client_id=self._cfg.client_id,
        )

    def start(self) -> None:
        """Create the client and start the background network thread.

        Uses ``connect_async`` + ``loop_start`` so a broker that is down (or
        not yet up) does not block or crash the app; paho retries with the
        configured backoff and fires ``on_connect`` whenever it gets through.
        """
        cfg = self._cfg
        client = self._client_factory()
        self._client = client
        if cfg.username:
            client.username_pw_set(cfg.username, cfg.password)
        # Last-will: broker marks us offline if we vanish ungracefully.
        client.will_set(self.availability_topic, "offline", qos=0, retain=True)
        client.reconnect_delay_set(min_delay=1, max_delay=30)
        client.on_connect = self._on_connect
        client.on_message = self._on_message
        client.connect_async(cfg.host, cfg.port, keepalive=60)
        client.loop_start()
        log.info("mqtt: connecting to %s:%d (base topic %r)", cfg.host, cfg.port, cfg.base_topic)

    def close(self) -> None:
        """Publish offline availability and shut the network thread down."""
        client = self._client
        if client is None:
            return
        self._client = None
        try:
            # Graceful disconnect does not trigger the LWT, so say goodbye
            # explicitly — and wait for the network thread to actually send it
            # (qos 1), or the broker would retain "online" forever.
            info = client.publish(self.availability_topic, "offline", qos=1, retain=True)
            try:
                if info is not None and hasattr(info, "wait_for_publish"):
                    info.wait_for_publish(timeout=2.0)
            except Exception:
                pass  # best effort; the disconnect below proceeds regardless
            client.disconnect()
            client.loop_stop()
        except Exception:
            log.exception("mqtt: error during close")

    # -- paho callbacks (never raise out of these) -------------------------

    def _on_connect(
        self,
        client: Any,
        userdata: Any,
        flags: Any,
        reason_code: Any,
        properties: Any = None,
    ) -> None:
        try:
            if getattr(reason_code, "is_failure", False):
                log.warning("mqtt: connection refused: %s", reason_code)
                return
            log.info("mqtt: connected")
            # (Re)subscribe on every connect: a reconnect with a clean session
            # loses server-side subscriptions.
            for topic in self._cmd_handlers:
                client.subscribe(topic, 0)
            client.publish(self.availability_topic, "online", qos=0, retain=True)
            if self._cfg.discovery:
                self._publish_discovery(client)
            provider = self.state_provider
            if provider is not None:
                theme, mode, brightness = provider()
                self.publish_state(theme, mode, brightness)
        except Exception:
            log.exception("mqtt: error in on_connect")

    def _on_message(self, client: Any, userdata: Any, message: Any) -> None:
        try:
            handler = self._cmd_handlers.get(message.topic)
            if handler is None:
                log.debug("mqtt: ignoring message on %r", message.topic)
                return
            try:
                payload = message.payload.decode("utf-8").strip()
            except UnicodeDecodeError:
                log.warning("mqtt: non-UTF-8 payload on %r, dropped", message.topic)
                return
            handler(payload)
        except Exception:
            log.exception("mqtt: error handling message on %r", getattr(message, "topic", "?"))

    # -- per-topic payload handlers ----------------------------------------

    def _handle_theme(self, payload: str) -> None:
        if not payload:
            log.warning("mqtt: empty theme payload, dropped")
            return
        self._events.put(Event("theme", payload))

    def _handle_mode(self, payload: str) -> None:
        try:
            mode = Mode(payload.lower())
        except ValueError:
            log.warning("mqtt: invalid mode %r, dropped (want one of %s)",
                        payload, "|".join(m.value for m in Mode))
            return
        self._events.put(Event("mode", mode))

    def _handle_brightness(self, payload: str) -> None:
        try:
            value = float(payload)
        except ValueError:
            log.warning("mqtt: invalid brightness %r, dropped", payload)
            return
        if not math.isfinite(value):
            log.warning("mqtt: non-finite brightness %r, dropped", payload)
            return
        self._events.put(Event("brightness", min(1.0, max(0.0, value))))

    def _handle_blink(self, payload: str) -> None:
        self._events.put(Event("blink"))

    # -- publishing (called from the app thread) ---------------------------

    def publish_state(self, theme: str, mode: str, brightness: float) -> None:
        """Publish retained state so HA entities reflect reality after changes."""
        client = self._client
        if client is None:
            return
        base = self._cfg.base_topic
        mode_str = mode.value if isinstance(mode, Mode) else str(mode)
        try:
            client.publish(f"{base}/state/theme", theme, qos=0, retain=True)
            client.publish(f"{base}/state/mode", mode_str, qos=0, retain=True)
            client.publish(f"{base}/state/brightness", format(brightness, "g"),
                           qos=0, retain=True)
        except Exception:
            log.exception("mqtt: error publishing state")

    # -- Home Assistant MQTT discovery -------------------------------------

    def _publish_discovery(self, client: Any) -> None:
        base = self._cfg.base_topic
        device = {
            "identifiers": ["spookyeyes"],
            "name": "Spooky Eyes",
            "manufacturer": "spookyeyes",
            "model": "Raspberry Pi dual GC9A01A",
        }
        common = {
            "availability_topic": self.availability_topic,
            "device": device,
        }
        configs: dict[str, dict[str, Any]] = {
            "homeassistant/select/spookyeyes_theme/config": {
                "name": "Theme",
                "unique_id": "spookyeyes_theme",
                "command_topic": f"{base}/cmd/theme",
                "state_topic": f"{base}/state/theme",
                "options": self._theme_options,
                "icon": "mdi:drama-masks",
                **common,
            },
            "homeassistant/select/spookyeyes_mode/config": {
                "name": "Mode",
                "unique_id": "spookyeyes_mode",
                "command_topic": f"{base}/cmd/mode",
                "state_topic": f"{base}/state/mode",
                "options": [m.value for m in Mode],
                "icon": "mdi:eye-settings",
                **common,
            },
            "homeassistant/number/spookyeyes_brightness/config": {
                "name": "Brightness",
                "unique_id": "spookyeyes_brightness",
                "command_topic": f"{base}/cmd/brightness",
                "state_topic": f"{base}/state/brightness",
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "mode": "slider",
                "icon": "mdi:brightness-6",
                **common,
            },
            "homeassistant/button/spookyeyes_blink/config": {
                "name": "Blink",
                "unique_id": "spookyeyes_blink",
                "command_topic": f"{base}/cmd/blink",
                "payload_press": "blink",
                "icon": "mdi:eye-refresh",
                **common,
            },
        }
        for topic, payload in configs.items():
            client.publish(topic, json.dumps(payload), qos=0, retain=True)
        log.info("mqtt: published Home Assistant discovery configs")
