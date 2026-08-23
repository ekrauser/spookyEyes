"""TOML configuration loading. Fully implemented — do not change field names,
they are referenced across modules and documented in config.example.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path


class ConfigError(Exception):
    """Invalid configuration file contents (wrong value type or encoding)."""


def _checked(section: str, key: str, value: object, default: object) -> object:
    """Validate a TOML value against the type of the dataclass default so a
    wrong-typed config fails at load time with a clear message, not with a
    TypeError deep inside the frame loop or a gpiozero callback thread."""
    err = ConfigError(
        f"[{section}] {key}: expected {type(default).__name__}, got {value!r}"
    )
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise err
        return value
    if isinstance(default, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise err
        return value
    if isinstance(default, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise err
        return float(value)
    if isinstance(default, str):
        if not isinstance(value, str):
            raise err
        return value
    return value


@dataclass
class DisplayConfig:
    output: str = "preview"       # "preview" | "fb" | "record" | "null"
    # Stable SPI addresses (resolved to /dev/fbN via sysfs at startup — fbdev
    # minor numbers follow async probe order and can swap between boots).
    # Plain /dev/fbN paths are also accepted.
    fb_left: str = "spi0.0"
    fb_right: str = "spi0.1"
    fps: int = 30
    mirror_left: bool = False     # horizontal flip for mirrored mounting
    mirror_right: bool = False
    scale: int = 2                # preview window scale factor
    record_dir: str = "out"       # record output directory


@dataclass
class ThemeConfig:
    name: str = "human"
    dir: str = "themes"


@dataclass
class MqttConfig:
    enabled: bool = False
    host: str = "homeassistant.local"
    port: int = 1883
    username: str = ""
    password: str = ""
    base_topic: str = "spookyeyes"
    discovery: bool = True        # publish Home Assistant MQTT discovery configs
    client_id: str = "spookyeyes"


@dataclass
class PirConfig:
    enabled: bool = False
    pin: int = 17                 # BCM
    cooldown: float = 10.0        # s between motion events forwarded to behavior


@dataclass
class AppConfig:
    display: DisplayConfig = field(default_factory=DisplayConfig)
    theme: ThemeConfig = field(default_factory=ThemeConfig)
    mqtt: MqttConfig = field(default_factory=MqttConfig)
    pir: PirConfig = field(default_factory=PirConfig)

    @classmethod
    def load(cls, path: str | Path | None) -> "AppConfig":
        """Load from a TOML file; missing file/sections/keys fall back to
        defaults, unknown keys are ignored."""
        cfg = cls()
        if path is None:
            return cfg
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p}")
        try:
            text = p.read_text(encoding="utf-8")  # TOML is UTF-8 by spec
        except UnicodeDecodeError as exc:
            raise ConfigError(f"{p} is not valid UTF-8: {exc}") from exc
        data = tomllib.loads(text)
        for f in fields(cfg):
            section = data.get(f.name)
            if not isinstance(section, dict):
                continue
            sub = getattr(cfg, f.name)
            assert is_dataclass(sub)
            valid = {sf.name for sf in fields(sub)}
            for key, value in section.items():
                if key in valid:
                    setattr(sub, key, _checked(f.name, key, value, getattr(sub, key)))
        return cfg
