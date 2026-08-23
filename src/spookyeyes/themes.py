"""Theme loading: parse ``theme.json`` and load the layer art into a `Theme`.

A theme lives in ``<themes_dir>/<name>/`` and contains ``theme.json`` plus the
PNG layers it references (see DESIGN.md for the authoritative schema). Missing
or invalid required pieces raise `ThemeError`; the optional highlight layer
degrades to ``None``; unknown JSON keys are ignored.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, fields
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .model import SIZE, MotionParams

logger = logging.getLogger("spookyeyes.themes")

_PUPIL_SHAPES = ("round", "slit", "none")
_LID_STYLES = ("curved", "straight")
_MOTION_PAIR_KEYS = frozenset({"saccade_interval", "saccade_duration", "blink_interval"})


class ThemeError(Exception):
    """A theme directory is missing, its JSON is malformed, or required art is bad."""


@dataclass
class Theme:
    """Parsed theme.json plus loaded PIL images. Produced by `load_theme`,
    consumed by `eye.EyeRenderer`."""

    name: str
    path: Path
    background: tuple[int, int, int]
    sclera: Image.Image          # RGB, square, side >= 240 (typically 320)
    iris: Image.Image            # RGBA
    highlight: Image.Image | None  # RGBA, optional
    pupil_shape: str             # "round" | "slit" | "none"
    pupil_color: tuple[int, int, int]
    pupil_min_frac: float
    pupil_max_frac: float
    slit_width_frac: float
    iris_frac: float             # iris diameter relative to 240
    gaze_range_px: float
    sclera_parallax: float
    lid_color: tuple[int, int, int]
    lid_style: str               # "curved" | "straight"
    lid_upper_bias: float
    motion: MotionParams
    # Iris rotation (the "Arcana" themes): revolutions per minute, and whether
    # the right eye should counter-rotate. 0 = static (the default).
    iris_spin_rpm: float = 0.0
    iris_spin_mirror: bool = False


def _num(value: object, key: str, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ThemeError(f"theme {name!r}: {key} must be a number, got {value!r}")
    return float(value)


def _rgb(value: object, key: str, name: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(isinstance(c, bool) or not isinstance(c, int) for c in value)
        or any(not 0 <= c <= 255 for c in value)
    ):
        raise ThemeError(
            f"theme {name!r}: {key} must be an [r, g, b] list of ints 0..255, got {value!r}"
        )
    r, g, b = value
    return (r, g, b)


def _pair(value: object, key: str, name: str) -> tuple[float, float]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value)
    ):
        raise ThemeError(f"theme {name!r}: {key} must be a [lo, hi] pair of numbers, got {value!r}")
    lo, hi = value
    return (float(lo), float(hi))


def _section(data: dict, key: str, name: str) -> dict:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ThemeError(f"theme {name!r}: {key!r} must be a JSON object, got {value!r}")
    return value


def _load_required_image(theme_dir: Path, layers: dict, key: str, mode: str, name: str) -> Image.Image:
    filename = layers.get(key)
    if not isinstance(filename, str) or not filename:
        raise ThemeError(f"theme {name!r}: layers.{key} missing from theme.json")
    path = theme_dir / filename
    try:
        with Image.open(path) as im:
            im.load()
            return im.convert(mode)
    except FileNotFoundError as exc:
        raise ThemeError(f"theme {name!r}: required art not found: {path}") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ThemeError(f"theme {name!r}: cannot load {path}: {exc}") from exc


def _load_optional_image(theme_dir: Path, layers: dict, key: str, mode: str, name: str) -> Image.Image | None:
    filename = layers.get(key)
    if filename is None:
        return None
    if not isinstance(filename, str) or not filename:
        logger.warning("theme %r: layers.%s is not a filename (%r); ignoring", name, key, filename)
        return None
    path = theme_dir / filename
    try:
        with Image.open(path) as im:
            im.load()
            return im.convert(mode)
    except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError) as exc:
        logger.warning("theme %r: optional art %s unusable (%s); continuing without it", name, path, exc)
        return None


def _parse_motion(data: dict, name: str) -> MotionParams:
    valid = {f.name for f in fields(MotionParams)}
    kwargs: dict[str, object] = {}
    for key, value in data.items():
        if key not in valid:
            logger.debug("theme %r: ignoring unknown motion key %r", name, key)
            continue
        if key in _MOTION_PAIR_KEYS:
            kwargs[key] = _pair(value, f"motion.{key}", name)
        else:
            kwargs[key] = _num(value, f"motion.{key}", name)
    return MotionParams(**kwargs)  # type: ignore[arg-type]


def load_theme(themes_dir: Path, name: str) -> Theme:
    """Load ``<themes_dir>/<name>/theme.json`` and its art layers.

    Raises `ThemeError` on a missing theme, malformed JSON, missing/invalid
    required art, or out-of-range enum values. Unknown JSON keys are ignored.
    """
    # Theme names arrive from untrusted places (MQTT); a bare join would let
    # "../x" or an absolute path escape the themes directory entirely.
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ThemeError(
            f"invalid theme name {name!r} (letters, digits, '-' and '_' only)"
        )
    theme_dir = Path(themes_dir) / name
    json_path = theme_dir / "theme.json"
    if not json_path.is_file():
        raise ThemeError(f"theme {name!r}: no theme.json at {json_path}")
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ThemeError(f"theme {name!r}: cannot parse {json_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ThemeError(f"theme {name!r}: theme.json must contain a JSON object")

    layers = data.get("layers")
    if not isinstance(layers, dict):
        raise ThemeError(f"theme {name!r}: theme.json is missing the \"layers\" object")

    sclera = _load_required_image(theme_dir, layers, "sclera", "RGB", name)
    iris = _load_required_image(theme_dir, layers, "iris", "RGBA", name)
    highlight = _load_optional_image(theme_dir, layers, "highlight", "RGBA", name)

    sw, sh = sclera.size
    if sw != sh or sw < SIZE:
        raise ThemeError(
            f"theme {name!r}: sclera must be square with side >= {SIZE}px, got {sw}x{sh}"
        )

    pupil = _section(data, "pupil", name)
    pupil_shape = pupil.get("shape", "round")
    if pupil_shape not in _PUPIL_SHAPES:
        raise ThemeError(
            f"theme {name!r}: pupil.shape must be one of {_PUPIL_SHAPES}, got {pupil_shape!r}"
        )

    eyelids = _section(data, "eyelids", name)
    lid_style = eyelids.get("style", "curved")
    if lid_style not in _LID_STYLES:
        raise ThemeError(
            f"theme {name!r}: eyelids.style must be one of {_LID_STYLES}, got {lid_style!r}"
        )

    theme = Theme(
        name=name,
        path=theme_dir,
        background=_rgb(data.get("background", [0, 0, 0]), "background", name),
        sclera=sclera,
        iris=iris,
        highlight=highlight,
        pupil_shape=pupil_shape,
        pupil_color=_rgb(pupil.get("color", [0, 0, 0]), "pupil.color", name),
        pupil_min_frac=_num(pupil.get("min_frac", 0.22), "pupil.min_frac", name),
        pupil_max_frac=_num(pupil.get("max_frac", 0.55), "pupil.max_frac", name),
        slit_width_frac=_num(pupil.get("slit_width_frac", 0.14), "pupil.slit_width_frac", name),
        iris_frac=_num(data.get("iris_frac", 0.6), "iris_frac", name),
        gaze_range_px=_num(data.get("gaze_range_px", 50), "gaze_range_px", name),
        sclera_parallax=_num(data.get("sclera_parallax", 0.35), "sclera_parallax", name),
        lid_color=_rgb(eyelids.get("color", [0, 0, 0]), "eyelids.color", name),
        lid_style=lid_style,
        lid_upper_bias=_num(eyelids.get("upper_bias", 0.6), "eyelids.upper_bias", name),
        motion=_parse_motion(_section(data, "motion", name), name),
        iris_spin_rpm=_num(data.get("iris_spin_rpm", 0.0), "iris_spin_rpm", name),
        iris_spin_mirror=bool(data.get("iris_spin_mirror", False)),
    )
    if theme.iris_spin_rpm and theme.pupil_shape != "none":
        raise ThemeError(
            f'theme {name!r}: iris_spin_rpm requires pupil shape "none" '
            f"(got {theme.pupil_shape!r}) — a dilating pupil cannot be pre-baked "
            f"together with rotation"
        )
    logger.info("loaded theme %r from %s", name, theme_dir)
    return theme
