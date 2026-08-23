"""Shared pytest fixtures: a programmatically generated minimal theme so the
test suite runs without the real ``themes/`` assets.

The fixture art uses deliberately distinct solid colors per layer (sclera vs
iris vs pupil vs lid vs highlight) so rendering tests can classify pixels and
assert visible differences.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

FIXTURE_NAME = "testtheme"

# Distinct, easily separable layer colors (see docstring).
SCLERA_RGB = (205, 205, 165)     # pale yellow-white: high green, high red
IRIS_RGB = (30, 90, 220)         # saturated blue: high blue, low red
PUPIL_RGB = (0, 0, 0)            # black
HIGHLIGHT_RGB = (255, 255, 255)  # white
LID_RGB = (60, 18, 18)           # dark red-brown
BACKGROUND_RGB = (10, 5, 5)

FIXTURE_JSON: dict = {
    "name": FIXTURE_NAME,
    "background": list(BACKGROUND_RGB),
    "layers": {"sclera": "sclera.png", "iris": "iris.png", "highlight": "highlight.png"},
    "pupil": {
        "shape": "round",
        "color": list(PUPIL_RGB),
        "min_frac": 0.22,
        "max_frac": 0.55,
        "slit_width_frac": 0.14,
    },
    "iris_frac": 0.5,
    "gaze_range_px": 40,
    "sclera_parallax": 0.3,
    "eyelids": {"color": list(LID_RGB), "style": "curved", "upper_bias": 0.6},
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


def write_fixture_theme(themes_dir: Path, name: str = FIXTURE_NAME) -> None:
    """Write a minimal schema-complete theme dir under ``themes_dir/name``."""
    theme_dir = themes_dir / name
    theme_dir.mkdir(parents=True, exist_ok=True)

    Image.new("RGB", (320, 320), SCLERA_RGB).save(theme_dir / "sclera.png")

    # iris: opaque blue disc on a transparent 120x120 canvas (iris_frac 0.5 * 240)
    iris = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
    ImageDraw.Draw(iris).ellipse((0, 0, 119, 119), fill=(*IRIS_RGB, 255))
    iris.save(theme_dir / "iris.png")

    highlight = Image.new("RGBA", (24, 24), (0, 0, 0, 0))
    ImageDraw.Draw(highlight).ellipse((0, 0, 23, 23), fill=(*HIGHLIGHT_RGB, 230))
    highlight.save(theme_dir / "highlight.png")

    payload = dict(FIXTURE_JSON, name=name)
    (theme_dir / "theme.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


@pytest.fixture()
def fixture_theme(tmp_path: Path) -> tuple[Path, str]:
    """Function-scoped: a fresh themes dir containing one valid theme.

    Returns ``(themes_dir, name)`` for ``themes.load_theme(themes_dir, name)``.
    Tests may freely mutate/delete files inside — each test gets its own copy.
    """
    themes_dir = tmp_path / "themes"
    write_fixture_theme(themes_dir)
    return themes_dir, FIXTURE_NAME
