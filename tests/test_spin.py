"""Tests for the pre-baked spinning-iris feature (Arcana themes)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from spookyeyes.eye import EyeRenderer
from spookyeyes.model import EyeState
from spookyeyes.themes import ThemeError, load_theme


def _spinified(fixture_theme, rpm: float = 60.0, mirror: bool = False,
               pupil: str = "none"):
    """Turn the fixture theme into a spin theme with an asymmetric iris
    (rotations of a rotationally-symmetric disc would be invisible)."""
    themes_dir, name = fixture_theme
    tdir = themes_dir / name
    spec = json.loads((tdir / "theme.json").read_text())
    spec["pupil"]["shape"] = pupil
    spec["iris_spin_rpm"] = rpm
    spec["iris_spin_mirror"] = mirror
    (tdir / "theme.json").write_text(json.dumps(spec))
    im = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    draw.ellipse((10, 10, 110, 110), fill=(200, 40, 40, 255))
    draw.pieslice((10, 10, 110, 110), 0, 90, fill=(40, 40, 200, 255))
    im.save(tdir / "iris.png")
    return themes_dir, name


def test_spin_rotates_over_time(fixture_theme):
    themes_dir, name = _spinified(fixture_theme)  # 60 rpm = 1 rev/s exactly
    r = EyeRenderer(load_theme(themes_dir, name))
    s = EyeState()
    f0 = r.render(s, 0.0)
    quarter = r.render(s, 0.25)
    assert not np.array_equal(f0, quarter)
    assert np.array_equal(f0, r.render(s, 1.0))  # full revolution wraps exactly


def test_spin_direction_mirrors(fixture_theme):
    themes_dir, name = _spinified(fixture_theme)
    theme = load_theme(themes_dir, name)
    fwd = EyeRenderer(theme, spin_dir=1).render(EyeState(), 0.1)
    rev = EyeRenderer(theme, spin_dir=-1).render(EyeState(), 0.1)
    assert not np.array_equal(fwd, rev)


def test_no_spin_ignores_time(fixture_theme):
    themes_dir, name = fixture_theme
    r = EyeRenderer(load_theme(themes_dir, name))
    s = EyeState()
    assert np.array_equal(r.render(s, 0.0), r.render(s, 7.3))


def test_spin_requires_pupil_none(fixture_theme):
    themes_dir, name = _spinified(fixture_theme, pupil="round")
    with pytest.raises(ThemeError, match="iris_spin"):
        load_theme(themes_dir, name)


def test_spin_defaults_absent(fixture_theme):
    themes_dir, name = fixture_theme
    theme = load_theme(themes_dir, name)
    assert theme.iris_spin_rpm == 0.0
    assert theme.iris_spin_mirror is False
