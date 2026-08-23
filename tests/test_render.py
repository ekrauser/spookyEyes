"""Tests for themes.load_theme and eye.EyeRenderer (agent RENDER)."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

import conftest as fx
from spookyeyes.eye import LID_STEPS, MAX_OPENNESS, PUPIL_STEPS, EyeRenderer
from spookyeyes.model import SIZE, EyeState, MotionParams
from spookyeyes.themes import Theme, ThemeError, load_theme

FixtureTheme = tuple[Path, str]


def _mutate_json(fixture_theme: FixtureTheme, mutate: Callable[[dict], None]) -> None:
    themes_dir, name = fixture_theme
    path = themes_dir / name / "theme.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def _load(fixture_theme: FixtureTheme) -> Theme:
    themes_dir, name = fixture_theme
    return load_theme(themes_dir, name)


def _renderer(
    fixture_theme: FixtureTheme, mutate: Callable[[dict], None] | None = None
) -> EyeRenderer:
    if mutate is not None:
        _mutate_json(fixture_theme, mutate)
    return EyeRenderer(_load(fixture_theme))


def _iris_mask(frame: np.ndarray) -> np.ndarray:
    """Pixels that look like the fixture's blue iris (not sclera/lid/pupil/highlight)."""
    return (frame[:, :, 2] > 150) & (frame[:, :, 0] < 100)


def _iris_centroid(frame: np.ndarray) -> tuple[float, float]:
    mask = _iris_mask(frame)
    assert mask.sum() > 200, "expected a visible iris in the frame"
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def _dark_count(frame: np.ndarray) -> int:
    """Near-black pixels in the central region (pupil territory; excludes the
    vignette corners, which are always black)."""
    half = SIZE // 2
    region = frame[half - 55 : half + 55, half - 55 : half + 55]
    return int((region.max(axis=2) < 40).sum())


# --------------------------------------------------------------------- themes


class TestLoadTheme:
    def test_parsed_fields(self, fixture_theme: FixtureTheme) -> None:
        theme = _load(fixture_theme)
        themes_dir, name = fixture_theme
        assert theme.name == fx.FIXTURE_NAME
        assert theme.path == themes_dir / name
        assert theme.background == fx.BACKGROUND_RGB
        assert theme.sclera.mode == "RGB" and theme.sclera.size == (320, 320)
        assert theme.iris.mode == "RGBA" and theme.iris.size == (120, 120)
        assert theme.highlight is not None and theme.highlight.mode == "RGBA"
        assert theme.pupil_shape == "round"
        assert theme.pupil_color == fx.PUPIL_RGB
        assert theme.pupil_min_frac == pytest.approx(0.22)
        assert theme.pupil_max_frac == pytest.approx(0.55)
        assert theme.slit_width_frac == pytest.approx(0.14)
        assert theme.iris_frac == pytest.approx(0.5)
        assert theme.gaze_range_px == pytest.approx(40.0)
        assert theme.sclera_parallax == pytest.approx(0.3)
        assert theme.lid_color == fx.LID_RGB
        assert theme.lid_style == "curved"
        assert theme.lid_upper_bias == pytest.approx(0.6)
        assert isinstance(theme.motion, MotionParams)
        assert theme.motion.saccade_interval == (0.6, 3.0)
        assert theme.motion.saccade_duration == (0.09, 0.22)
        assert theme.motion.blink_interval == (2.0, 8.0)
        assert theme.motion.wander == pytest.approx(0.35)
        assert theme.motion.crazy == pytest.approx(0.1)
        assert theme.motion.flicker == pytest.approx(0.0)

    def test_missing_theme_dir_raises(self, fixture_theme: FixtureTheme) -> None:
        themes_dir, _ = fixture_theme
        with pytest.raises(ThemeError):
            load_theme(themes_dir, "no-such-theme")

    def test_missing_sclera_art_raises(self, fixture_theme: FixtureTheme) -> None:
        themes_dir, name = fixture_theme
        (themes_dir / name / "sclera.png").unlink()
        with pytest.raises(ThemeError, match="sclera"):
            load_theme(themes_dir, name)

    def test_missing_iris_art_raises(self, fixture_theme: FixtureTheme) -> None:
        themes_dir, name = fixture_theme
        (themes_dir / name / "iris.png").unlink()
        with pytest.raises(ThemeError, match="iris"):
            load_theme(themes_dir, name)

    def test_missing_iris_layer_key_raises(self, fixture_theme: FixtureTheme) -> None:
        _mutate_json(fixture_theme, lambda d: d["layers"].pop("iris"))
        with pytest.raises(ThemeError, match="iris"):
            _load(fixture_theme)

    def test_invalid_json_raises(self, fixture_theme: FixtureTheme) -> None:
        themes_dir, name = fixture_theme
        (themes_dir / name / "theme.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(ThemeError):
            load_theme(themes_dir, name)

    def test_corrupt_required_art_raises(self, fixture_theme: FixtureTheme) -> None:
        themes_dir, name = fixture_theme
        (themes_dir / name / "iris.png").write_bytes(b"this is not a png")
        with pytest.raises(ThemeError):
            load_theme(themes_dir, name)

    def test_bad_pupil_shape_raises(self, fixture_theme: FixtureTheme) -> None:
        _mutate_json(fixture_theme, lambda d: d["pupil"].__setitem__("shape", "square"))
        with pytest.raises(ThemeError, match="shape"):
            _load(fixture_theme)

    def test_bad_lid_style_raises(self, fixture_theme: FixtureTheme) -> None:
        _mutate_json(fixture_theme, lambda d: d["eyelids"].__setitem__("style", "wavy"))
        with pytest.raises(ThemeError, match="style"):
            _load(fixture_theme)

    def test_non_square_sclera_raises(self, fixture_theme: FixtureTheme) -> None:
        from PIL import Image

        themes_dir, name = fixture_theme
        Image.new("RGB", (320, 300), fx.SCLERA_RGB).save(themes_dir / name / "sclera.png")
        with pytest.raises(ThemeError, match="square"):
            load_theme(themes_dir, name)

    def test_too_small_sclera_raises(self, fixture_theme: FixtureTheme) -> None:
        from PIL import Image

        themes_dir, name = fixture_theme
        Image.new("RGB", (200, 200), fx.SCLERA_RGB).save(themes_dir / name / "sclera.png")
        with pytest.raises(ThemeError):
            load_theme(themes_dir, name)

    def test_missing_highlight_gives_none(self, fixture_theme: FixtureTheme) -> None:
        themes_dir, name = fixture_theme
        (themes_dir / name / "highlight.png").unlink()
        theme = load_theme(themes_dir, name)
        assert theme.highlight is None
        # and the renderer copes without a highlight
        frame = EyeRenderer(theme).render(EyeState())
        assert frame.shape == (SIZE, SIZE, 3)

    def test_unknown_keys_ignored(self, fixture_theme: FixtureTheme) -> None:
        def add_junk(data: dict) -> None:
            data["totally_unknown"] = {"nested": True}
            data["pupil"]["sparkle"] = 9
            data["motion"]["warp_factor"] = 9.9

        _mutate_json(fixture_theme, add_junk)
        theme = _load(fixture_theme)
        assert theme.motion.wander == pytest.approx(0.35)

    def test_missing_optional_sections_use_defaults(self, fixture_theme: FixtureTheme) -> None:
        def strip(data: dict) -> None:
            for key in ("background", "pupil", "eyelids", "motion", "iris_frac",
                        "gaze_range_px", "sclera_parallax"):
                data.pop(key, None)

        _mutate_json(fixture_theme, strip)
        theme = _load(fixture_theme)
        assert theme.pupil_shape == "round"
        assert theme.lid_style == "curved"
        assert theme.motion == MotionParams()
        assert EyeRenderer(theme).render(EyeState()).shape == (SIZE, SIZE, 3)


# --------------------------------------------------------------------- render


class TestRender:
    def test_bake_step_constants(self) -> None:
        assert EyeRenderer.PUPIL_STEPS == PUPIL_STEPS == 12
        assert EyeRenderer.LID_STEPS == LID_STEPS == 24

    def test_frame_shape_and_dtype(self, fixture_theme: FixtureTheme) -> None:
        frame = _renderer(fixture_theme).render(EyeState())
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (SIZE, SIZE, 3)
        assert frame.dtype == np.uint8

    def test_gaze_x_moves_iris(self, fixture_theme: FixtureTheme) -> None:
        renderer = _renderer(fixture_theme)
        left = renderer.render(EyeState(gaze_x=-0.8))
        right = renderer.render(EyeState(gaze_x=0.8))
        cx_left, _ = _iris_centroid(left)
        cx_right, _ = _iris_centroid(right)
        # gaze_range_px=40 -> expected shift ~2*0.8*40 = 64 px
        assert cx_right - cx_left > 30, (cx_left, cx_right)

    def test_gaze_y_positive_is_up_on_screen(self, fixture_theme: FixtureTheme) -> None:
        renderer = _renderer(fixture_theme)
        up = renderer.render(EyeState(gaze_y=0.8))
        down = renderer.render(EyeState(gaze_y=-0.8))
        _, cy_up = _iris_centroid(up)
        _, cy_down = _iris_centroid(down)
        # screen y grows downward, so looking up means a smaller row index
        assert cy_down - cy_up > 30, (cy_up, cy_down)

    def test_openness_zero_is_lid_colored(self, fixture_theme: FixtureTheme) -> None:
        renderer = _renderer(fixture_theme)
        closed = renderer.render(EyeState(openness=0.0))
        opened = renderer.render(EyeState(openness=1.0))
        lid = np.array(fx.LID_RGB, dtype=np.uint8)
        # center of a closed eye is exactly lid color (vignette is 1.0 there)
        assert np.all(closed[100:140, 100:140] == lid)
        # closed: no sclera (high green) and no iris (blue) anywhere
        assert not np.any(closed[:, :, 1] > 150)
        assert not np.any(_iris_mask(closed))
        # open: sclera and iris clearly visible
        assert np.any(opened[:, :, 1] > 150)
        assert _iris_mask(opened).sum() > 1000

    def test_openness_startled_shows_more_than_relaxed(self, fixture_theme: FixtureTheme) -> None:
        renderer = _renderer(fixture_theme)
        relaxed = renderer.render(EyeState(openness=1.0))
        startled = renderer.render(EyeState(openness=MAX_OPENNESS))
        # lids retract fully at 1.25, exposing more sclera than at 1.0
        assert int((startled[:, :, 1] > 150).sum()) > int((relaxed[:, :, 1] > 150).sum())

    def test_pupil_round_dilation_changes_frame(self, fixture_theme: FixtureTheme) -> None:
        renderer = _renderer(fixture_theme)
        constricted = renderer.render(EyeState(pupil=0.0))
        dilated = renderer.render(EyeState(pupil=1.0))
        assert not np.array_equal(constricted, dilated)
        assert _dark_count(dilated) > _dark_count(constricted) + 300

    def test_pupil_slit_dilation_changes_frame(self, fixture_theme: FixtureTheme) -> None:
        renderer = _renderer(
            fixture_theme, mutate=lambda d: d["pupil"].__setitem__("shape", "slit")
        )
        constricted = renderer.render(EyeState(pupil=0.0))
        dilated = renderer.render(EyeState(pupil=1.0))
        assert not np.array_equal(constricted, dilated)
        assert _dark_count(dilated) > _dark_count(constricted) + 100

    def test_pupil_none_is_static_across_dilation(self, fixture_theme: FixtureTheme) -> None:
        renderer = _renderer(
            fixture_theme, mutate=lambda d: d["pupil"].__setitem__("shape", "none")
        )
        frame_a = renderer.render(EyeState(pupil=0.0))
        frame_b = renderer.render(EyeState(pupil=1.0))
        assert np.array_equal(frame_a, frame_b)

    def test_straight_lid_style(self, fixture_theme: FixtureTheme) -> None:
        renderer = _renderer(
            fixture_theme, mutate=lambda d: d["eyelids"].__setitem__("style", "straight")
        )
        closed = renderer.render(EyeState(openness=0.0))
        assert np.all(closed[110:130, 110:130] == np.array(fx.LID_RGB, dtype=np.uint8))
        half = renderer.render(EyeState(openness=0.5))
        # upper lid covers the top-center at half openness...
        assert tuple(half[40, 120]) == fx.LID_RGB
        # ...but the eye midline shows iris (sampled off the central pupil)
        assert _iris_mask(half)[120, 80]

    def test_corners_and_outside_vignette_black(self, fixture_theme: FixtureTheme) -> None:
        renderer = _renderer(fixture_theme)
        # brightest plausible frame: wide open, bright
        frame = renderer.render(EyeState(openness=MAX_OPENNESS, brightness=1.0))
        for y, x in ((0, 0), (0, SIZE - 1), (SIZE - 1, 0), (SIZE - 1, SIZE - 1)):
            assert tuple(frame[y, x]) == (0, 0, 0)
        c = (SIZE - 1) / 2.0
        yy, xx = np.mgrid[0:SIZE, 0:SIZE]
        outside = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) > SIZE / 2.0 + 1.5
        assert not frame[outside].any()

    def test_brightness_half_halves_mean(self, fixture_theme: FixtureTheme) -> None:
        renderer = _renderer(fixture_theme)
        full = renderer.render(EyeState(brightness=1.0))
        half = renderer.render(EyeState(brightness=0.5))
        ratio = float(half.mean()) / float(full.mean())
        assert 0.44 <= ratio <= 0.52, ratio

    def test_brightness_zero_is_black(self, fixture_theme: FixtureTheme) -> None:
        renderer = _renderer(fixture_theme)
        assert not renderer.render(EyeState(brightness=0.0)).any()

    def test_render_loop_100_frames(self, fixture_theme: FixtureTheme) -> None:
        renderer = _renderer(fixture_theme)
        states = [
            EyeState(
                gaze_x=math.sin(i * 0.37) * 0.9,
                gaze_y=math.cos(i * 0.29) * 0.9,
                pupil=(i % PUPIL_STEPS) / (PUPIL_STEPS - 1),
                openness=MAX_OPENNESS * ((i % 25) / 24.0),
                brightness=0.6 + 0.4 * ((i % 5) / 4.0),
            )
            for i in range(100)
        ]
        start = time.perf_counter()
        for state in states:
            frame = renderer.render(state)
        elapsed = time.perf_counter() - start
        assert frame.shape == (SIZE, SIZE, 3) and frame.dtype == np.uint8
        per_frame_ms = elapsed / len(states) * 1000.0
        print(f"\n[render perf] {per_frame_ms:.3f} ms/frame over {len(states)} frames "
              f"(budget: 8 ms/eye on Pi 3)")
