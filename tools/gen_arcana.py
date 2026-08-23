#!/usr/bin/env python3
"""The Arcana collection — a second theme suite built on a different approach.

Where gen_art.py grows *organic* eyes (fbm noise, fiber streaks, veins), this
module builds *mechanisms and apparitions that turn in the socket*: everything
is polar geometry — log-spirals, gear-tooth square waves, spiral-arm density
functions, exponential phosphor decay — plus baked glow. All six themes use
pupil shape "none" and declare `iris_spin_rpm` / `iris_spin_mirror`, the
renderer's prebaked iris-rotation feature (older engines without it simply
render them static — the keys are ignored by the loader's unknown-key rule).

Themes: hypno, portal, gears, galaxy, sigil, radar.
Deterministic: same per-asset rng scheme as gen_art (theme_rng).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_art as G  # noqa: E402  (shared helpers + constants)

ARCANA_NAMES = ("hypno", "portal", "gears", "galaxy", "sigil", "radar")


# ---------------------------------------------------------------------------
# shared assembly
# ---------------------------------------------------------------------------


def base_json(
    name: str,
    *,
    iris_frac: float,
    gaze_px: int,
    parallax: float,
    lid_color: tuple[int, int, int],
    lid_style: str,
    upper_bias: float,
    motion: dict,
    spin_rpm: float,
    spin_mirror: bool,
    background: tuple[int, int, int] = (0, 0, 0),
) -> dict:
    return {
        "name": name,
        "background": list(background),
        "layers": {"sclera": "sclera.png", "iris": "iris.png"},
        "pupil": {
            "shape": "none",
            "color": [0, 0, 0],
            "min_frac": 0.3,
            "max_frac": 0.3,
            "slit_width_frac": 0.15,
        },
        "iris_frac": iris_frac,
        "gaze_range_px": gaze_px,
        "sclera_parallax": parallax,
        "eyelids": {"color": list(lid_color), "style": lid_style, "upper_bias": upper_bias},
        "motion": motion,
        "iris_spin_rpm": spin_rpm,
        "iris_spin_mirror": spin_mirror,
    }


def dark_sclera(rng: np.random.Generator, tint: tuple[float, float, float],
                haze: float) -> np.ndarray:
    """Near-black sclera with a faint tinted haze so panels never look dead."""
    size = G.SCLERA_SIZE * G.SS
    r, _ = G.radial_grid(size)
    n = G.fbm(rng, size, 5, 3)
    rgb = np.zeros((size, size, 3)) + np.array([0.012, 0.012, 0.015])
    rgb += (haze * n * (1.0 - G.smoothstep(0.55, 1.0, r)))[..., None] * np.array(tint)
    rgb *= (1.0 - 0.5 * G.smoothstep(0.75, 1.05, r))[..., None]
    return rgb


# ---------------------------------------------------------------------------
# HYPNO — counter-rotating monochrome log-spiral
# ---------------------------------------------------------------------------


def gen_hypno_sclera(rng: np.random.Generator) -> np.ndarray:
    return dark_sclera(rng, (0.5, 0.5, 0.55), 0.02)


def gen_hypno_iris(rng: np.random.Generator, diam: int) -> tuple[np.ndarray, np.ndarray]:
    size = diam * G.SS
    r, th = G.radial_grid(size)
    phase = 3.0 * th + 9.5 * np.log(np.maximum(r, 1e-3))
    band = G.smoothstep(-0.22, 0.22, np.cos(phase))  # 1 = white band, 0 = black
    lum = 0.04 + 0.92 * band
    # calm the singularity: fade to mid-gray disc at the very center
    mix = G.smoothstep(0.03, 0.10, r)
    lum = 0.5 * (1.0 - mix) + lum * mix
    lum *= 1.0 - 0.65 * G.smoothstep(0.88, 1.0, r)  # dark rim
    rgb = np.repeat(lum[..., None], 3, axis=2)
    alpha = 1.0 - G.smoothstep(0.955, 0.995, r)
    return rgb, alpha


def hypno_json() -> dict:
    return base_json(
        "hypno", iris_frac=0.95, gaze_px=10, parallax=0.0,
        lid_color=(10, 10, 10), lid_style="straight", upper_bias=0.5,
        motion={"saccade_interval": [2.0, 6.0], "saccade_duration": [0.4, 0.8],
                "wander": 0.15, "crazy": 0.0, "blink_interval": [8.0, 18.0],
                "blink_speed": 0.8, "pupil_speed": 0.0, "drift": 0.05,
                "flicker": 0.0},
        spin_rpm=10.0, spin_mirror=True,
    )


# ---------------------------------------------------------------------------
# PORTAL — slow violet vortex into nothing
# ---------------------------------------------------------------------------


def gen_portal_sclera(rng: np.random.Generator) -> np.ndarray:
    return dark_sclera(rng, (0.45, 0.1, 0.7), 0.05)


def gen_portal_iris(rng: np.random.Generator, diam: int) -> tuple[np.ndarray, np.ndarray]:
    size = diam * G.SS
    r, th = G.radial_grid(size)
    th_w = th + 4.2 * (1.0 - np.clip(r, 0.0, 1.0)) ** 1.2  # swirl tightens inward
    streak = G.fiber_field(rng, th_w, r, cells=26, octaves=4, swirl=2.5)
    turb = G.fbm(rng, size, 6, 5)
    rgb = G.ramp(r, [
        (0.00, (0.010, 0.000, 0.020)),
        (0.28, (0.050, 0.010, 0.100)),
        (0.55, (0.220, 0.050, 0.380)),
        (0.78, (0.450, 0.120, 0.620)),
        (0.95, (0.100, 0.020, 0.180)),
        (1.00, (0.000, 0.000, 0.000)),
    ]) * (0.55 + 0.45 * turb)[..., None]
    filaments = (np.clip(streak, 0.0, None) ** 1.8
                 * G.smoothstep(0.15, 0.45, r) * (1.0 - G.smoothstep(0.85, 1.0, r)))
    rgb += filaments[..., None] * np.array([0.65, 0.35, 0.95]) * 0.8
    rgb *= (0.15 + 0.85 * G.smoothstep(0.06, 0.30, r))[..., None]  # void core
    alpha = 1.0 - G.smoothstep(0.93, 1.0, r)
    return rgb, alpha


def portal_json() -> dict:
    return base_json(
        "portal", iris_frac=0.92, gaze_px=12, parallax=0.1,
        lid_color=(12, 6, 18), lid_style="curved", upper_bias=0.6,
        motion={"saccade_interval": [3.0, 8.0], "saccade_duration": [0.6, 1.2],
                "wander": 0.2, "crazy": 0.1, "blink_interval": [10.0, 25.0],
                "blink_speed": 0.6, "pupil_speed": 0.0, "drift": 0.1,
                "flicker": 0.06},
        spin_rpm=3.0, spin_mirror=True,
    )


# ---------------------------------------------------------------------------
# GEARS — brass clockwork; counter-rotation reads as meshed gears
# ---------------------------------------------------------------------------


def gen_gears_sclera(rng: np.random.Generator) -> np.ndarray:
    size = G.SCLERA_SIZE * G.SS
    r, th = G.radial_grid(size)
    n = G.fbm(rng, size, 8, 4)
    rgb = np.zeros((size, size, 3)) + np.array([0.035, 0.038, 0.045])
    rgb += (0.03 * n)[..., None] * np.array([0.9, 0.9, 1.0])  # brushed iron
    # rivet ring near the visible rim (sclera shows through gear tooth gaps)
    r_vis = r * (G.SCLERA_SIZE / G.PANEL)
    for k in range(12):
        a = k * math.pi / 6.0 + 0.26
        cx, cy = 0.94 * math.cos(a), 0.94 * math.sin(a)
        d = np.sqrt((r_vis * np.cos(th) - cx) ** 2 + (r_vis * np.sin(th) - cy) ** 2)
        rivet = np.exp(-((d / 0.035) ** 2))
        rgb += rivet[..., None] * np.array([0.10, 0.09, 0.08])
    rgb *= (1.0 - 0.5 * G.smoothstep(0.8, 1.1, r))[..., None]
    return rgb


def gen_gears_iris(rng: np.random.Generator, diam: int) -> tuple[np.ndarray, np.ndarray]:
    size = diam * G.SS
    r, th = G.radial_grid(size)
    tooth_phase = np.mod(th * 12.0 / (2 * math.pi), 1.0)
    teeth = (tooth_phase < 0.5) & (r > 0.86) & (r <= 0.985)
    ring = (r > 0.78) & (r <= 0.86)
    body = (r > 0.30) & (r <= 0.78)
    hub = r <= 0.30
    hole = r <= 0.075
    solid = teeth | ring | body | hub

    n = G.fbm(rng, size, 7, 4)
    patina = G.fbm(rng, size, 5, 3)
    brass = np.array([0.69, 0.52, 0.22]) * (0.75 + 0.35 * n)[..., None]
    brass += ((patina > 0.72).astype(float))[..., None] * np.array([-0.08, 0.04, 0.02])
    # directional light + recessed spoke wedges + engraved ticks
    brass *= (0.86 + 0.22 * np.cos(2.0 * th + 0.7))[..., None]
    spoke_phase = np.mod(th * 5.0 / (2 * math.pi), 1.0)
    recess = (spoke_phase < 0.62) & (r > 0.34) & (r < 0.74)
    brass[recess] *= 0.62
    deg = np.degrees(th) % 30.0
    tick = (np.abs(deg - 15.0) < 0.6) & (r > 0.42) & (r < 0.72)
    brass[tick] *= 0.55
    brass[teeth] *= 1.12
    brass[hub] *= 0.80
    # six hub bolts
    for k in range(6):
        a = k * math.pi / 3.0 + 0.5
        cx, cy = 0.20 * math.cos(a), 0.20 * math.sin(a)
        d = np.sqrt((r * np.cos(th) - cx) ** 2 + (r * np.sin(th) - cy) ** 2)
        brass[d < 0.035] *= 0.55
        brass[d < 0.018] *= 1.9

    rgb = np.where(solid[..., None], brass, 0.0)
    rgb[hole] = 0.02
    alpha = solid.astype(np.float64)
    alpha[hole] = 1.0  # the axle hole is dark metal, not a see-through gap
    return np.clip(rgb, 0.0, 1.0), alpha


def gears_json() -> dict:
    return base_json(
        "gears", iris_frac=0.88, gaze_px=14, parallax=0.15,
        lid_color=(20, 18, 16), lid_style="straight", upper_bias=0.5,
        motion={"saccade_interval": [2.0, 7.0], "saccade_duration": [0.3, 0.7],
                "wander": 0.2, "crazy": 0.0, "blink_interval": [7.0, 15.0],
                "blink_speed": 0.9, "pupil_speed": 0.0, "drift": 0.0,
                "flicker": 0.0},
        spin_rpm=2.0, spin_mirror=True,
    )


# ---------------------------------------------------------------------------
# GALAXY — spinning two-arm nebula over a static starfield
# ---------------------------------------------------------------------------


def _star_layer(rng: np.random.Generator, size: int, count: int,
                bright: int, max_r: float) -> np.ndarray:
    """Additive starfield luminance layer with a few cross-flared bright stars."""
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    c = size / 2.0
    pts = []
    for _ in range(count):
        a = rng.uniform(0, 2 * math.pi)
        rr = math.sqrt(rng.uniform(0, 1)) * max_r * c
        pts.append((c + rr * math.cos(a), c + rr * math.sin(a), rng.uniform(0.25, 1.0)))
    for x, y, v in pts:
        rad = G.SS * (0.6 + 1.1 * v)
        draw.ellipse([x - rad, y - rad, x + rad, y + rad], fill=int(v * 255))
    for x, y, v in pts[:bright]:
        ln = G.SS * 7 * v
        draw.line([x - ln, y, x + ln, y], fill=int(v * 127), width=1)
        draw.line([x, y - ln, x, y + ln], fill=int(v * 127), width=1)
    img = img.filter(ImageFilter.GaussianBlur(G.SS * 0.7))
    return np.asarray(img, dtype=np.float64) / 255.0


def gen_galaxy_sclera(rng: np.random.Generator) -> np.ndarray:
    size = G.SCLERA_SIZE * G.SS
    rgb = dark_sclera(rng, (0.2, 0.25, 0.7), 0.03)
    stars = _star_layer(rng, size, 70, 0, 0.98) * 0.5
    return rgb + stars[..., None] * np.array([0.85, 0.9, 1.0])


def gen_galaxy_iris(rng: np.random.Generator, diam: int) -> tuple[np.ndarray, np.ndarray]:
    size = diam * G.SS
    r, th = G.radial_grid(size)
    phase = 2.0 * th - 3.2 * np.log(np.maximum(r, 0.04))
    arm = G.smoothstep(0.10, 0.90, np.cos(phase)) * np.exp(-r * 2.2)
    turb = G.fbm(rng, size, 6, 4)
    arm *= 0.55 + 0.45 * turb
    hue = G.fbm(rng, size, 4, 3)
    blue = np.array([0.45, 0.60, 1.00])
    pink = np.array([0.90, 0.50, 0.80])
    rgb = arm[..., None] * (blue * (1 - hue)[..., None] + pink * hue[..., None]) * 1.25
    core = np.exp(-((r / 0.16) ** 2))
    rgb += core[..., None] * np.array([1.0, 0.9, 0.7]) * 1.1
    dust = G.fbm(rng, size, 9, 4)
    rgb *= (1.0 - 0.45 * dust * G.smoothstep(0.2, 0.7, arm))[..., None]
    stars = _star_layer(rng, size, 150, 6, 0.92)
    rgb += stars[..., None] * np.array([0.9, 0.95, 1.0]) * 0.9
    lum = rgb.max(axis=2)
    alpha = np.clip(lum * 2.2, 0.0, 1.0) * (1.0 - G.smoothstep(0.9, 1.0, r))
    return np.clip(rgb, 0.0, 1.4), alpha


def galaxy_json() -> dict:
    return base_json(
        "galaxy", iris_frac=0.90, gaze_px=12, parallax=0.25,
        lid_color=(2, 2, 8), lid_style="curved", upper_bias=0.55,
        motion={"saccade_interval": [4.0, 9.0], "saccade_duration": [0.8, 1.5],
                "wander": 0.15, "crazy": 0.0, "blink_interval": [15.0, 30.0],
                "blink_speed": 0.5, "pupil_speed": 0.0, "drift": 0.08,
                "flicker": 0.03},
        spin_rpm=1.2, spin_mirror=False,
    )


# ---------------------------------------------------------------------------
# SIGIL — smoldering rune-ring summoning circle
# ---------------------------------------------------------------------------


def gen_sigil_sclera(rng: np.random.Generator) -> np.ndarray:
    size = G.SCLERA_SIZE * G.SS
    r, _ = G.radial_grid(size)
    rgb = dark_sclera(rng, (0.8, 0.5, 0.2), 0.04)
    # warm light spill from the circle onto the "stone"
    rgb += (0.05 * (1.0 - G.smoothstep(0.2, 0.75, r)))[..., None] * np.array([1.0, 0.55, 0.2])
    return rgb


def _rune_strokes(rng: np.random.Generator) -> list[tuple[float, float, float, float]]:
    """A few random strokes in a [-1,1] local glyph box — angular, rune-like."""
    anchors = [(-0.8, -1.0), (0.8, -1.0), (-0.8, 0.0), (0.8, 0.0),
               (-0.8, 1.0), (0.8, 1.0), (0.0, -0.5), (0.0, 0.5)]
    strokes = []
    for _ in range(rng.integers(3, 6)):
        i, j = rng.choice(len(anchors), size=2, replace=False)
        strokes.append((*anchors[i], *anchors[j]))
    return strokes


def gen_sigil_iris(rng: np.random.Generator, diam: int) -> tuple[np.ndarray, np.ndarray]:
    size = diam * G.SS
    r, th = G.radial_grid(size)
    c = (size - 1) / 2.0
    scale = size / 2.0

    sharp = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(sharp)

    def line(p, q, width, val):
        draw.line([(c + p[0] * scale, c + p[1] * scale),
                   (c + q[0] * scale, c + q[1] * scale)],
                  fill=int(val * 255), width=width)

    # hexagram between the ring's inner edge
    pts = [(0.585 * math.cos(k * math.pi / 3 - math.pi / 2),
            0.585 * math.sin(k * math.pi / 3 - math.pi / 2)) for k in range(6)]
    for tri in ((0, 2, 4), (1, 3, 5)):
        for a, b in zip(tri, tri[1:] + tri[:1]):
            line(pts[a], pts[b], max(1, G.SS), 0.55)

    # 14 runes around the ring at radius 0.70
    for k in range(14):
        ang = k * 2 * math.pi / 14.0
        gx, gy = 0.70 * math.cos(ang), 0.70 * math.sin(ang)
        rot = ang + math.pi / 2.0
        ca, sa = math.cos(rot), math.sin(rot)
        gs = 0.055
        for x0, y0, x1, y1 in _rune_strokes(rng):
            p = (gx + gs * (x0 * ca - y0 * sa), gy + gs * (x0 * sa + y0 * ca))
            q = (gx + gs * (x1 * ca - y1 * sa), gy + gs * (x1 * sa + y1 * ca))
            line(p, q, max(1, int(G.SS * 1.5)), 1.0)

    sharp_a = np.asarray(sharp, dtype=np.float64) / 255.0
    glow_a = np.asarray(sharp.filter(ImageFilter.GaussianBlur(G.SS * 3.0)),
                        dtype=np.float64) / 255.0 * 1.6

    band = G.smoothstep(0.60, 0.63, r) * (1.0 - G.smoothstep(0.77, 0.80, r))
    ember = G.fbm(rng, size, 6, 3)
    haze = G.fbm(rng, size, 4, 3) * (1.0 - G.smoothstep(0.55, 0.9, r)) * 0.10
    core = np.exp(-((r / 0.10) ** 2)) * 0.8

    lum = np.clip(band * (0.45 + 0.3 * ember) + glow_a + core + haze, 0.0, 1.6)
    rgb = (np.clip(lum, 0, 1) ** 1.1)[..., None] * np.array([1.0, 0.52, 0.12])
    rgb += np.clip(sharp_a, 0, 1)[..., None] * np.array([1.0, 0.85, 0.45])
    alpha = np.clip(band * 0.85 + glow_a * 0.9 + np.clip(sharp_a, 0, 1)
                    + core + haze * 2.0, 0.0, 1.0)
    alpha *= 1.0 - G.smoothstep(0.93, 1.0, r)
    return np.clip(rgb, 0.0, 1.2), alpha


def sigil_json() -> dict:
    return base_json(
        "sigil", iris_frac=0.90, gaze_px=15, parallax=0.1,
        lid_color=(14, 8, 4), lid_style="curved", upper_bias=0.6,
        motion={"saccade_interval": [2.5, 7.0], "saccade_duration": [0.4, 0.9],
                "wander": 0.25, "crazy": 0.15, "blink_interval": [9.0, 20.0],
                "blink_speed": 0.7, "pupil_speed": 0.0, "drift": 0.06,
                "flicker": 0.12},
        spin_rpm=1.5, spin_mirror=True,
    )


# ---------------------------------------------------------------------------
# RADAR — phosphor sweep + decay trail rotating over a static scope
# ---------------------------------------------------------------------------


def gen_radar_sclera(rng: np.random.Generator) -> np.ndarray:
    size = G.SCLERA_SIZE * G.SS
    r, th = G.radial_grid(size)
    r_vis = r * (G.SCLERA_SIZE / G.PANEL)  # 1.0 at the visible panel edge
    n = G.fbm(rng, size, 9, 3)
    rgb = np.zeros((size, size, 3)) + np.array([0.008, 0.030, 0.012])
    rgb += (0.02 * n)[..., None] * np.array([0.4, 1.0, 0.5])
    green = np.array([0.30, 1.0, 0.42])
    for v in (0.33, 0.60, 0.87):
        ring = np.exp(-(((r_vis - v) / 0.007) ** 2))
        rgb += ring[..., None] * green * 0.10
    x = r_vis * np.cos(th)
    y = r_vis * np.sin(th)
    cross = (np.exp(-((x / 0.006) ** 2)) + np.exp(-((y / 0.006) ** 2)))
    rgb += (cross * (r_vis < 0.9))[..., None] * green * 0.05
    for _ in range(9):  # contacts on the scope
        a = rng.uniform(0, 2 * math.pi)
        rr = math.sqrt(rng.uniform(0.05, 0.72))
        d = np.sqrt((x - rr * math.cos(a)) ** 2 + (y - rr * math.sin(a)) ** 2)
        rgb += (np.exp(-((d / 0.016) ** 2)) * rng.uniform(0.3, 0.8))[..., None] * \
            np.array([0.55, 1.0, 0.6])
    rgb *= (1.0 - 0.6 * G.smoothstep(0.95, 1.25, r_vis))[..., None]
    return np.clip(rgb, 0.0, 1.0)


def gen_radar_iris(rng: np.random.Generator, diam: int) -> tuple[np.ndarray, np.ndarray]:
    size = diam * G.SS
    r, th = G.radial_grid(size)
    ang = np.mod(-th, 2 * math.pi)  # 0 at the leading edge, growing behind it
    trail = np.exp(-ang / (math.pi * 0.66))
    edge = np.exp(-((np.minimum(ang, 2 * math.pi - ang) / 0.03) ** 2)) * 1.2
    lum = np.clip(trail * 0.7 + edge, 0.0, 1.4)
    lum *= (r < 0.96) * (0.25 + 0.75 * G.smoothstep(0.02, 0.15, r))
    rgb = lum[..., None] * np.array([0.35, 1.0, 0.45]) * 0.95
    alpha = np.clip(lum, 0.0, 1.0) * 0.92
    return np.clip(rgb, 0.0, 1.2), alpha


def radar_json() -> dict:
    return base_json(
        "radar", iris_frac=0.94, gaze_px=8, parallax=0.0,
        lid_color=(4, 12, 5), lid_style="straight", upper_bias=0.5,
        motion={"saccade_interval": [3.0, 8.0], "saccade_duration": [0.5, 1.0],
                "wander": 0.1, "crazy": 0.0, "blink_interval": [12.0, 25.0],
                "blink_speed": 1.5, "pupil_speed": 0.0, "drift": 0.0,
                "flicker": 0.05},
        spin_rpm=6.0, spin_mirror=False,
    )


# ---------------------------------------------------------------------------
# registry + CLI (mirrors gen_art's generate_theme, sans highlight layers)
# ---------------------------------------------------------------------------

ARCANA_GENERATORS = {
    "hypno": (gen_hypno_sclera, gen_hypno_iris, hypno_json),
    "portal": (gen_portal_sclera, gen_portal_iris, portal_json),
    "gears": (gen_gears_sclera, gen_gears_iris, gears_json),
    "galaxy": (gen_galaxy_sclera, gen_galaxy_iris, galaxy_json),
    "sigil": (gen_sigil_sclera, gen_sigil_iris, sigil_json),
    "radar": (gen_radar_sclera, gen_radar_iris, radar_json),
}


def generate_theme(name: str, out_dir: Path) -> Path:
    gen_sclera, gen_iris, make_json = ARCANA_GENERATORS[name]
    spec = make_json()
    G.validate_theme_json(spec)
    tdir = out_dir / name
    tdir.mkdir(parents=True, exist_ok=True)
    iris_diam = math.ceil(spec["iris_frac"] * G.PANEL)
    print(f"[{name}] sclera ...")
    G.save_rgb(gen_sclera(G.theme_rng(name, "sclera")), tdir / "sclera.png", G.SCLERA_SIZE)
    print(f"[{name}] iris {iris_diam}px ...")
    rgb, a = gen_iris(G.theme_rng(name, "iris"), iris_diam)
    G.save_rgba(rgb, a, tdir / "iris.png", iris_diam)
    (tdir / "theme.json").write_text(json.dumps(spec, indent=2) + "\n")
    print(f"[{name}] wrote {tdir}/")
    return tdir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the Arcana theme collection")
    parser.add_argument("--theme", choices=ARCANA_NAMES, default=None,
                        help="single theme (default: all)")
    parser.add_argument("--out", type=Path, default=Path("themes"))
    args = parser.parse_args(argv)
    for name in ([args.theme] if args.theme else ARCANA_NAMES):
        generate_theme(name, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
