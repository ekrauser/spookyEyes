#!/usr/bin/env python3
"""Procedural theme-art generator for spookyeyes.

Generates all art assets for the three built-in themes (human, demon, ghost)
using only numpy + PIL — no network, no external assets. Deterministic: every
asset is derived from a fixed master seed, so re-running produces identical
PNGs bit-for-bit (given the same numpy/PIL versions).

Per theme, writes into <out>/<name>/:
  theme.json     exact schema per DESIGN.md
  sclera.png     RGB  320x320 (renderer crops a gaze-shifted 240x240 window)
  iris.png       RGBA square, side = ceil(iris_frac * 240)  (pupil NOT baked;
                 glow/halo effects are baked into the RGBA alpha falloff)
  highlight.png  RGBA small specular/glow sprite

Also writes a composed visual check per theme to out/art_check_<name>.png
(sclera center-crop + iris + mock pupil + highlight + round vignette) so the
result can be inspected as it will roughly appear on the panel.

Usage:
  python tools/gen_art.py [--theme NAME] [--out themes] [--check-dir out]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

MASTER_SEED = 0xE1E5
PANEL = 240          # panel resolution (model.SIZE)
SCLERA_SIZE = 320    # sclera source canvas
SS = 2               # supersample factor for anti-aliased art

THEME_NAMES = ("human", "demon", "ghost")

# ---------------------------------------------------------------------------
# math / noise helpers
# ---------------------------------------------------------------------------


def smoothstep(e0: float, e1: float, x: np.ndarray | float) -> np.ndarray | float:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def radial_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (r, theta): r normalized to 1.0 at the inscribed-circle edge,
    theta in [-pi, pi], both shaped (size, size)."""
    c = (size - 1) / 2.0
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    dx = x - c
    dy = y - c
    r = np.sqrt(dx * dx + dy * dy) / (size / 2.0)
    theta = np.arctan2(dy, dx)
    return r, theta


def value_noise(rng: np.random.Generator, size: int, cells: int) -> np.ndarray:
    """Smooth 2D value noise in [0,1] via bicubic upscale of a coarse grid."""
    g = rng.standard_normal((cells + 2, cells + 2)).astype(np.float32)
    im = Image.fromarray(g, mode="F").resize((size, size), Image.BICUBIC)
    a = np.asarray(im, dtype=np.float64)
    a -= a.min()
    peak = a.max()
    if peak > 0:
        a /= peak
    return a


def fbm(rng: np.random.Generator, size: int, cells: int, octaves: int = 4) -> np.ndarray:
    """Fractal (multi-octave) value noise in [0,1]."""
    total = np.zeros((size, size), dtype=np.float64)
    amp = 1.0
    amp_sum = 0.0
    for o in range(octaves):
        total += amp * value_noise(rng, size, cells * (2 ** o))
        amp_sum += amp
        amp *= 0.5
    return total / amp_sum


def angular_noise(rng: np.random.Generator, n: int, cells: int) -> np.ndarray:
    """Periodic 1D noise over n angle samples, zero-mean, ~unit std."""
    vals = rng.standard_normal(cells).astype(np.float32)
    tiled = np.tile(vals, 3)[None, :]
    im = Image.fromarray(tiled, mode="F").resize((3 * n, 1), Image.BICUBIC)
    a = np.asarray(im, dtype=np.float64)[0, n : 2 * n].copy()
    a -= a.mean()
    std = a.std()
    return a / std if std > 1e-9 else a


def fiber_field(
    rng: np.random.Generator,
    theta: np.ndarray,
    r: np.ndarray,
    cells: int,
    octaves: int = 3,
    swirl: float = 0.0,
) -> np.ndarray:
    """Radial fiber streak field: per-angle noise (optionally sheared by radius
    for a flame-lick swirl), zero-mean ~unit amplitude."""
    n = 1440
    total = np.zeros_like(theta)
    amp = 1.0
    amp_sum = 0.0
    for o in range(octaves):
        samples = angular_noise(rng, n, cells * (2 ** o))
        ang = theta + swirl * r
        idx = np.mod((ang + math.pi) / (2 * math.pi) * n, n).astype(np.int64)
        total += amp * samples[idx]
        amp_sum += amp
        amp *= 0.55
    return total / amp_sum


def draw_veins(
    rng: np.random.Generator,
    size: int,
    count: int,
    color: tuple[int, int, int],
    alpha: int,
    width0: float,
    width_min: float,
    jitter: float,
    steps: int,
    branch_p: float,
    start_r: tuple[float, float],
    blur: float,
) -> Image.Image:
    """Random-walk vein polylines with decreasing width, drawn on a transparent
    RGBA overlay. Veins start near the rim and wander inward."""
    overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    c = size / 2.0
    step_len = size * 0.014
    # stack entries: (x, y, heading, width, remaining_steps)
    stack: list[tuple[float, float, float, float, int]] = []
    for _ in range(count):
        ang0 = rng.uniform(0.0, 2 * math.pi)
        r0 = rng.uniform(*start_r) * c
        x = c + math.cos(ang0) * r0
        y = c + math.sin(ang0) * r0
        heading = ang0 + math.pi + rng.normal(0.0, 0.5)
        stack.append((x, y, heading, width0, steps))
    while stack:
        x, y, heading, w, n = stack.pop()
        shrink = (w - width_min) / max(n, 1)
        for _ in range(n):
            heading += rng.normal(0.0, jitter)
            nx = x + math.cos(heading) * step_len
            ny = y + math.sin(heading) * step_len
            draw.line(
                [(x, y), (nx, ny)],
                fill=(*color, alpha),
                width=max(1, int(round(w))),
                joint="curve",
            )
            x, y = nx, ny
            w = max(width_min, w - shrink)
            n -= 1
            if n > 3 and rng.random() < branch_p:
                side = 1.0 if rng.random() < 0.5 else -1.0
                stack.append(
                    (x, y, heading + side * rng.uniform(0.35, 0.9), w * 0.7, n // 2)
                )
    if blur > 0:
        overlay = overlay.filter(ImageFilter.GaussianBlur(blur))
    return overlay


# ---------------------------------------------------------------------------
# image assembly helpers
# ---------------------------------------------------------------------------


def to_u8(a: np.ndarray) -> np.ndarray:
    return (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def save_rgb(rgb01: np.ndarray, path: Path, final_size: int) -> None:
    img = Image.fromarray(to_u8(rgb01), "RGB")
    if img.size != (final_size, final_size):
        img = img.resize((final_size, final_size), Image.LANCZOS)
    img.save(path)


def save_rgba(rgb01: np.ndarray, a01: np.ndarray, path: Path, final_size: int) -> None:
    """Downscale color and alpha separately (color fields are defined across
    the whole canvas, so no dark-fringe artifacts), then merge and save."""
    rgb = Image.fromarray(to_u8(rgb01), "RGB")
    alpha = Image.fromarray(to_u8(a01), "L")
    if rgb.size != (final_size, final_size):
        rgb = rgb.resize((final_size, final_size), Image.LANCZOS)
        alpha = alpha.resize((final_size, final_size), Image.LANCZOS)
    rgba = rgb.convert("RGBA")
    rgba.putalpha(alpha)
    rgba.save(path)


def composite_overlay(rgb01: np.ndarray, overlay: Image.Image, alpha_mask: np.ndarray | None = None) -> np.ndarray:
    """Alpha-composite an RGBA PIL overlay onto a float RGB array; optionally
    modulate the overlay's alpha by a float mask first."""
    ov = np.asarray(overlay, dtype=np.float64) / 255.0
    a = ov[..., 3]
    if alpha_mask is not None:
        a = a * alpha_mask
    a = a[..., None]
    return rgb01 * (1.0 - a) + ov[..., :3] * a


def ramp(r: np.ndarray, stops: list[tuple[float, tuple[float, float, float]]]) -> np.ndarray:
    """Piecewise-linear radial color ramp. stops = [(radius, (r,g,b)), ...],
    radii ascending."""
    out = np.zeros((*r.shape, 3), dtype=np.float64)
    radii = np.array([s[0] for s in stops])
    cols = np.array([s[1] for s in stops])
    for ch in range(3):
        out[..., ch] = np.interp(r, radii, cols[:, ch])
    return out


# ---------------------------------------------------------------------------
# HUMAN
# ---------------------------------------------------------------------------


def gen_human_sclera(rng: np.random.Generator) -> np.ndarray:
    size = SCLERA_SIZE * SS
    r, theta = radial_grid(size)
    base = np.array([0.93, 0.905, 0.865])
    edge = np.array([0.55, 0.43, 0.40])  # fleshy desaturated shadow at rim
    t = smoothstep(0.38, 0.98, r)[..., None]
    rgb = base * (1 - t) + edge * t
    # ambient occlusion from the upper lid: darken top third
    yy = np.linspace(0.0, 1.0, size)[:, None]
    lid_shadow = 1.0 - 0.16 * smoothstep(0.35, 0.0, yy)
    rgb *= lid_shadow[..., None]
    # subtle wet mottling
    mott = fbm(rng, size, 5, 4)
    rgb *= (1.0 + 0.05 * (mott - 0.5))[..., None]
    # faint warm tint near the horizontal corners (caruncle-ish)
    corner = smoothstep(0.55, 1.0, np.abs(np.cos(theta)) * r)
    rgb[..., 0] += 0.05 * corner
    rgb[..., 2] -= 0.02 * corner
    # red veins: hug the rim and fade out well before the iris zone
    veins = draw_veins(
        rng, size,
        count=10, color=(178, 48, 44), alpha=130,
        width0=3.0 * SS, width_min=0.8 * SS, jitter=0.22, steps=36,
        branch_p=0.06, start_r=(0.88, 1.05), blur=0.7 * SS,
    )
    fade = smoothstep(0.42, 0.85, r)
    rgb = composite_overlay(rgb, veins, fade)
    # a second, finer capillary pass at the very edge
    caps = draw_veins(
        rng, size,
        count=16, color=(200, 85, 80), alpha=80,
        width0=1.5 * SS, width_min=0.6 * SS, jitter=0.34, steps=24,
        branch_p=0.10, start_r=(0.86, 1.04), blur=0.5 * SS,
    )
    rgb = composite_overlay(rgb, caps, smoothstep(0.55, 0.95, r))
    return rgb


def gen_human_iris(rng: np.random.Generator, diam: int) -> tuple[np.ndarray, np.ndarray]:
    size = diam * SS
    r, theta = radial_grid(size)
    rgb = ramp(r, [
        (0.00, (0.86, 0.64, 0.28)),   # golden amber around the pupil
        (0.30, (0.72, 0.48, 0.20)),
        (0.55, (0.50, 0.32, 0.15)),   # amber-brown mid
        (0.80, (0.30, 0.17, 0.09)),   # dark brown outer
        (1.00, (0.16, 0.09, 0.05)),
    ])
    # radial fiber streaks: coarse + fine
    window = smoothstep(0.12, 0.32, r) * (1.0 - smoothstep(0.80, 0.96, r))
    fibers = fiber_field(rng, theta, r, cells=56, octaves=3, swirl=0.08)
    fine = fiber_field(rng, theta, r, cells=220, octaves=2, swirl=0.08)
    gain = 1.0 + (0.28 * fibers + 0.20 * fine) * window
    rgb *= gain[..., None]
    # crypts: low-frequency dark patches
    crypts = fbm(rng, size, 4, 3)
    rgb *= (1.0 - 0.18 * smoothstep(0.55, 1.0, crypts) * window)[..., None]
    # collarette: brighter wobbly ring near the pupil zone
    wob = fiber_field(rng, theta, r, cells=14, octaves=1)
    ring = np.exp(-(((r - (0.34 + 0.03 * wob)) / 0.05) ** 2))
    rgb += 0.10 * ring[..., None] * np.array([1.0, 0.85, 0.5])
    # limbal ring: darken hard toward the rim
    rgb *= (1.0 - 0.72 * smoothstep(0.80, 0.985, r))[..., None]
    # alpha: opaque disc with a soft anti-aliased limbal edge that melts into
    # the sclera slightly
    a = 1.0 - smoothstep(0.94, 1.0, r)
    return rgb, a


def gen_human_highlight(rng: np.random.Generator, size: int = 56) -> tuple[np.ndarray, np.ndarray]:
    s = size * SS
    c = (s - 1) / 2.0
    y, x = np.mgrid[0:s, 0:s].astype(np.float64)
    # primary soft specular, slightly upper-left of sprite center
    d1 = np.sqrt((x - c * 0.82) ** 2 + (y - c * 0.82) ** 2) / (s / 2.0)
    a = 0.88 * np.exp(-((d1 / 0.42) ** 2.2))
    # small hard core inside the primary
    a = np.maximum(a, 0.95 * np.exp(-((d1 / 0.16) ** 2)))
    # secondary tiny glint lower-right (wet look)
    d2 = np.sqrt((x - c * 1.45) ** 2 + (y - c * 1.45) ** 2) / (s / 2.0)
    a = np.maximum(a, 0.42 * np.exp(-((d2 / 0.14) ** 2)))
    rgb = np.ones((s, s, 3), dtype=np.float64) * np.array([1.0, 0.99, 0.96])
    return rgb, np.clip(a, 0.0, 1.0)


def human_theme_json() -> dict:
    return {
        "name": "human",
        "background": [10, 5, 5],
        "layers": {"sclera": "sclera.png", "iris": "iris.png",
                   "highlight": "highlight.png"},
        "pupil": {"shape": "round", "color": [8, 4, 4], "min_frac": 0.22,
                  "max_frac": 0.55, "slit_width_frac": 0.14},
        "iris_frac": 0.62,
        "gaze_range_px": 52,
        "sclera_parallax": 0.35,
        "eyelids": {"color": [24, 12, 10], "style": "curved", "upper_bias": 0.6},
        "motion": {"saccade_interval": [0.8, 3.2],
                   "saccade_duration": [0.09, 0.22],
                   "wander": 0.35, "crazy": 0.05,
                   "blink_interval": [2.5, 7.0], "blink_speed": 1.0,
                   "pupil_speed": 0.5, "drift": 0.0, "flicker": 0.0},
    }


# ---------------------------------------------------------------------------
# DEMON
# ---------------------------------------------------------------------------


def gen_demon_sclera(rng: np.random.Generator) -> np.ndarray:
    size = SCLERA_SIZE * SS
    r, theta = radial_grid(size)
    # charred red base, mottled
    rgb = ramp(r, [
        (0.00, (0.30, 0.045, 0.03)),
        (0.45, (0.22, 0.030, 0.02)),
        (0.80, (0.10, 0.012, 0.01)),
        (1.05, (0.03, 0.004, 0.004)),
    ])
    mott = fbm(rng, size, 6, 4)
    rgb *= (0.66 + 0.68 * mott)[..., None]
    # cracked-ember hot filaments: ridged noise glowing faint orange
    ridge = (1.0 - np.abs(2.0 * fbm(rng, size, 7, 3) - 1.0)) ** 6
    ember = ridge * (1.0 - smoothstep(0.75, 1.0, r)) * smoothstep(0.15, 0.45, r)
    rgb += ember[..., None] * np.array([0.55, 0.16, 0.01]) * 0.8
    # heat glow behind the iris
    glow = np.exp(-((r / 0.42) ** 2))
    rgb += glow[..., None] * np.array([0.28, 0.05, 0.0])
    # dark burnt veins, jagged and branching
    veins = draw_veins(
        rng, size,
        count=16, color=(12, 0, 0), alpha=175,
        width0=4.2 * SS, width_min=1.0 * SS, jitter=0.5, steps=50,
        branch_p=0.17, start_r=(0.82, 1.05), blur=0.6 * SS,
    )
    fade = smoothstep(0.20, 0.60, r)
    rgb = composite_overlay(rgb, veins, fade)
    return rgb


def gen_demon_iris(rng: np.random.Generator, diam: int) -> tuple[np.ndarray, np.ndarray]:
    size = diam * SS
    r, theta = radial_grid(size)
    rgb = ramp(r, [
        (0.00, (1.00, 0.96, 0.62)),   # white-hot core
        (0.18, (1.00, 0.83, 0.25)),
        (0.40, (0.97, 0.52, 0.06)),   # molten orange
        (0.62, (0.80, 0.22, 0.02)),
        (0.74, (0.58, 0.09, 0.01)),   # deep red rim
        (0.84, (0.85, 0.30, 0.03)),   # hot ember halo bleeding onto sclera
        (1.00, (0.95, 0.45, 0.06)),
    ])
    # flame-lick fibers: mostly radial with a slight lick — no pinwheel
    window = smoothstep(0.08, 0.25, r) * (1.0 - smoothstep(0.78, 0.95, r))
    licks = fiber_field(rng, theta, r, cells=64, octaves=3, swirl=0.55)
    fine = fiber_field(rng, theta, r, cells=210, octaves=2, swirl=0.55)
    gain = 1.0 + (0.27 * licks + 0.30 * fine) * window
    rgb *= gain[..., None]
    # roiling heat mottling
    heat = fbm(rng, size, 5, 3)
    rgb *= (0.90 + 0.24 * heat)[..., None]
    # alpha: opaque ember disc, then a wide soft glow ring bleeding onto the
    # sclera (the halo baked into the RGBA edge)
    core_a = 1.0 - smoothstep(0.70, 0.78, r)
    halo_a = 0.62 * np.exp(-(((r - 0.72).clip(min=0.0) / 0.21) ** 1.6))
    a = np.clip(np.maximum(core_a, halo_a), 0.0, 1.0)
    a *= 1.0 - smoothstep(0.96, 1.0, r)  # clean canvas edge
    return rgb, a


def gen_demon_highlight(rng: np.random.Generator, size: int = 40) -> tuple[np.ndarray, np.ndarray]:
    s = size * SS
    c = (s - 1) / 2.0
    y, x = np.mgrid[0:s, 0:s].astype(np.float64)
    d = np.sqrt((x - c) ** 2 + (y - c) ** 2) / (s / 2.0)
    a = 0.80 * np.exp(-((d / 0.38) ** 2))
    a = np.maximum(a, 0.95 * np.exp(-((d / 0.15) ** 2)))
    rgb = np.zeros((s, s, 3), dtype=np.float64)
    rgb[..., 0] = 1.0
    rgb[..., 1] = 0.92 - 0.35 * smoothstep(0.1, 0.6, d)
    rgb[..., 2] = 0.72 - 0.55 * smoothstep(0.05, 0.5, d)
    return rgb, np.clip(a, 0.0, 1.0)


def demon_theme_json() -> dict:
    return {
        "name": "demon",
        "background": [8, 0, 0],
        "layers": {"sclera": "sclera.png", "iris": "iris.png",
                   "highlight": "highlight.png"},
        "pupil": {"shape": "slit", "color": [2, 0, 0], "min_frac": 0.20,
                  "max_frac": 0.75, "slit_width_frac": 0.24},
        "iris_frac": 0.70,
        "gaze_range_px": 55,
        "sclera_parallax": 0.30,
        "eyelids": {"color": [16, 2, 2], "style": "curved", "upper_bias": 0.5},
        "motion": {"saccade_interval": [0.25, 1.1],
                   "saccade_duration": [0.05, 0.12],
                   "wander": 0.5, "crazy": 0.15,
                   "blink_interval": [1.5, 5.0], "blink_speed": 1.7,
                   "pupil_speed": 0.9, "drift": 0.0, "flicker": 0.12},
    }


# ---------------------------------------------------------------------------
# GHOST
# ---------------------------------------------------------------------------


def gen_ghost_sclera(rng: np.random.Generator) -> np.ndarray:
    size = SCLERA_SIZE * SS
    r, theta = radial_grid(size)
    rgb = ramp(r, [
        (0.00, (0.045, 0.065, 0.095)),
        (0.55, (0.030, 0.045, 0.070)),
        (1.05, (0.008, 0.012, 0.022)),
    ])
    # smoky wisps: ridged fbm filaments, cold cyan-gray
    ridge = (1.0 - np.abs(2.0 * fbm(rng, size, 5, 4) - 1.0)) ** 3.5
    body = fbm(rng, size, 3, 3)  # large-scale patchiness so wisps come and go
    wisp = ridge * (0.35 + 0.65 * body)
    wisp *= 0.55 + 0.45 * np.cos(theta * 2 + 4.0 * r)  # gentle swirl bias
    wisp *= 1.0 - smoothstep(0.85, 1.05, r)
    rgb += np.clip(wisp, 0.0, 1.0)[..., None] * np.array([0.10, 0.16, 0.20])
    # cold ambient halo where the orb lives
    rgb += np.exp(-((r / 0.5) ** 2))[..., None] * np.array([0.015, 0.035, 0.05])
    return rgb


def gen_ghost_iris(rng: np.random.Generator, diam: int) -> tuple[np.ndarray, np.ndarray]:
    size = diam * SS
    r, theta = radial_grid(size)
    rgb = ramp(r, [
        (0.00, (0.97, 1.00, 1.00)),   # white-hot spectral core
        (0.30, (0.87, 0.98, 1.00)),
        (0.55, (0.65, 0.90, 0.97)),   # pale cyan
        (0.80, (0.38, 0.68, 0.84)),
        (1.00, (0.20, 0.44, 0.60)),   # cold haze at the fringe
    ])
    # faint internal churn
    churn = fbm(rng, size, 4, 3)
    rgb *= (0.93 + 0.14 * churn)[..., None]
    # very soft wisp variation inside the glow (no spiral arms)
    tendrils = fiber_field(rng, theta, r, cells=18, octaves=2, swirl=0.9)
    rgb *= (1.0 + 0.07 * tendrils * smoothstep(0.3, 0.65, r))[..., None]
    # tiny dark spectral core (pupil shape is "none"; this hint is baked)
    rgb *= (1.0 - 0.5 * np.exp(-((r / 0.075) ** 2)))[..., None]
    # alpha: gaussian-ish orb falloff — bright core + wide soft halo to the
    # canvas edge (glow baked in)
    core = np.exp(-((r / 0.48) ** 2.1))
    halo = 0.62 * np.exp(-((r / 0.72) ** 2.4))
    a = np.clip(core + halo, 0.0, 1.0)
    a *= 1.0 - smoothstep(0.70, 1.0, r) ** 2  # wide taper: no visible alpha cliff
    return rgb, a


def gen_ghost_highlight(rng: np.random.Generator, size: int = 40) -> tuple[np.ndarray, np.ndarray]:
    s = size * SS
    c = (s - 1) / 2.0
    y, x = np.mgrid[0:s, 0:s].astype(np.float64)
    d = np.sqrt((x - c) ** 2 + (y - c) ** 2) / (s / 2.0)
    a = 0.5 * np.exp(-((d / 0.45) ** 2))
    rgb = np.ones((s, s, 3), dtype=np.float64) * np.array([0.9, 1.0, 1.0])
    return rgb, np.clip(a, 0.0, 1.0)


def ghost_theme_json() -> dict:
    return {
        "name": "ghost",
        "background": [2, 3, 6],
        "layers": {"sclera": "sclera.png", "iris": "iris.png",
                   "highlight": "highlight.png"},
        "pupil": {"shape": "none", "color": [0, 0, 0], "min_frac": 0.15,
                  "max_frac": 0.30, "slit_width_frac": 0.10},
        "iris_frac": 0.55,
        "gaze_range_px": 45,
        "sclera_parallax": 0.25,
        "eyelids": {"color": [3, 5, 9], "style": "straight", "upper_bias": 0.5},
        "motion": {"saccade_interval": [3.0, 7.0],
                   "saccade_duration": [0.8, 1.6],
                   "wander": 0.45, "crazy": 0.05,
                   "blink_interval": [15.0, 40.0], "blink_speed": 0.5,
                   "pupil_speed": 0.2, "drift": 0.6, "flicker": 0.25},
    }


# ---------------------------------------------------------------------------
# theme assembly / validation / check renders
# ---------------------------------------------------------------------------

GENERATORS = {
    "human": (gen_human_sclera, gen_human_iris, gen_human_highlight, human_theme_json),
    "demon": (gen_demon_sclera, gen_demon_iris, gen_demon_highlight, demon_theme_json),
    "ghost": (gen_ghost_sclera, gen_ghost_iris, gen_ghost_highlight, ghost_theme_json),
}

SCHEMA_KEYS = {
    "": {"name", "background", "layers", "pupil", "iris_frac", "gaze_range_px",
         "sclera_parallax", "eyelids", "motion"},
    "layers": {"sclera", "iris"},  # highlight optional
    "pupil": {"shape", "color", "min_frac", "max_frac", "slit_width_frac"},
    "eyelids": {"color", "style", "upper_bias"},
    "motion": {"saccade_interval", "saccade_duration", "wander", "crazy",
               "blink_interval", "blink_speed", "pupil_speed", "drift",
               "flicker"},
}


def theme_rng(name: str, asset: str) -> np.random.Generator:
    """Deterministic per-asset rng so each asset regenerates identically."""
    sub = sum(ord(ch) * (i + 1) for i, ch in enumerate(f"{name}/{asset}"))
    return np.random.default_rng(MASTER_SEED * 100003 + sub)


def validate_theme_json(spec: dict) -> None:
    missing = SCHEMA_KEYS[""] - spec.keys()
    if missing:
        raise ValueError(f"theme.json missing top-level keys: {sorted(missing)}")
    for section in ("layers", "pupil", "eyelids", "motion"):
        missing = SCHEMA_KEYS[section] - spec[section].keys()
        if missing:
            raise ValueError(f"theme.json [{section}] missing keys: {sorted(missing)}")


def generate_theme(name: str, out_dir: Path) -> Path:
    gen_sclera, gen_iris, gen_highlight, make_json = GENERATORS[name]
    spec = make_json()
    validate_theme_json(spec)
    tdir = out_dir / name
    tdir.mkdir(parents=True, exist_ok=True)

    iris_diam = math.ceil(spec["iris_frac"] * PANEL)

    print(f"[{name}] sclera {SCLERA_SIZE}x{SCLERA_SIZE} ...")
    save_rgb(gen_sclera(theme_rng(name, "sclera")), tdir / "sclera.png", SCLERA_SIZE)
    print(f"[{name}] iris {iris_diam}x{iris_diam} ...")
    rgb, a = gen_iris(theme_rng(name, "iris"), iris_diam)
    save_rgba(rgb, a, tdir / "iris.png", iris_diam)
    print(f"[{name}] highlight ...")
    rgb, a = gen_highlight(theme_rng(name, "highlight"))
    save_rgba(rgb, a, tdir / "highlight.png", rgb.shape[0] // SS)
    (tdir / "theme.json").write_text(json.dumps(spec, indent=2) + "\n")
    print(f"[{name}] wrote {tdir}/")
    return tdir


def render_check(theme_dir: Path, check_path: Path) -> None:
    """Compose a rough preview of the eye as the renderer would show it:
    sclera center crop + iris + mock pupil at mid dilation + highlight +
    round vignette. For visual QA only — the real renderer owns the pupil."""
    spec = json.loads((theme_dir / "theme.json").read_text())
    sclera = Image.open(theme_dir / "sclera.png").convert("RGB")
    iris = Image.open(theme_dir / "iris.png").convert("RGBA")
    m = (sclera.width - PANEL) // 2
    frame = sclera.crop((m, m, m + PANEL, m + PANEL)).convert("RGBA")

    d = iris.width
    # mock pupil baked onto a copy of the iris sprite (renderer does this live)
    iris_mock = iris.copy()
    pupil = spec["pupil"]
    frac = lerp(pupil["min_frac"], pupil["max_frac"], 0.5)
    if pupil["shape"] != "none":
        ss = 4
        mask = Image.new("L", (d * ss, d * ss), 0)
        pd = ImageDraw.Draw(mask)
        cx = cy = d * ss / 2.0
        if pupil["shape"] == "round":
            pr = frac * d * ss / 2.0
            pd.ellipse([cx - pr, cy - pr, cx + pr, cy + pr], fill=255)
        else:  # slit
            w = spec["pupil"]["slit_width_frac"] * d * frac * ss / 2.0
            h = d * ss / 2.0
            pd.ellipse([cx - w, cy - h, cx + w, cy + h], fill=255)
        mask = mask.resize((d, d), Image.LANCZOS)
        solid = Image.new("RGBA", (d, d), (*pupil["color"], 255))
        iris_mock.paste(solid, (0, 0), mask)
    pos = ((PANEL - d) // 2, (PANEL - d) // 2)
    layer = Image.new("RGBA", (PANEL, PANEL), (0, 0, 0, 0))
    layer.paste(iris_mock, pos)
    frame = Image.alpha_composite(frame, layer)

    hl_path = theme_dir / spec["layers"].get("highlight", "highlight.png")
    if "highlight" in spec["layers"] and hl_path.exists():
        hl = Image.open(hl_path).convert("RGBA")
        off = int(d * 0.16)
        hx = PANEL // 2 - hl.width // 2 - off
        hy = PANEL // 2 - hl.height // 2 - off
        layer = Image.new("RGBA", (PANEL, PANEL), (0, 0, 0, 0))
        layer.paste(hl, (hx, hy))
        frame = Image.alpha_composite(frame, layer)

    # round vignette: background color outside the panel circle
    ss = 4
    vmask = Image.new("L", (PANEL * ss, PANEL * ss), 0)
    ImageDraw.Draw(vmask).ellipse([0, 0, PANEL * ss - 1, PANEL * ss - 1], fill=255)
    vmask = vmask.resize((PANEL, PANEL), Image.LANCZOS)
    bg = Image.new("RGBA", (PANEL, PANEL), (*spec["background"], 255))
    bg.paste(frame, (0, 0), vmask)

    check_path.parent.mkdir(parents=True, exist_ok=True)
    # upscale 2x so detail is easy to judge by eye
    bg.convert("RGB").resize((PANEL * 2, PANEL * 2), Image.NEAREST).save(check_path)
    print(f"[{spec['name']}] check image -> {check_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--theme", choices=THEME_NAMES, default=None,
                        help="generate a single theme (default: all)")
    parser.add_argument("--out", type=Path, default=Path("themes"),
                        help="output themes directory (default: themes/)")
    parser.add_argument("--check-dir", type=Path, default=Path("out"),
                        help="directory for art_check_<name>.png previews")
    args = parser.parse_args(argv)

    names = [args.theme] if args.theme else list(THEME_NAMES)
    for name in names:
        tdir = generate_theme(name, args.out)
        render_check(tdir, args.check_dir / f"art_check_{name}.png")
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
