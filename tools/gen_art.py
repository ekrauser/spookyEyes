#!/usr/bin/env python3
"""Procedural theme-art generator for spookyeyes.

Generates all art assets for the built-in themes (see THEME_NAMES)
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

THEME_NAMES = ("human", "demon", "ghost", "doomguy", "vampire", "zombie",
               "cyber", "cat", "wisp", "werewolf", "possessed", "serpent",
               "hal")

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


def voronoi_cells(
    rng: np.random.Generator, size: int, count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Voronoi cell field: returns (edge, cell_val), both (size, size).
    `edge` is the F2-F1 nearest-site distance difference normalized by the
    mean site spacing (0 at cell borders, larger inside cells); `cell_val`
    is a per-cell random value in [0, 1] (flat within each cell)."""
    pts = rng.uniform(0.0, size, (count, 2)).astype(np.float32)
    vals = rng.uniform(0.0, 1.0, count).astype(np.float32)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    f1 = np.full((size, size), np.inf, dtype=np.float32)
    f2 = np.full((size, size), np.inf, dtype=np.float32)
    cell = np.zeros((size, size), dtype=np.float32)
    for i in range(count):
        d = (xx - pts[i, 0]) ** 2 + (yy - pts[i, 1]) ** 2
        closer = d < f1
        f2 = np.where(closer, f1, np.minimum(f2, d))
        cell = np.where(closer, vals[i], cell)
        f1 = np.where(closer, d, f1)
    edge = np.sqrt(f2) - np.sqrt(f1)
    spacing = size / math.sqrt(count)
    return (edge / spacing).astype(np.float64), cell.astype(np.float64)


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
# DOOMGUY
# ---------------------------------------------------------------------------


def gen_doomguy_sclera(rng: np.random.Generator) -> np.ndarray:
    size = SCLERA_SIZE * SS
    r, theta = radial_grid(size)
    # battle-worn, slightly yellowed sclera
    base = np.array([0.88, 0.83, 0.62])
    edge = np.array([0.46, 0.34, 0.24])
    t = smoothstep(0.35, 0.96, r)[..., None]
    rgb = base * (1 - t) + edge * t
    # sweat/grime mottling
    mott = fbm(rng, size, 5, 4)
    rgb *= (1.0 + 0.09 * (mott - 0.5))[..., None]
    # burst veins: heavy, densest toward the horizontal corners
    corner = smoothstep(0.30, 0.90, np.abs(np.cos(theta)) * r)
    veins = draw_veins(
        rng, size,
        count=26, color=(172, 20, 14), alpha=190,
        width0=3.6 * SS, width_min=0.8 * SS, jitter=0.30, steps=46,
        branch_p=0.15, start_r=(0.84, 1.05), blur=0.45 * SS,
    )
    rgb = composite_overlay(rgb, veins, (0.30 + 0.70 * corner) * smoothstep(0.28, 0.55, r))
    caps = draw_veins(
        rng, size,
        count=34, color=(205, 55, 40), alpha=125,
        width0=1.7 * SS, width_min=0.6 * SS, jitter=0.38, steps=30,
        branch_p=0.12, start_r=(0.80, 1.04), blur=0.4 * SS,
    )
    rgb = composite_overlay(rgb, caps, (0.35 + 0.65 * corner) * smoothstep(0.40, 0.75, r))
    # blood spatter: droplet clusters, biased toward the corners
    spat = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    sd = ImageDraw.Draw(spat)
    c = size / 2.0
    for _ in range(80):
        ang = (0.0 if rng.random() < 0.5 else math.pi) + rng.normal(0.0, 0.55)
        rr = rng.uniform(0.40, 1.02) * c
        x = c + math.cos(ang) * rr
        y = c + math.sin(ang) * rr
        rad = rng.uniform(1.2, 4.8) * SS
        al = int(rng.uniform(90, 200))
        col = (int(rng.uniform(100, 150)), int(rng.uniform(10, 24)), int(rng.uniform(8, 18)))
        sd.ellipse([x - rad, y - rad * rng.uniform(0.7, 1.0), x + rad, y + rad],
                   fill=(*col, al))
        for _ in range(int(rng.integers(0, 4))):
            sx = x + rng.normal(0.0, rad * 2.6)
            sy = y + rng.normal(0.0, rad * 2.6)
            sr = rad * rng.uniform(0.15, 0.45)
            sd.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=(*col, al))
    spat = spat.filter(ImageFilter.GaussianBlur(0.4 * SS))
    rgb = composite_overlay(rgb, spat, 0.35 + 0.65 * smoothstep(0.30, 0.70, r))
    # furrowed-brow shadow baked across the top: dips lowest mid-eye (scowl)
    xx = np.linspace(0.0, 1.0, size)[None, :]
    yy = np.linspace(0.0, 1.0, size)[:, None]
    wob = angular_noise(rng, size, 9)[None, :] * 0.015
    brow_edge = 0.34 - 0.10 * (np.abs(xx - 0.5) * 2.0) ** 1.4 + wob
    shade = smoothstep(brow_edge + 0.045, brow_edge - 0.16, yy)
    brow_col = np.array([0.13, 0.05, 0.03])
    rgb = rgb * (1.0 - 0.90 * shade[..., None]) + brow_col * (0.90 * shade[..., None])
    # general upper-lid ambient occlusion below the brow
    rgb *= (1.0 - 0.12 * smoothstep(0.55, 0.30, yy))[..., None]
    return rgb


def gen_doomguy_iris(rng: np.random.Generator, diam: int) -> tuple[np.ndarray, np.ndarray]:
    size = diam * SS
    r, theta = radial_grid(size)
    # steel blue, cold and hard
    rgb = ramp(r, [
        (0.00, (0.72, 0.80, 0.90)),
        (0.25, (0.48, 0.60, 0.74)),
        (0.50, (0.30, 0.42, 0.58)),
        (0.75, (0.17, 0.25, 0.40)),
        (1.00, (0.07, 0.11, 0.19)),
    ])
    # tight high-contrast fibers
    window = smoothstep(0.10, 0.30, r) * (1.0 - smoothstep(0.82, 0.96, r))
    fibers = fiber_field(rng, theta, r, cells=80, octaves=3, swirl=0.04)
    fine = fiber_field(rng, theta, r, cells=260, octaves=2, swirl=0.04)
    gain = 1.0 + (0.42 * fibers + 0.30 * fine) * window
    rgb *= np.clip(gain, 0.0, None)[..., None]
    # subtle steel collarette
    wobble = fiber_field(rng, theta, r, cells=12, octaves=1)
    ring = np.exp(-(((r - (0.32 + 0.02 * wobble)) / 0.045) ** 2))
    rgb += 0.08 * ring[..., None] * np.array([0.75, 0.85, 1.0])
    # thin dark limbal ring
    rgb *= (1.0 - 0.88 * smoothstep(0.87, 0.975, r))[..., None]
    a = 1.0 - smoothstep(0.94, 1.0, r)
    return rgb, a


def gen_doomguy_highlight(rng: np.random.Generator, size: int = 28) -> tuple[np.ndarray, np.ndarray]:
    s = size * SS
    c = (s - 1) / 2.0
    y, x = np.mgrid[0:s, 0:s].astype(np.float64)
    d = np.sqrt((x - c) ** 2 + (y - c) ** 2) / (s / 2.0)
    # small, harsh: hard bright core, minimal bloom
    a = 0.95 * np.exp(-((d / 0.30) ** 4))
    a = np.maximum(a, 0.25 * np.exp(-((d / 0.55) ** 2)))
    rgb = np.ones((s, s, 3), dtype=np.float64) * np.array([1.0, 0.98, 0.94])
    return rgb, np.clip(a, 0.0, 1.0)


def doomguy_theme_json() -> dict:
    return {
        "name": "doomguy",
        "background": [10, 4, 3],
        "layers": {"sclera": "sclera.png", "iris": "iris.png",
                   "highlight": "highlight.png"},
        "pupil": {"shape": "round", "color": [8, 7, 8], "min_frac": 0.18,
                  "max_frac": 0.35, "slit_width_frac": 0.14},
        "iris_frac": 0.60,
        "gaze_range_px": 52,
        "sclera_parallax": 0.35,
        "eyelids": {"color": [50, 18, 12], "style": "curved", "upper_bias": 0.75},
        "motion": {"saccade_interval": [0.25, 1.2],
                   "saccade_duration": [0.05, 0.11],
                   "wander": 0.55, "crazy": 0.04,
                   "blink_interval": [3.0, 8.0], "blink_speed": 1.4,
                   "pupil_speed": 0.6, "drift": 0.0, "flicker": 0.0},
    }


# ---------------------------------------------------------------------------
# VAMPIRE
# ---------------------------------------------------------------------------


def gen_vampire_sclera(rng: np.random.Generator) -> np.ndarray:
    size = SCLERA_SIZE * SS
    r, theta = radial_grid(size)
    # near-black with the faintest red warmth
    rgb = ramp(r, [
        (0.00, (0.055, 0.012, 0.018)),
        (0.50, (0.035, 0.008, 0.012)),
        (1.05, (0.010, 0.002, 0.004)),
    ])
    # dried-blood marbling: ridged filaments that come and go in patches
    ridge = (1.0 - np.abs(2.0 * fbm(rng, size, 5, 4) - 1.0)) ** 4
    patch = fbm(rng, size, 3, 3)
    marble = ridge * (0.30 + 0.70 * patch)
    marble *= 1.0 - smoothstep(0.88, 1.05, r)
    rgb += np.clip(marble, 0.0, 1.0)[..., None] * np.array([0.105, 0.014, 0.020])
    # dark crimson veins
    veins = draw_veins(
        rng, size,
        count=14, color=(96, 10, 16), alpha=150,
        width0=3.4 * SS, width_min=0.9 * SS, jitter=0.34, steps=48,
        branch_p=0.12, start_r=(0.80, 1.05), blur=0.8 * SS,
    )
    rgb = composite_overlay(rgb, veins, smoothstep(0.12, 0.50, r))
    # faint arterial warmth behind the iris
    rgb += np.exp(-((r / 0.45) ** 2))[..., None] * np.array([0.045, 0.004, 0.008])
    return rgb


def gen_vampire_iris(rng: np.random.Generator, diam: int) -> tuple[np.ndarray, np.ndarray]:
    size = diam * SS
    r, theta = radial_grid(size)
    # deep arterial crimson
    rgb = ramp(r, [
        (0.00, (0.48, 0.022, 0.058)),
        (0.25, (0.58, 0.042, 0.078)),
        (0.50, (0.50, 0.034, 0.068)),
        (0.70, (0.36, 0.022, 0.050)),
        (0.86, (0.19, 0.013, 0.032)),
        (1.00, (0.075, 0.006, 0.016)),
    ])
    # black radial striations: only the positive fiber lobes darken
    window = smoothstep(0.12, 0.30, r) * (1.0 - smoothstep(0.80, 0.95, r))
    stri = fiber_field(rng, theta, r, cells=64, octaves=3, swirl=0.02)
    rgb *= (1.0 - 0.60 * np.clip(stri, 0.0, 1.3) / 1.3 * window)[..., None]
    fine = fiber_field(rng, theta, r, cells=220, octaves=2, swirl=0.02)
    rgb *= (1.0 + 0.14 * fine * window)[..., None]
    # thin brighter blood-orange collarette (outside the mid-dilation pupil)
    wobble = fiber_field(rng, theta, r, cells=16, octaves=1)
    ring = np.exp(-(((r - (0.47 + 0.015 * wobble)) / 0.026) ** 2))
    rgb += 0.36 * ring[..., None] * np.array([1.0, 0.34, 0.06])
    # alpha: opaque disc + subtle dark glow halo bleeding onto the sclera
    core_a = 1.0 - smoothstep(0.80, 0.875, r)
    halo_a = 0.55 * np.exp(-(((r - 0.82).clip(min=0.0) / 0.13) ** 1.8))
    a = np.maximum(core_a, halo_a)
    a *= 1.0 - smoothstep(0.96, 1.0, r)
    return rgb, np.clip(a, 0.0, 1.0)


def gen_vampire_highlight(rng: np.random.Generator, size: int = 26) -> tuple[np.ndarray, np.ndarray]:
    s = size * SS
    c = (s - 1) / 2.0
    y, x = np.mgrid[0:s, 0:s].astype(np.float64)
    d = np.sqrt((x - c) ** 2 + (y - c) ** 2) / (s / 2.0)
    # small cold pale glint: moonlight
    a = 0.90 * np.exp(-((d / 0.20) ** 2))
    a = np.maximum(a, 0.35 * np.exp(-((d / 0.48) ** 2.2)))
    rgb = np.ones((s, s, 3), dtype=np.float64) * np.array([0.82, 0.88, 1.0])
    return rgb, np.clip(a, 0.0, 1.0)


def vampire_theme_json() -> dict:
    return {
        "name": "vampire",
        "background": [5, 1, 2],
        "layers": {"sclera": "sclera.png", "iris": "iris.png",
                   "highlight": "highlight.png"},
        "pupil": {"shape": "round", "color": [0, 0, 0], "min_frac": 0.15,
                  "max_frac": 0.5, "slit_width_frac": 0.14},
        "iris_frac": 0.64,
        "gaze_range_px": 50,
        "sclera_parallax": 0.30,
        "eyelids": {"color": [12, 3, 6], "style": "curved", "upper_bias": 0.65},
        "motion": {"saccade_interval": [2.0, 6.0],
                   "saccade_duration": [0.25, 0.5],
                   "wander": 0.25, "crazy": 0.0,
                   "blink_interval": [8.0, 20.0], "blink_speed": 0.7,
                   "pupil_speed": 0.3, "drift": 0.05, "flicker": 0.0},
    }


# ---------------------------------------------------------------------------
# ZOMBIE
# ---------------------------------------------------------------------------


def gen_zombie_sclera(rng: np.random.Generator) -> np.ndarray:
    size = SCLERA_SIZE * SS
    r, theta = radial_grid(size)
    # pallid gray-green
    base = np.array([0.66, 0.70, 0.58])
    edge = np.array([0.26, 0.30, 0.22])
    t = smoothstep(0.30, 1.0, r)[..., None]
    rgb = base * (1 - t) + edge * t
    mott = fbm(rng, size, 5, 4)
    rgb *= (0.90 + 0.20 * mott)[..., None]
    # bruise blotches: purple contusions + sickly yellow-green patches
    b1 = fbm(rng, size, 4, 3)
    m1 = smoothstep(0.55, 0.82, b1)
    rgb = rgb * (1.0 - 0.62 * m1[..., None]) + np.array([0.33, 0.25, 0.33]) * (0.62 * m1[..., None])
    b2 = fbm(rng, size, 6, 3)
    m2 = smoothstep(0.60, 0.90, b2)
    rgb = rgb * (1.0 - 0.45 * m2[..., None]) + np.array([0.47, 0.48, 0.21]) * (0.45 * m2[..., None])
    # dark rot veins: jagged, branching, olive-black — hug the rim, sparing
    # the center so the cataract stays the focal point
    veins = draw_veins(
        rng, size,
        count=11, color=(28, 34, 18), alpha=150,
        width0=3.2 * SS, width_min=1.0 * SS, jitter=0.42, steps=32,
        branch_p=0.12, start_r=(0.78, 1.05), blur=0.9 * SS,
    )
    rgb = composite_overlay(rgb, veins, smoothstep(0.35, 0.72, r))
    return rgb


def gen_zombie_iris(rng: np.random.Generator, diam: int) -> tuple[np.ndarray, np.ndarray]:
    size = diam * SS
    r, theta = radial_grid(size)
    # desaturated gray-blue base, low contrast
    rgb = ramp(r, [
        (0.00, (0.50, 0.56, 0.58)),
        (0.40, (0.43, 0.49, 0.52)),
        (0.75, (0.35, 0.41, 0.44)),
        (1.00, (0.27, 0.32, 0.35)),
    ])
    window = smoothstep(0.12, 0.35, r) * (1.0 - smoothstep(0.75, 0.95, r))
    fibers = fiber_field(rng, theta, r, cells=48, octaves=2, swirl=0.1)
    rgb *= (1.0 + 0.10 * fibers * window)[..., None]
    # heavy cloudy cataract veil, densest over the center
    cloud = fbm(rng, size, 3, 4)
    veil = 0.38 + 0.28 * cloud + 0.22 * np.exp(-((r / 0.55) ** 2))
    veil = np.clip(veil, 0.0, 0.88)
    milky = np.array([0.80, 0.82, 0.78])
    rgb = rgb * (1.0 - veil[..., None]) + milky * veil[..., None]
    # smeared edges: no limbal ring, soft blotchy fade into the sclera
    smear = fbm(rng, size, 5, 3)
    a = 1.0 - smoothstep(0.72 + 0.10 * smear, 1.0, r)
    return rgb, np.clip(a, 0.0, 1.0)


def gen_zombie_highlight(rng: np.random.Generator, size: int = 44) -> tuple[np.ndarray, np.ndarray]:
    s = size * SS
    c = (s - 1) / 2.0
    y, x = np.mgrid[0:s, 0:s].astype(np.float64)
    d = np.sqrt((x - c) ** 2 + (y - c) ** 2) / (s / 2.0)
    # faint diffuse sheen — a dead eye barely reflects
    a = 0.32 * np.exp(-((d / 0.60) ** 2))
    rgb = np.ones((s, s, 3), dtype=np.float64) * np.array([0.90, 0.93, 0.88])
    return rgb, np.clip(a, 0.0, 1.0)


def zombie_theme_json() -> dict:
    return {
        "name": "zombie",
        "background": [8, 10, 6],
        "layers": {"sclera": "sclera.png", "iris": "iris.png",
                   "highlight": "highlight.png"},
        "pupil": {"shape": "round", "color": [42, 46, 42], "min_frac": 0.3,
                  "max_frac": 0.45, "slit_width_frac": 0.14},
        "iris_frac": 0.60,
        "gaze_range_px": 50,
        "sclera_parallax": 0.32,
        "eyelids": {"color": [44, 50, 38], "style": "straight", "upper_bias": 0.5},
        "motion": {"saccade_interval": [2.5, 7.0],
                   "saccade_duration": [0.6, 1.2],
                   "wander": 0.45, "crazy": 0.6,
                   "blink_interval": [6.0, 18.0], "blink_speed": 0.35,
                   "pupil_speed": 0.1, "drift": 0.3, "flicker": 0.0},
    }


# ---------------------------------------------------------------------------
# CYBER
# ---------------------------------------------------------------------------


def gen_cyber_sclera(rng: np.random.Generator) -> np.ndarray:
    size = SCLERA_SIZE * SS
    r, theta = radial_grid(size)
    rgb = ramp(r, [
        (0.00, (0.012, 0.022, 0.030)),
        (0.60, (0.008, 0.016, 0.022)),
        (1.05, (0.003, 0.007, 0.010)),
    ])
    overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    # faint hex-cell plating
    hex_r = size * 0.052
    dx = 1.5 * hex_r
    dy = math.sqrt(3.0) * hex_r
    for i in range(-1, int(size / dx) + 2):
        for j in range(-1, int(size / dy) + 2):
            cx = i * dx
            cy = j * dy + (dy / 2.0 if i % 2 else 0.0)
            pts = [(cx + hex_r * math.cos(k * math.pi / 3.0),
                    cy + hex_r * math.sin(k * math.pi / 3.0)) for k in range(6)]
            od.line(pts + [pts[0]], fill=(0, 130, 150, 26), width=SS)
    # circuit traces: orthogonal/45-degree runs with solder-pad dots
    for _ in range(24):
        x = rng.uniform(0.0, size)
        y = rng.uniform(0.0, size)
        heading = float(rng.integers(0, 4)) * (math.pi / 2.0)
        pts2 = [(x, y)]
        for _ in range(int(rng.integers(3, 7))):
            length = rng.uniform(0.035, 0.11) * size
            x += math.cos(heading) * length
            y += math.sin(heading) * length
            pts2.append((x, y))
            heading += (math.pi / 4.0) * float(rng.integers(1, 3)) * (1.0 if rng.random() < 0.5 else -1.0)
        od.line(pts2, fill=(0, 120, 140, 44), width=SS)
        for px, py in (pts2[0], pts2[-1]):
            pr = 2.2 * SS
            od.ellipse([px - pr, py - pr, px + pr, py + pr], fill=(0, 140, 160, 60))
    rgb = composite_overlay(rgb, overlay)
    # dim cyan rim glow right where the round panel edge sits
    ringp = np.exp(-(((r - 0.73) / 0.08) ** 2))
    rgb += ringp[..., None] * np.array([0.0, 0.155, 0.180])
    return rgb


def gen_cyber_iris(rng: np.random.Generator, diam: int) -> tuple[np.ndarray, np.ndarray]:
    size = diam * SS
    r, theta = radial_grid(size)
    half = size / 2.0
    pxr = 1.0 / half  # r-units per source pixel

    def ring(r0: float, hw_px: float, gain: float) -> np.ndarray:
        return gain * np.clip((hw_px * pxr - np.abs(r - r0)) / (1.2 * pxr) + 0.5, 0.0, 1.0)

    def ticks(step_deg: float, hw_px: float, r_in: float, r_out: float, gain: float) -> np.ndarray:
        step = math.radians(step_deg)
        dm = np.abs((theta / step + 0.5) % 1.0 - 0.5) * step
        arc_px = dm * r * half
        ang = np.clip((hw_px - arc_px) / 1.2 + 0.5, 0.0, 1.0)
        rad = (np.clip((r - r_in) / (1.5 * pxr) + 0.5, 0.0, 1.0)
               * np.clip((r_out - r) / (1.5 * pxr) + 0.5, 0.0, 1.0))
        return gain * ang * rad

    def arc(r0: float, hw_px: float, a0: float, span: float, gain: float) -> np.ndarray:
        d_ang = np.mod(theta - a0, 2.0 * math.pi)
        ang = smoothstep(0.0, 0.06, d_ang) * (1.0 - smoothstep(span - 0.06, span, d_ang))
        rad = np.clip((hw_px * pxr - np.abs(r - r0)) / (1.2 * pxr) + 0.5, 0.0, 1.0)
        return gain * ang * rad

    rgb = ramp(r, [
        (0.00, (0.010, 0.030, 0.040)),
        (0.70, (0.008, 0.026, 0.036)),
        (1.00, (0.004, 0.014, 0.020)),
    ])
    # subtle radial scanlines
    scan = 0.5 + 0.5 * np.cos(theta * 96.0)
    win = smoothstep(0.24, 0.34, r) * (1.0 - smoothstep(0.88, 0.95, r))
    rgb += (0.10 * scan * win)[..., None] * np.array([0.0, 0.55, 0.60])
    # concentric HUD rings at exact radii
    teal = np.zeros_like(r)
    teal += ring(0.22, 1.4, 0.45)   # aperture guide, just outside min pupil
    teal += ring(0.30, 1.6, 0.55)
    teal += ring(0.44, 1.2, 0.40)
    teal += ring(0.58, 1.6, 0.60)
    teal += ring(0.72, 1.2, 0.40)
    teal += ring(0.86, 2.2, 0.75)
    # tick marks: fine every 12 degrees, long cardinal ticks every 60
    teal += ticks(12.0, 1.4, 0.76, 0.83, 0.50)
    teal += ticks(60.0, 2.0, 0.71, 0.86, 0.85)
    # brighter arc segments (HUD gauges)
    white = np.zeros_like(r)
    white += arc(0.44, 3.0, rng.uniform(0.0, 2.0 * math.pi), math.radians(rng.uniform(75.0, 105.0)), 0.90)
    white += arc(0.72, 2.6, rng.uniform(0.0, 2.0 * math.pi), math.radians(rng.uniform(50.0, 80.0)), 0.80)
    white += arc(0.30, 2.4, rng.uniform(0.0, 2.0 * math.pi), math.radians(rng.uniform(40.0, 60.0)), 0.70)
    rgb += np.clip(teal, 0.0, 1.2)[..., None] * np.array([0.05, 0.80, 0.88])
    rgb += np.clip(white, 0.0, 1.0)[..., None] * np.array([0.55, 1.00, 1.00])
    # soft luminous haze inside the outer ring
    rgb += (0.06 * np.exp(-(((r - 0.86) / 0.10) ** 2)))[..., None] * np.array([0.0, 0.50, 0.55])
    rgb = np.clip(rgb, 0.0, 1.0)
    a = 1.0 - smoothstep(0.955, 1.0, r)
    return rgb, a


def gen_cyber_highlight(rng: np.random.Generator, size: int = 18) -> tuple[np.ndarray, np.ndarray]:
    s = size * SS
    c = (s - 1) / 2.0
    y, x = np.mgrid[0:s, 0:s].astype(np.float64)
    d = np.sqrt((x - c) ** 2 + (y - c) ** 2) / (s / 2.0)
    # tiny sharp specular dot
    a = 0.95 * np.exp(-((d / 0.30) ** 6))
    rgb = np.ones((s, s, 3), dtype=np.float64) * np.array([0.85, 1.00, 1.00])
    return rgb, np.clip(a, 0.0, 1.0)


def cyber_theme_json() -> dict:
    return {
        "name": "cyber",
        "background": [2, 4, 6],
        "layers": {"sclera": "sclera.png", "iris": "iris.png",
                   "highlight": "highlight.png"},
        "pupil": {"shape": "round", "color": [3, 6, 14], "min_frac": 0.2,
                  "max_frac": 0.6, "slit_width_frac": 0.14},
        "iris_frac": 0.68,
        "gaze_range_px": 50,
        "sclera_parallax": 0.30,
        "eyelids": {"color": [5, 8, 14], "style": "straight", "upper_bias": 0.5},
        "motion": {"saccade_interval": [0.8, 2.5],
                   "saccade_duration": [0.04, 0.06],
                   "wander": 0.4, "crazy": 0.0,
                   "blink_interval": [5.0, 12.0], "blink_speed": 2.2,
                   "pupil_speed": 0.8, "drift": 0.0, "flicker": 0.10},
    }


# ---------------------------------------------------------------------------
# CAT
# ---------------------------------------------------------------------------


def gen_cat_sclera(rng: np.random.Generator) -> np.ndarray:
    size = SCLERA_SIZE * SS
    r, theta = radial_grid(size)
    # dark warm fur-shadow vignette (the huge iris hides most of it)
    rgb = ramp(r, [
        (0.00, (0.155, 0.105, 0.070)),
        (0.45, (0.100, 0.068, 0.045)),
        (0.80, (0.045, 0.030, 0.020)),
        (1.05, (0.012, 0.008, 0.006)),
    ])
    # radial fur strokes around the socket
    fur = fiber_field(rng, theta, r, cells=90, octaves=2, swirl=0.15)
    rgb *= (1.0 + 0.30 * fur * smoothstep(0.35, 0.75, r))[..., None]
    mott = fbm(rng, size, 5, 3)
    rgb *= (0.92 + 0.16 * mott)[..., None]
    # warm glow behind the iris
    rgb += np.exp(-((r / 0.5) ** 2))[..., None] * np.array([0.045, 0.028, 0.008])
    return rgb


def gen_cat_iris(rng: np.random.Generator, diam: int) -> tuple[np.ndarray, np.ndarray]:
    size = diam * SS
    r, theta = radial_grid(size)
    # luminous gold core melting to green rim
    rgb = ramp(r, [
        (0.00, (1.00, 0.80, 0.26)),
        (0.22, (0.98, 0.74, 0.20)),
        (0.45, (0.82, 0.70, 0.21)),
        (0.62, (0.54, 0.62, 0.20)),
        (0.78, (0.30, 0.48, 0.20)),
        (0.92, (0.13, 0.30, 0.14)),
        (1.00, (0.05, 0.13, 0.07)),
    ])
    # fine radial fibers
    window = smoothstep(0.06, 0.20, r) * (1.0 - smoothstep(0.84, 0.96, r))
    coarse = fiber_field(rng, theta, r, cells=64, octaves=3, swirl=0.10)
    fine = fiber_field(rng, theta, r, cells=230, octaves=2, swirl=0.10)
    rgb *= (1.0 + (0.22 * coarse + 0.24 * fine) * window)[..., None]
    # crypt mottling, subtle
    crypts = fbm(rng, size, 4, 3)
    rgb *= (1.0 - 0.10 * smoothstep(0.55, 1.0, crypts) * window)[..., None]
    # luminous core: the night-shine glow around the slit
    rgb += (0.12 * np.exp(-((r / 0.35) ** 2)))[..., None] * np.array([1.0, 0.78, 0.30])
    # strong dark limbal ring
    rgb *= (1.0 - 0.88 * smoothstep(0.84, 0.965, r))[..., None]
    a = 1.0 - smoothstep(0.955, 1.0, r)
    return rgb, np.clip(a, 0.0, 1.0)


def gen_cat_highlight(rng: np.random.Generator, size: int = 52) -> tuple[np.ndarray, np.ndarray]:
    s = size * SS
    c = (s - 1) / 2.0
    y, x = np.mgrid[0:s, 0:s].astype(np.float64)
    # two glints: one big soft-edged, one small sharp, lower-right
    d1 = np.sqrt((x - c * 0.72) ** 2 + (y - c * 0.72) ** 2) / (s / 2.0)
    a = 0.92 * np.exp(-((d1 / 0.34) ** 3))
    d2 = np.sqrt((x - c * 1.52) ** 2 + (y - c * 1.48) ** 2) / (s / 2.0)
    a = np.maximum(a, 0.75 * np.exp(-((d2 / 0.13) ** 2.5)))
    rgb = np.ones((s, s, 3), dtype=np.float64) * np.array([1.0, 0.98, 0.92])
    return rgb, np.clip(a, 0.0, 1.0)


def cat_theme_json() -> dict:
    return {
        "name": "cat",
        "background": [4, 3, 2],
        "layers": {"sclera": "sclera.png", "iris": "iris.png",
                   "highlight": "highlight.png"},
        "pupil": {"shape": "slit", "color": [4, 5, 4], "min_frac": 0.06,
                  "max_frac": 0.85, "slit_width_frac": 0.22},
        "iris_frac": 0.88,
        "gaze_range_px": 30,
        "sclera_parallax": 0.2,
        "eyelids": {"color": [9, 7, 6], "style": "curved", "upper_bias": 0.6},
        "motion": {"saccade_interval": [1.2, 6.0],
                   "saccade_duration": [0.06, 0.12],
                   "wander": 0.4, "crazy": 0.03,
                   "blink_interval": [7.0, 20.0], "blink_speed": 0.5,
                   "pupil_speed": 0.7, "drift": 0.0, "flicker": 0.0},
    }


# ---------------------------------------------------------------------------
# WISP
# ---------------------------------------------------------------------------


def gen_wisp_sclera(rng: np.random.Generator) -> np.ndarray:
    size = SCLERA_SIZE * SS
    r, theta = radial_grid(size)
    # empty skull socket: recessed near-black depth gradient, never flat
    rgb = ramp(r, [
        (0.00, (0.020, 0.016, 0.014)),
        (0.35, (0.030, 0.024, 0.020)),
        (0.62, (0.052, 0.042, 0.034)),
        (0.80, (0.040, 0.032, 0.026)),
        (1.05, (0.012, 0.010, 0.008)),
    ])
    # sooty variation inside the socket
    mott = fbm(rng, size, 4, 4)
    rgb *= (0.82 + 0.36 * mott)[..., None]
    # warm bone rim just inside the panel edge (the renderer's vignette kills
    # anything at r >= 0.75 of this canvas, so sit at ~0.69)
    wob = angular_noise(rng, 1440, 24)
    idx = np.mod((theta + math.pi) / (2 * math.pi) * 1440, 1440).astype(np.int64)
    rim_r = 0.69 + 0.012 * wob[idx]
    rim = np.exp(-(((r - rim_r) / 0.060) ** 2))
    rgb += rim[..., None] * np.array([0.165, 0.125, 0.080])
    # a fainter inner bounce so the rim reads as a rounded bone edge
    rgb += np.exp(-(((r - 0.57) / 0.09) ** 2))[..., None] * np.array([0.035, 0.026, 0.016])
    return rgb


def gen_wisp_iris(rng: np.random.Generator, diam: int) -> tuple[np.ndarray, np.ndarray]:
    size = diam * SS
    r, _theta = radial_grid(size)
    # the wisp: pale amber-white pinprick, wide soft halo baked in the alpha
    rgb = ramp(r, [
        (0.00, (1.00, 0.97, 0.86)),
        (0.12, (1.00, 0.92, 0.70)),
        (0.35, (0.95, 0.78, 0.45)),
        (0.70, (0.80, 0.55, 0.25)),
        (1.00, (0.60, 0.38, 0.15)),
    ])
    core = np.exp(-((r / 0.14) ** 2))
    halo = 0.70 * np.exp(-((r / 0.52) ** 2))
    a = np.clip(core + halo, 0.0, 1.0)
    a *= 1.0 - smoothstep(0.80, 1.0, r) ** 2  # melt into the void, no cliff
    return rgb, a


def wisp_theme_json() -> dict:
    return {
        "name": "wisp",
        "background": [1, 1, 2],
        "layers": {"sclera": "sclera.png", "iris": "iris.png"},
        "pupil": {"shape": "none", "color": [0, 0, 0], "min_frac": 0.2,
                  "max_frac": 0.4, "slit_width_frac": 0.1},
        "iris_frac": 0.28,
        "gaze_range_px": 70,
        "sclera_parallax": 0.15,
        "eyelids": {"color": [0, 0, 0], "style": "straight", "upper_bias": 0.5},
        "motion": {"saccade_interval": [2.0, 6.0],
                   "saccade_duration": [0.8, 1.6],
                   "wander": 0.6, "crazy": 0.3,
                   "blink_interval": [60.0, 120.0], "blink_speed": 1.0,
                   "pupil_speed": 0.0, "drift": 0.7, "flicker": 0.35},
    }


# ---------------------------------------------------------------------------
# WEREWOLF
# ---------------------------------------------------------------------------


def gen_werewolf_sclera(rng: np.random.Generator) -> np.ndarray:
    size = SCLERA_SIZE * SS
    r, theta = radial_grid(size)
    # yellowed amber, grimy dark rim
    base = np.array([0.85, 0.68, 0.34])
    edge = np.array([0.24, 0.15, 0.07])
    t = smoothstep(0.32, 0.95, r)[..., None]
    rgb = base * (1 - t) + edge * t
    mott = fbm(rng, size, 5, 4)
    rgb *= (0.92 + 0.16 * mott)[..., None]
    # HEAVY bloodshot: two dense vein passes flooding in from the corners
    corner = smoothstep(0.25, 0.85, np.abs(np.cos(theta)) * r)
    veins = draw_veins(
        rng, size,
        count=40, color=(165, 22, 14), alpha=195,
        width0=3.4 * SS, width_min=0.8 * SS, jitter=0.32, steps=48,
        branch_p=0.16, start_r=(0.82, 1.05), blur=0.45 * SS,
    )
    rgb = composite_overlay(rgb, veins, (0.35 + 0.65 * corner) * smoothstep(0.22, 0.50, r))
    caps = draw_veins(
        rng, size,
        count=48, color=(200, 52, 34), alpha=135,
        width0=1.6 * SS, width_min=0.6 * SS, jitter=0.40, steps=32,
        branch_p=0.14, start_r=(0.76, 1.04), blur=0.4 * SS,
    )
    rgb = composite_overlay(rgb, caps, (0.40 + 0.60 * corner) * smoothstep(0.32, 0.68, r))
    # warm umber brow shadow across the top, dipping mid-eye
    xx = np.linspace(0.0, 1.0, size)[None, :]
    yy = np.linspace(0.0, 1.0, size)[:, None]
    wob = angular_noise(rng, size, 9)[None, :] * 0.018
    brow_edge = 0.33 - 0.09 * (np.abs(xx - 0.5) * 2.0) ** 1.4 + wob
    shade = smoothstep(brow_edge + 0.05, brow_edge - 0.16, yy)
    brow_col = np.array([0.16, 0.085, 0.035])
    rgb = rgb * (1.0 - 0.88 * shade[..., None]) + brow_col * (0.88 * shade[..., None])
    rgb *= (1.0 - 0.12 * smoothstep(0.55, 0.30, yy))[..., None]
    return rgb


def gen_werewolf_iris(rng: np.random.Generator, diam: int) -> tuple[np.ndarray, np.ndarray]:
    size = diam * SS
    r, theta = radial_grid(size)
    # amber gold, hot around the pupil, darkening to a thick umber limbus
    rgb = ramp(r, [
        (0.00, (1.00, 0.72, 0.16)),
        (0.28, (0.96, 0.62, 0.10)),
        (0.50, (0.85, 0.52, 0.08)),
        (0.72, (0.60, 0.34, 0.06)),
        (0.90, (0.32, 0.16, 0.04)),
        (1.00, (0.14, 0.07, 0.02)),
    ])
    # high-contrast radial fibers
    window = smoothstep(0.10, 0.28, r) * (1.0 - smoothstep(0.80, 0.95, r))
    fibers = fiber_field(rng, theta, r, cells=72, octaves=3, swirl=0.10)
    fine = fiber_field(rng, theta, r, cells=250, octaves=2, swirl=0.10)
    rgb *= np.clip(1.0 + (0.40 * fibers + 0.28 * fine) * window, 0.0, None)[..., None]
    # hot orange annulus around the pupil zone (visible past mid dilation)
    hot = np.exp(-(((r - 0.45) / 0.10) ** 2))
    rgb += 0.22 * hot[..., None] * np.array([1.0, 0.42, 0.02])
    # thick dark limbal ring
    rgb *= (1.0 - 0.90 * smoothstep(0.80, 0.96, r))[..., None]
    a = 1.0 - smoothstep(0.94, 1.0, r)
    return rgb, a


def gen_werewolf_highlight(rng: np.random.Generator, size: int = 26) -> tuple[np.ndarray, np.ndarray]:
    s = size * SS
    c = (s - 1) / 2.0
    y, x = np.mgrid[0:s, 0:s].astype(np.float64)
    d = np.sqrt((x - c) ** 2 + (y - c) ** 2) / (s / 2.0)
    # small and fierce: hard bright core, faint warm bloom
    a = 0.95 * np.exp(-((d / 0.28) ** 4))
    a = np.maximum(a, 0.30 * np.exp(-((d / 0.55) ** 2)))
    rgb = np.ones((s, s, 3), dtype=np.float64) * np.array([1.0, 0.97, 0.88])
    return rgb, np.clip(a, 0.0, 1.0)


def werewolf_theme_json() -> dict:
    return {
        "name": "werewolf",
        "background": [10, 6, 3],
        "layers": {"sclera": "sclera.png", "iris": "iris.png",
                   "highlight": "highlight.png"},
        "pupil": {"shape": "round", "color": [10, 6, 4], "min_frac": 0.15,
                  "max_frac": 0.55, "slit_width_frac": 0.14},
        "iris_frac": 0.70,
        "gaze_range_px": 52,
        "sclera_parallax": 0.35,
        "eyelids": {"color": [42, 24, 12], "style": "curved", "upper_bias": 0.7},
        "motion": {"saccade_interval": [0.3, 1.5],
                   "saccade_duration": [0.05, 0.12],
                   "wander": 0.6, "crazy": 0.1,
                   "blink_interval": [4.0, 10.0], "blink_speed": 1.3,
                   "pupil_speed": 0.7, "drift": 0.0, "flicker": 0.0},
    }


# ---------------------------------------------------------------------------
# POSSESSED
# ---------------------------------------------------------------------------


def gen_possessed_sclera(rng: np.random.Generator) -> np.ndarray:
    size = SCLERA_SIZE * SS
    r, theta = radial_grid(size)
    # blue-black void with only a wet sheen: 2-4% luminance play, no veins
    rgb = ramp(r, [
        (0.00, (0.036, 0.038, 0.055)),
        (0.55, (0.030, 0.032, 0.048)),
        (1.05, (0.018, 0.020, 0.032)),
    ])
    # large-scale wet-sheen gradient
    sheen = fbm(rng, size, 2, 2)
    rgb *= (0.93 + 0.14 * sheen)[..., None]
    # soft glancing light from the upper-left, as on a wet curve
    xx = np.linspace(0.0, 1.0, size)[None, :]
    yy = np.linspace(0.0, 1.0, size)[:, None]
    glance = np.exp(-(((xx - 0.36) ** 2 + (yy - 0.34) ** 2) / 0.16))
    rgb += glance[..., None] * np.array([0.014, 0.015, 0.022])
    return rgb


def gen_possessed_iris(rng: np.random.Generator, diam: int) -> tuple[np.ndarray, np.ndarray]:
    size = diam * SS
    r, _theta = radial_grid(size)
    # black-on-black: a few percent darker than the sclera, soft-edged
    rgb = ramp(r, [
        (0.00, (0.022, 0.023, 0.036)),
        (0.50, (0.025, 0.026, 0.040)),
        (1.00, (0.028, 0.030, 0.045)),
    ])
    mott = fbm(rng, size, 3, 2)
    rgb *= (0.97 + 0.06 * mott)[..., None]
    # very soft alpha edge: the disc's motion should be subliminal
    a = 1.0 - smoothstep(0.45, 0.98, r)
    return rgb, np.clip(a, 0.0, 1.0)


def gen_possessed_highlight(rng: np.random.Generator, size: int = 24) -> tuple[np.ndarray, np.ndarray]:
    s = size * SS
    c = (s - 1) / 2.0
    y, x = np.mgrid[0:s, 0:s].astype(np.float64)
    d = np.sqrt((x - c) ** 2 + (y - c) ** 2) / (s / 2.0)
    # the star of the theme: crisp wet glint with slight bloom
    a = 1.00 * np.exp(-((d / 0.24) ** 6))
    a = np.maximum(a, 0.38 * np.exp(-((d / 0.70) ** 2)))
    rgb = np.ones((s, s, 3), dtype=np.float64) * np.array([1.0, 1.0, 1.0])
    return rgb, np.clip(a, 0.0, 1.0)


def possessed_theme_json() -> dict:
    return {
        "name": "possessed",
        "background": [2, 2, 4],
        "layers": {"sclera": "sclera.png", "iris": "iris.png",
                   "highlight": "highlight.png"},
        "pupil": {"shape": "none", "color": [0, 0, 0], "min_frac": 0.2,
                  "max_frac": 0.4, "slit_width_frac": 0.1},
        "iris_frac": 0.60,
        "gaze_range_px": 50,
        "sclera_parallax": 0.30,
        "eyelids": {"color": [22, 16, 24], "style": "curved", "upper_bias": 0.6},
        "motion": {"saccade_interval": [2.5, 8.0],
                   "saccade_duration": [0.3, 0.6],
                   "wander": 0.3, "crazy": 0.0,
                   "blink_interval": [10.0, 25.0], "blink_speed": 0.8,
                   "pupil_speed": 0.0, "drift": 0.08, "flicker": 0.0},
    }


# ---------------------------------------------------------------------------
# SERPENT
# ---------------------------------------------------------------------------


def gen_serpent_sclera(rng: np.random.Generator) -> np.ndarray:
    size = SCLERA_SIZE * SS
    r, theta = radial_grid(size)
    # gold-bronze reptile scales: voronoi cells with dark interstices
    edge, cell = voronoi_cells(rng, size, 360)
    lo = np.array([0.52, 0.38, 0.15])
    hi = np.array([0.70, 0.53, 0.22])
    rgb = lo + (hi - lo) * cell[..., None]
    # each scale slightly domed: brighter toward its center
    dome = smoothstep(0.06, 0.55, edge)
    rgb *= (0.87 + 0.22 * dome)[..., None]
    # dark interstices between scales
    seam = 1.0 - smoothstep(0.035, 0.16, edge)
    rgb = rgb * (1.0 - 0.72 * seam[..., None]) + np.array([0.10, 0.065, 0.025]) * (0.72 * seam[..., None])
    # organic mottle so the scales aren't flat
    mott = fbm(rng, size, 6, 3)
    rgb *= (0.90 + 0.20 * mott)[..., None]
    # darken toward the rim
    rgb *= (1.0 - 0.78 * smoothstep(0.45, 1.02, r))[..., None]
    return rgb


def gen_serpent_iris(rng: np.random.Generator, diam: int) -> tuple[np.ndarray, np.ndarray]:
    size = diam * SS
    r, theta = radial_grid(size)
    # acid yellow blazing at the slit, grading to venom green rim
    rgb = ramp(r, [
        (0.00, (0.98, 0.95, 0.22)),
        (0.22, (0.92, 0.88, 0.14)),
        (0.45, (0.68, 0.74, 0.12)),
        (0.68, (0.38, 0.55, 0.12)),
        (0.88, (0.16, 0.32, 0.09)),
        (1.00, (0.06, 0.14, 0.04)),
    ])
    # subtle fibers only
    window = smoothstep(0.10, 0.30, r) * (1.0 - smoothstep(0.80, 0.95, r))
    fibers = fiber_field(rng, theta, r, cells=64, octaves=2, swirl=0.06)
    fine = fiber_field(rng, theta, r, cells=200, octaves=2, swirl=0.06)
    rgb *= (1.0 + (0.14 * fibers + 0.12 * fine) * window)[..., None]
    # faint venom mottling
    mott = fbm(rng, size, 4, 3)
    rgb *= (0.95 + 0.10 * mott)[..., None]
    # dark limbal ring
    rgb *= (1.0 - 0.85 * smoothstep(0.85, 0.97, r))[..., None]
    a = 1.0 - smoothstep(0.95, 1.0, r)
    return rgb, a


def gen_serpent_highlight(rng: np.random.Generator, size: int = 18) -> tuple[np.ndarray, np.ndarray]:
    s = size * SS
    c = (s - 1) / 2.0
    y, x = np.mgrid[0:s, 0:s].astype(np.float64)
    d = np.sqrt((x - c) ** 2 + (y - c) ** 2) / (s / 2.0)
    # thin small glint
    a = 0.95 * np.exp(-((d / 0.26) ** 3))
    rgb = np.ones((s, s, 3), dtype=np.float64) * np.array([1.0, 1.0, 0.94])
    return rgb, np.clip(a, 0.0, 1.0)


def serpent_theme_json() -> dict:
    return {
        "name": "serpent",
        "background": [8, 6, 2],
        "layers": {"sclera": "sclera.png", "iris": "iris.png",
                   "highlight": "highlight.png"},
        "pupil": {"shape": "slit", "color": [4, 6, 3], "min_frac": 0.10,
                  "max_frac": 0.50, "slit_width_frac": 0.18},
        "iris_frac": 0.66,
        "gaze_range_px": 48,
        "sclera_parallax": 0.30,
        "eyelids": {"color": [30, 26, 10], "style": "straight", "upper_bias": 0.4},
        "motion": {"saccade_interval": [3.0, 9.0],
                   "saccade_duration": [0.4, 0.9],
                   "wander": 0.3, "crazy": 0.0,
                   "blink_interval": [45.0, 90.0], "blink_speed": 0.5,
                   "pupil_speed": 0.2, "drift": 0.1, "flicker": 0.0},
    }


# ---------------------------------------------------------------------------
# HAL
# ---------------------------------------------------------------------------


def gen_hal_sclera(rng: np.random.Generator) -> np.ndarray:
    size = SCLERA_SIZE * SS
    r, theta = radial_grid(size)
    # dark metallic housing
    rgb = ramp(r, [
        (0.00, (0.040, 0.030, 0.030)),
        (0.50, (0.032, 0.026, 0.026)),
        (0.80, (0.022, 0.018, 0.018)),
        (1.05, (0.010, 0.008, 0.008)),
    ])
    # brushed-metal hint: fine concentric grooves (1D noise indexed by radius)
    grooves = angular_noise(rng, 2048, 256)
    idx = np.mod(r * 900.0, 2048).astype(np.int64)
    rgb *= (1.0 + 0.16 * grooves[idx] * (1.0 - smoothstep(0.85, 1.05, r)))[..., None]
    # machined highlight ring just inside the panel edge
    rgb += np.exp(-(((r - 0.72) / 0.030) ** 2))[..., None] * np.array([0.030, 0.026, 0.026])
    # deep red ambient glow toward the center, as if the lens leaks light
    rgb += np.exp(-((r / 0.48) ** 2))[..., None] * np.array([0.075, 0.006, 0.005])
    return rgb


def gen_hal_iris(rng: np.random.Generator, diam: int) -> tuple[np.ndarray, np.ndarray]:
    size = diam * SS
    r, _theta = radial_grid(size)
    half = size / 2.0
    pxr = 1.0 / half

    def dark_ring(r0: float, hw_px: float, gain: float) -> np.ndarray:
        return gain * np.clip((hw_px * pxr - np.abs(r - r0)) / (1.4 * pxr) + 0.5, 0.0, 1.0)

    # the lens: bright red-orange core that outlives the max-dilation pupil
    # (pupil radius spans r 0.12..0.30, so the hot zone reaches past 0.32 and
    # the aperture pulse reads as the glow shrinking and swelling)
    rgb = ramp(r, [
        (0.00, (1.00, 0.45, 0.10)),
        (0.20, (0.98, 0.32, 0.06)),
        (0.34, (0.88, 0.16, 0.03)),
        (0.48, (0.52, 0.045, 0.012)),
        (0.64, (0.34, 0.020, 0.008)),
        (0.80, (0.21, 0.012, 0.005)),
        (0.92, (0.29, 0.020, 0.008)),
        (1.00, (0.16, 0.010, 0.005)),
    ])
    # crisp concentric darker element rings — precise geometry, no noise
    dark = np.zeros_like(r)
    for r0, hw, g in ((0.42, 1.6, 0.45), (0.56, 1.6, 0.50), (0.70, 1.6, 0.50),
                      (0.82, 1.6, 0.45), (0.92, 2.2, 0.55)):
        dark += dark_ring(r0, hw, g)
    rgb *= (1.0 - np.clip(dark, 0.0, 0.85))[..., None]
    # faint hot rim hugging the glowing core's edge (lens glass refraction)
    rgb += (0.16 * np.exp(-(((r - 0.36) / 0.030) ** 2)))[..., None] * np.array([1.0, 0.42, 0.10])
    # alpha: opaque lens + soft red halo bleeding onto the housing
    core_a = 1.0 - smoothstep(0.86, 0.93, r)
    halo_a = 0.45 * np.exp(-(((r - 0.88).clip(min=0.0) / 0.10) ** 1.8))
    a = np.maximum(core_a, halo_a)
    a *= 1.0 - smoothstep(0.97, 1.0, r)
    return rgb, np.clip(a, 0.0, 1.0)


def gen_hal_highlight(rng: np.random.Generator, size: int = 14) -> tuple[np.ndarray, np.ndarray]:
    s = size * SS
    c = (s - 1) / 2.0
    y, x = np.mgrid[0:s, 0:s].astype(np.float64)
    d = np.sqrt((x - c) ** 2 + (y - c) ** 2) / (s / 2.0)
    # tiny sharp glass specular
    a = 0.92 * np.exp(-((d / 0.30) ** 6))
    rgb = np.ones((s, s, 3), dtype=np.float64) * np.array([1.0, 0.96, 0.94])
    return rgb, np.clip(a, 0.0, 1.0)


def hal_theme_json() -> dict:
    return {
        "name": "hal",
        "background": [2, 0, 0],
        "layers": {"sclera": "sclera.png", "iris": "iris.png",
                   "highlight": "highlight.png"},
        "pupil": {"shape": "round", "color": [26, 3, 2], "min_frac": 0.12,
                  "max_frac": 0.30, "slit_width_frac": 0.14},
        "iris_frac": 0.60,
        "gaze_range_px": 15,
        "sclera_parallax": 0.30,
        "eyelids": {"color": [0, 0, 0], "style": "straight", "upper_bias": 0.5},
        "motion": {"saccade_interval": [4.0, 10.0],
                   "saccade_duration": [0.5, 1.0],
                   "wander": 0.05, "crazy": 0.0,
                   "blink_interval": [90.0, 180.0], "blink_speed": 1.0,
                   "pupil_speed": 0.4, "drift": 0.02, "flicker": 0.04},
    }


# ---------------------------------------------------------------------------
# theme assembly / validation / check renders
# ---------------------------------------------------------------------------

GENERATORS = {
    "human": (gen_human_sclera, gen_human_iris, gen_human_highlight, human_theme_json),
    "demon": (gen_demon_sclera, gen_demon_iris, gen_demon_highlight, demon_theme_json),
    "ghost": (gen_ghost_sclera, gen_ghost_iris, gen_ghost_highlight, ghost_theme_json),
    "doomguy": (gen_doomguy_sclera, gen_doomguy_iris, gen_doomguy_highlight,
                doomguy_theme_json),
    "vampire": (gen_vampire_sclera, gen_vampire_iris, gen_vampire_highlight,
                vampire_theme_json),
    "zombie": (gen_zombie_sclera, gen_zombie_iris, gen_zombie_highlight,
               zombie_theme_json),
    "cyber": (gen_cyber_sclera, gen_cyber_iris, gen_cyber_highlight,
              cyber_theme_json),
    "cat": (gen_cat_sclera, gen_cat_iris, gen_cat_highlight, cat_theme_json),
    "wisp": (gen_wisp_sclera, gen_wisp_iris, None, wisp_theme_json),
    "werewolf": (gen_werewolf_sclera, gen_werewolf_iris, gen_werewolf_highlight,
                 werewolf_theme_json),
    "possessed": (gen_possessed_sclera, gen_possessed_iris,
                  gen_possessed_highlight, possessed_theme_json),
    "serpent": (gen_serpent_sclera, gen_serpent_iris, gen_serpent_highlight,
                serpent_theme_json),
    "hal": (gen_hal_sclera, gen_hal_iris, gen_hal_highlight, hal_theme_json),
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
    if gen_highlight is not None and "highlight" in spec["layers"]:
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
