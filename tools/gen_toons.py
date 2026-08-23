#!/usr/bin/env python3
"""The Toons collection — flat-cartoon character themes.

Where gen_art.py grows *organic* eyes (fbm noise, veins, fiber streaks) and
gen_arcana.py builds spinning polar mechanisms, this module draws *paper-flat
cartoon characters*: uniform fills and crisp boundaries, anti-aliased only at
the edges (analytic ~1px smoothstep for numpy shapes, a supersampled BOX
downscale for drawn strokes), no gradients or organic noise unless a theme
note explicitly calls for one. Every iris_frac is chosen so that
``round(iris_frac * 240)`` equals the authored sprite size — the renderer
bakes the art 1:1 with no LANCZOS resize, keeping the cartoon edges hard.

South Park pack (same cutout language as gen_art's southpark theme):
kenny, canadian, tweek, towelie.  Toon hall of fame: bender, spongebob,
anime, rick.

Deterministic: same per-asset rng scheme as gen_art (theme_rng).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_art as G  # noqa: E402  (shared helpers + constants)

TOON_NAMES = ("kenny", "canadian", "tweek", "towelie",
              "bender", "spongebob", "anime", "rick")

MASK_SS = 4  # supersample factor for stroke masks (BOX-averaged down)


# ---------------------------------------------------------------------------
# flat-cartoon helpers
# ---------------------------------------------------------------------------


def C(r: int, g: int, b: int) -> np.ndarray:
    """A flat fill color as a float RGB triple."""
    return np.array([r, g, b], dtype=np.float64) / 255.0


def flat(size: int, rgb: tuple[int, int, int]) -> np.ndarray:
    """A uniform flat-color canvas."""
    return np.zeros((size, size, 3), dtype=np.float64) + C(*rgb)


def grids(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(rr, theta, dx, dy): radius in *pixels* from the canvas center plus
    signed pixel offsets (screen y grows downward)."""
    r, th = G.radial_grid(size)
    rr = r * (size / 2.0)
    dx = rr * np.cos(th)
    dy = rr * np.sin(th)
    return rr, th, dx, dy


def inside(d: np.ndarray, aa: float = 1.5) -> np.ndarray:
    """1 where the signed distance d is inside (negative), 0 outside —
    anti-aliased only across the last ``aa`` pixels."""
    return 1.0 - G.smoothstep(-aa, 0.0, d)


def band(v: np.ndarray, lo: float, hi: float, aa: float = 1.3) -> np.ndarray:
    """1 where lo <= v <= hi, 0 elsewhere, edge-AA over ``aa``."""
    return G.smoothstep(lo - aa, lo, v) * (1.0 - G.smoothstep(hi, hi + aa, v))


def blend(dst: np.ndarray, mask: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Lerp the canvas toward a flat color where the mask is set."""
    return dst + mask[..., None] * (rgb - dst)


def zigzag(theta: np.ndarray, teeth: int, phase: float = 0.0) -> np.ndarray:
    """Periodic triangle wave over angle, in [-1, 1] — cutout scallop edges."""
    t = np.mod(theta / (2.0 * math.pi) * teeth + phase, 1.0)
    return np.abs(t - 0.5) * 4.0 - 1.0


def mask_overlay(size: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    """A MASK_SS-supersampled L-mode stroke canvas (draw with fill=255)."""
    img = Image.new("L", (size * MASK_SS, size * MASK_SS), 0)
    return img, ImageDraw.Draw(img)


def composite_mask(rgb: np.ndarray, overlay: Image.Image,
                   color: np.ndarray) -> np.ndarray:
    """BOX-average the supersampled mask down (edge-only AA, no LANCZOS
    ringing over flat fills) and blend its flat color onto the canvas."""
    small = overlay.resize((overlay.width // MASK_SS,) * 2, Image.BOX)
    return blend(rgb, np.asarray(small, dtype=np.float64) / 255.0, color)


def draw_squiggle(draw: ImageDraw.ImageDraw, rng: np.random.Generator,
                  cx: float, cy: float, ang: float, length: float,
                  amp: float, width: float, segs: int = 6) -> None:
    """A jagged flat polyline (stress vein / bloodshot squiggle) centered at
    (cx, cy), running along ``ang``, zigzagging perpendicular by ~amp px."""
    ca, sa = math.cos(ang), math.sin(ang)
    pts = []
    for i in range(segs + 1):
        t = (i / segs - 0.5) * length
        off = 0.0
        if 0 < i < segs:
            off = (amp if i % 2 else -amp) * float(rng.uniform(0.55, 1.0))
        pts.append(((cx + ca * t - sa * off) * MASK_SS,
                    (cy + sa * t + ca * off) * MASK_SS))
    draw.line(pts, fill=255, width=max(1, round(width * MASK_SS)),
              joint="curve")


def tapered_stroke(draw: ImageDraw.ImageDraw, p0: tuple[float, float],
                   p1: tuple[float, float], bow: tuple[float, float],
                   w0: float, w1: float, steps: int = 14) -> None:
    """A flat tapered lash stroke: quadratic bezier from p0 to p1 (control
    point offset by ``bow``), width lerping w0 -> w1, drawn as one polygon."""
    mx = (p0[0] + p1[0]) / 2.0 + bow[0]
    my = (p0[1] + p1[1]) / 2.0 + bow[1]
    left: list[tuple[float, float]] = []
    right: list[tuple[float, float]] = []
    for i in range(steps + 1):
        t = i / steps
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * mx + t * t * p1[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * my + t * t * p1[1]
        ddx = 2 * (1 - t) * (mx - p0[0]) + 2 * t * (p1[0] - mx)
        ddy = 2 * (1 - t) * (my - p0[1]) + 2 * t * (p1[1] - my)
        n = math.hypot(ddx, ddy) or 1.0
        nx, ny = -ddy / n, ddx / n
        w = (w0 + (w1 - w0) * t) / 2.0
        left.append(((x + nx * w) * MASK_SS, (y + ny * w) * MASK_SS))
        right.append(((x - nx * w) * MASK_SS, (y - ny * w) * MASK_SS))
    draw.polygon(left + right[::-1], fill=255)


def star_points(cx: float, cy: float, s: float,
                k: float = 0.30) -> list[tuple[float, float]]:
    """A 4-point sparkle glint polygon (supersampled coords)."""
    pts = [(cx, cy - s), (cx + k * s, cy - k * s), (cx + s, cy),
           (cx + k * s, cy + k * s), (cx, cy + s), (cx - k * s, cy + k * s),
           (cx - s, cy), (cx - k * s, cy - k * s)]
    return [(x * MASK_SS, y * MASK_SS) for x, y in pts]


def dot_iris(diam: int) -> tuple[np.ndarray, np.ndarray]:
    """Southpark-style solid black bead: flat core, AA only at the disc rim."""
    rr = grids(diam)[0]
    rgb = np.zeros((diam, diam, 3), dtype=np.float64)
    return rgb, inside(rr - diam / 2.0, aa=1.6)


def base_json(name: str, *, iris_frac: float, gaze_px: int,
              lid_color: tuple[int, int, int], lid_style: str,
              upper_bias: float, motion: dict,
              background: tuple[int, int, int],
              pupil: dict | None = None, highlight: bool = False,
              parallax: float = 0.0) -> dict:
    layers = {"sclera": "sclera.png", "iris": "iris.png"}
    if highlight:
        layers["highlight"] = "highlight.png"
    return {
        "name": name,
        "background": list(background),
        "layers": layers,
        "pupil": pupil or {"shape": "none", "color": [0, 0, 0],
                           "min_frac": 0.3, "max_frac": 0.3,
                           "slit_width_frac": 0.15},
        "iris_frac": iris_frac,
        "gaze_range_px": gaze_px,
        "sclera_parallax": parallax,
        "eyelids": {"color": list(lid_color), "style": lid_style,
                    "upper_bias": upper_bias},
        "motion": motion,
    }


# ---------------------------------------------------------------------------
# KENNY — the parka hood: orange, fur-trim ring, beady dot peeking out
# ---------------------------------------------------------------------------

KENNY_ORANGE = (238, 130, 20)


def gen_kenny_sclera(rng: np.random.Generator) -> np.ndarray:
    size = G.SCLERA_SIZE
    rr, th, _, _ = grids(size)
    rgb = flat(size, KENNY_ORANGE)
    # fur trim: scalloped tan-brown band around the hood opening. Opening
    # ~55% of the panel (r=66), band ~12% of the panel (~29 px) wide.
    r_in = 66.0 + 3.0 * zigzag(th, 32)
    r_out = 95.0 + 4.5 * zigzag(th, 32, phase=0.5)
    rgb = blend(rgb, inside(rr - r_out), C(205, 160, 96))
    # flat two-tone tufts so the trim reads as fur, not a plain ring
    tuft = G.smoothstep(0.1, 0.5, zigzag(th, 16, phase=0.25))
    rgb = blend(rgb, tuft * inside(rr - r_out) * inside(r_in - rr), C(190, 146, 80))
    rgb = blend(rgb, inside(rr - r_in), C(255, 255, 255))
    # thin paper-cutout outlines along both scalloped edges
    dark = C(96, 66, 30)
    rgb = blend(rgb, inside(np.abs(rr - r_out) - 1.3, aa=1.1), dark)
    rgb = blend(rgb, inside(np.abs(rr - r_in) - 1.3, aa=1.1), dark)
    return rgb


def gen_kenny_iris(rng: np.random.Generator, diam: int) -> tuple[np.ndarray, np.ndarray]:
    return dot_iris(diam)


def kenny_json() -> dict:
    return base_json(
        "kenny", iris_frac=0.1417, gaze_px=28,  # 0.1417 * 240 -> 34 px exactly
        lid_color=KENNY_ORANGE, lid_style="curved", upper_bias=0.5,
        background=(30, 16, 4),
        motion={"saccade_interval": [0.5, 2.2], "saccade_duration": [0.08, 0.15],
                "wander": 0.55, "crazy": 0.1, "blink_interval": [2.0, 6.0],
                "blink_speed": 1.2, "pupil_speed": 0.0, "drift": 0.0,
                "flicker": 0.0},
    )


# ---------------------------------------------------------------------------
# CANADIAN — Terrance & Phillip: beady black dot on flat skin, no sclera
# ---------------------------------------------------------------------------

CANADIAN_SKIN = (255, 224, 177)


def gen_canadian_sclera(rng: np.random.Generator) -> np.ndarray:
    # no white of the eye at all: the whole panel is flat flappy-head skin
    return flat(G.SCLERA_SIZE, CANADIAN_SKIN)


def gen_canadian_iris(rng: np.random.Generator, diam: int) -> tuple[np.ndarray, np.ndarray]:
    return dot_iris(diam)


def canadian_json() -> dict:
    return base_json(
        "canadian", iris_frac=0.2, gaze_px=70,  # 0.2 * 240 -> 48 px exactly
        lid_color=CANADIAN_SKIN, lid_style="curved", upper_bias=0.55,
        background=(36, 26, 16),
        motion={"saccade_interval": [0.35, 1.5], "saccade_duration": [0.06, 0.12],
                "wander": 0.75, "crazy": 0.15, "blink_interval": [1.5, 4.5],
                "blink_speed": 1.3, "pupil_speed": 0.0, "drift": 0.0,
                "flicker": 0.0},
    )


# ---------------------------------------------------------------------------
# TWEEK — standard SP white eye under caffeine panic: red stress squiggles
# ---------------------------------------------------------------------------


def gen_tweek_sclera(rng: np.random.Generator) -> np.ndarray:
    size = G.SCLERA_SIZE
    rgb = flat(size, (255, 255, 255))
    overlay, draw = mask_overlay(size)
    c = size / 2.0
    # 4 thin flat red stress squiggles near the visible rim (screen y down:
    # 0 deg = right, 90 = bottom; the resting upper lid hides the very top)
    for ang_deg, rad in ((8.0, 96.0), (118.0, 94.0), (168.0, 97.0), (302.0, 95.0)):
        a = math.radians(ang_deg)
        draw_squiggle(draw, rng, c + rad * math.cos(a), c + rad * math.sin(a),
                      a + math.pi / 2.0, length=float(rng.uniform(30.0, 42.0)),
                      amp=3.6, width=2.7)
    return composite_mask(rgb, overlay, C(203, 32, 28))


def gen_tweek_iris(rng: np.random.Generator, diam: int) -> tuple[np.ndarray, np.ndarray]:
    return dot_iris(diam)


def tweek_json() -> dict:
    return base_json(
        "tweek", iris_frac=0.15, gaze_px=70,  # 0.15 * 240 -> 36 px exactly
        lid_color=(252, 216, 168), lid_style="curved", upper_bias=0.55,
        background=(24, 22, 18),
        motion={"saccade_interval": [0.1, 0.45], "saccade_duration": [0.03, 0.06],
                "wander": 0.5, "crazy": 0.25, "blink_interval": [0.8, 2.5],
                "blink_speed": 2.0, "pupil_speed": 0.0, "drift": 0.0,
                "flicker": 0.05},
    )


# ---------------------------------------------------------------------------
# TOWELIE — towel-blue field, smaller white eye oval, stoned pink halo
# ---------------------------------------------------------------------------

TOWEL_BLUE = (150, 205, 224)


def gen_towelie_sclera(rng: np.random.Generator) -> np.ndarray:
    size = G.SCLERA_SIZE
    _, _, dx, dy = grids(size)
    rgb = flat(size, TOWEL_BLUE)
    # white eye oval ~62% of the panel; signed px distance to the ellipse rim
    rx, ry = 75.0, 72.0
    d_px = (np.sqrt((dx / rx) ** 2 + (dy / ry) ** 2) - 1.0) * ry
    rgb = blend(rgb, inside(d_px - 3.8, aa=1.4), C(226, 118, 112))  # stoned halo
    rgb = blend(rgb, inside(d_px, aa=1.4), C(255, 255, 255))
    return rgb


def gen_towelie_iris(rng: np.random.Generator, diam: int) -> tuple[np.ndarray, np.ndarray]:
    return dot_iris(diam)


def towelie_json() -> dict:
    return base_json(
        # lid is a slightly deeper towel blue than the field so the straight
        # droopy edge actually reads as it sags over the white oval
        "towelie", iris_frac=0.13, gaze_px=44,  # 0.13 * 240 -> 31 px
        lid_color=(124, 178, 200), lid_style="straight", upper_bias=0.85,
        background=(20, 30, 34),
        motion={"saccade_interval": [2.5, 7.0], "saccade_duration": [0.5, 1.1],
                "wander": 0.35, "crazy": 0.35, "blink_interval": [5.0, 14.0],
                "blink_speed": 0.4, "pupil_speed": 0.0, "drift": 0.3,
                "flicker": 0.0},
    )


# ---------------------------------------------------------------------------
# BENDER — visor-slot metal, huge white disc, big black robot pupil
# ---------------------------------------------------------------------------

BENDER_METAL = (158, 163, 168)


def gen_bender_sclera(rng: np.random.Generator) -> np.ndarray:
    size = G.SCLERA_SIZE
    rr, _, _, dy = grids(size)
    rgb = flat(size, BENDER_METAL)
    # darker horizontal visor-slot band across the middle, ~62% of height
    half = 74.0
    slot = inside(np.abs(dy) - half, aa=1.2)
    rgb = blend(rgb, slot, C(133, 138, 144))
    # subtle rolled-edge shading just inside the slot edges (allowed gradient)
    edge_t = 1.0 - G.smoothstep(0.0, 12.0, half - np.abs(dy))
    rgb = blend(rgb, slot * edge_t * 0.9, C(106, 112, 120))
    # thin darker recess ring just inside the panel edge (housing lip)
    rgb = blend(rgb, band(rr, 111.0, 115.5), C(104, 109, 116))
    return rgb


def gen_bender_iris(rng: np.random.Generator, diam: int) -> tuple[np.ndarray, np.ndarray]:
    rr = grids(diam)[0]
    half = diam / 2.0
    rgb = flat(diam, (248, 248, 245))  # flat white eye disc (matte plastic)
    rgb = blend(rgb, band(rr, half - 4.6, half - 1.8), C(172, 176, 181))
    return rgb, inside(rr - half, aa=1.5)


def bender_json() -> dict:
    return base_json(
        "bender", iris_frac=0.725, gaze_px=18,  # 0.725 * 240 -> 174 px exactly
        lid_color=(150, 155, 160), lid_style="straight", upper_bias=0.5,
        background=(26, 28, 30),
        pupil={"shape": "round", "color": [0, 0, 0], "min_frac": 0.42,
               "max_frac": 0.5, "slit_width_frac": 0.15},
        motion={"saccade_interval": [1.2, 4.5], "saccade_duration": [0.08, 0.2],
                "wander": 0.45, "crazy": 0.0, "blink_interval": [4.0, 9.0],
                "blink_speed": 1.6, "pupil_speed": 0.1, "drift": 0.0,
                "flicker": 0.0},
    )


# ---------------------------------------------------------------------------
# SPONGEBOB — lashes on white, light-blue iris, glossy highlight, yellow lids
# ---------------------------------------------------------------------------


def gen_spongebob_sclera(rng: np.random.Generator) -> np.ndarray:
    size = G.SCLERA_SIZE
    rr, _, _, _ = grids(size)
    rgb = flat(size, (255, 255, 255))
    # the faintest cool shading toward the rim (allowed gradient)
    rgb = blend(rgb, G.smoothstep(84.0, 122.0, rr) * 0.85, C(228, 235, 242))
    # THREE thick flat tapered black eyelashes radiating from the top edge
    overlay, draw = mask_overlay(size)
    c = size / 2.0
    for ang_deg, bow_sign in ((64.0, 1.0), (90.0, 0.0), (116.0, -1.0)):
        a = math.radians(ang_deg)
        ux, uy = math.cos(a), -math.sin(a)  # unit vector toward the top rim
        base = (c + ux * 124.0, c + uy * 124.0)  # thick end, past the vignette
        tip = (c + ux * 62.0, c + uy * 62.0)
        bow = (-uy * 7.0 * bow_sign, ux * 7.0 * bow_sign)  # outward splay
        tapered_stroke(draw, base, tip, bow, w0=11.5, w1=2.2)
    return composite_mask(rgb, overlay, C(12, 12, 12))


def gen_spongebob_iris(rng: np.random.Generator, diam: int) -> tuple[np.ndarray, np.ndarray]:
    rr = grids(diam)[0]
    half = diam / 2.0
    rgb = flat(diam, (94, 159, 222))  # SpongeBob light blue, dead flat
    rgb = blend(rgb, G.smoothstep(half - 4.8, half - 3.0, rr), C(56, 106, 176))
    return rgb, inside(rr - half, aa=1.5)


def gen_spongebob_highlight(rng: np.random.Generator, size: int = 92) -> tuple[np.ndarray, np.ndarray]:
    # big glossy white circle (authored at SS scale, saved at 46 px)
    rr = grids(size)[0]
    rgb = np.ones((size, size, 3), dtype=np.float64)
    return rgb, inside(rr - (size / 2.0 - 1.5), aa=2.5)


def spongebob_json() -> dict:
    return base_json(
        "spongebob", iris_frac=0.55, gaze_px=40,  # 0.55 * 240 -> 132 px exactly
        lid_color=(255, 229, 63), lid_style="curved", upper_bias=0.55,
        background=(28, 26, 12), highlight=True,
        pupil={"shape": "round", "color": [0, 0, 0], "min_frac": 0.35,
               "max_frac": 0.5, "slit_width_frac": 0.15},
        motion={"saccade_interval": [0.5, 2.0], "saccade_duration": [0.07, 0.14],
                "wander": 0.6, "crazy": 0.05, "blink_interval": [2.0, 6.0],
                "blink_speed": 1.2, "pupil_speed": 0.4, "drift": 0.0,
                "flicker": 0.0},
    )


# ---------------------------------------------------------------------------
# ANIME — big sparkly shoujo eye: gradient iris, sheen spokes, baked sparkles
# ---------------------------------------------------------------------------


def gen_anime_sclera(rng: np.random.Generator) -> np.ndarray:
    size = G.SCLERA_SIZE
    _, _, dx, dy = grids(size)
    py = dy + 120.0  # y measured from the visible panel top
    rgb = flat(size, (255, 255, 255))
    # soft lash shadow falling from the top (allowed gradient)
    rgb = blend(rgb, (1.0 - G.smoothstep(6.0, 92.0, py)) * 0.6, C(198, 203, 214))
    # bold black upper lash line: an arc band parallel to the curved lid edge
    sag = 260.0 - np.sqrt(260.0 ** 2 - np.clip(dx, -119.9, 119.9) ** 2)
    d_top = py - (33.0 + sag)
    thick = 13.0 - 6.0 * G.smoothstep(68.0, 112.0, np.abs(dx))
    lash = G.smoothstep(-1.2, 0.0, d_top) * (1.0 - G.smoothstep(thick - 1.2, thick, d_top))
    return blend(rgb, lash, C(18, 16, 20))


def gen_anime_iris(rng: np.random.Generator, diam: int) -> tuple[np.ndarray, np.ndarray]:
    rr, th, _, dy = grids(diam)
    half = diam / 2.0
    rn = rr / half
    # vertical gradient sky-blue -> deep navy (G.ramp works on any field)
    ty = np.clip(dy / diam + 0.5, 0.0, 1.0)
    rgb = G.ramp(ty, [
        (0.00, (0.52, 0.79, 0.97)),
        (0.42, (0.24, 0.47, 0.83)),
        (0.75, (0.10, 0.20, 0.52)),
        (1.00, (0.05, 0.10, 0.30)),
    ])
    # fine radial sheen lines, mid-annulus only
    p = np.mod(th / (2.0 * math.pi) * 56.0, 1.0)
    spokes = (1.0 - G.smoothstep(0.035, 0.10, np.minimum(p, 1.0 - p)))
    spokes = spokes * band(rn, 0.42, 0.85, aa=0.05) * 0.32
    rgb = blend(rgb, spokes, C(178, 216, 250))
    # thick dark limbal ring + dark core (pupil shape is "none")
    rgb = blend(rgb, G.smoothstep(0.865, 0.905, rn), C(12, 20, 54))
    rgb = blend(rgb, inside(rr - 0.30 * half, aa=2.4), C(9, 11, 26))
    # SPARKLES, baked in: big upper-left, small lower-right, 2 star glints
    overlay, draw = mask_overlay(diam)
    c = diam / 2.0
    for ox, oy, rad in ((-0.30, -0.30, 0.205), (0.30, 0.40, 0.085)):
        x, y, r_px = c + ox * half, c + oy * half, rad * half
        draw.ellipse([(x - r_px) * MASK_SS, (y - r_px) * MASK_SS,
                      (x + r_px) * MASK_SS, (y + r_px) * MASK_SS], fill=255)
    draw.polygon(star_points(c + 0.44 * half, c - 0.14 * half, 0.075 * half), fill=255)
    draw.polygon(star_points(c - 0.10 * half, c + 0.44 * half, 0.06 * half), fill=255)
    rgb = composite_mask(rgb, overlay, C(255, 255, 255))
    return rgb, inside(rr - half, aa=1.5)


def anime_json() -> dict:
    return base_json(
        "anime", iris_frac=0.8, gaze_px=16,  # 0.8 * 240 -> 192 px exactly
        lid_color=(16, 14, 18), lid_style="curved", upper_bias=0.7,
        background=(10, 8, 12),
        motion={"saccade_interval": [1.0, 3.5], "saccade_duration": [0.1, 0.25],
                "wander": 0.5, "crazy": 0.0, "blink_interval": [3.0, 8.0],
                "blink_speed": 0.9, "pupil_speed": 0.0, "drift": 0.05,
                "flicker": 0.0},
    )


# ---------------------------------------------------------------------------
# RICK — morning-after: under-eye sag arcs, bloodshot squiggles, tiny pupil
# ---------------------------------------------------------------------------


def gen_rick_sclera(rng: np.random.Generator) -> np.ndarray:
    size = G.SCLERA_SIZE
    _, _, dx, dy = grids(size)
    rgb = flat(size, (255, 255, 255))
    # pale blue-gray under-eye sag arcs along the bottom (same 260 px arc
    # family as the renderer's curved lids, so they track the lower lid)
    # a stack of narrowing arcs: the classic cartoon eye-bag wrinkle pile
    sag = 260.0 - np.sqrt(260.0 ** 2 - np.clip(dx, -119.9, 119.9) ** 2)
    for y0, w, x0, x1 in ((54.0, 5.4, 66.0, 88.0), (71.0, 4.4, 46.0, 66.0),
                          (86.0, 3.6, 28.0, 46.0)):
        fade = 1.0 - G.smoothstep(x0, x1, np.abs(dx))
        d = np.abs(dy - (y0 - sag)) - w / 2.0
        rgb = blend(rgb, inside(d, aa=1.3) * fade, C(170, 186, 202))
    # 3 flat red bloodshot squiggles
    overlay, draw = mask_overlay(size)
    c = size / 2.0
    for ang_deg, rad in ((14.0, 94.0), (166.0, 92.0), (299.0, 96.0)):
        a = math.radians(ang_deg)
        draw_squiggle(draw, rng, c + rad * math.cos(a), c + rad * math.sin(a),
                      a + math.pi / 2.0, length=float(rng.uniform(28.0, 40.0)),
                      amp=3.2, width=2.2)
    return composite_mask(rgb, overlay, C(198, 44, 38))


def gen_rick_iris(rng: np.random.Generator, diam: int) -> tuple[np.ndarray, np.ndarray]:
    return dot_iris(diam)


def rick_json() -> dict:
    return base_json(
        "rick", iris_frac=0.1, gaze_px=78,  # 0.1 * 240 -> 24 px exactly
        lid_color=(224, 205, 178), lid_style="curved", upper_bias=0.6,
        background=(18, 14, 12),
        motion={"saccade_interval": [0.3, 3.5], "saccade_duration": [0.05, 0.3],
                "wander": 0.65, "crazy": 0.2, "blink_interval": [3.0, 9.0],
                "blink_speed": 0.8, "pupil_speed": 0.0, "drift": 0.15,
                "flicker": 0.0},
    )


# ---------------------------------------------------------------------------
# registry + CLI (mirrors gen_art's generate_theme, optional highlight layer)
# ---------------------------------------------------------------------------

TOON_GENERATORS = {
    "kenny": (gen_kenny_sclera, gen_kenny_iris, None, kenny_json),
    "canadian": (gen_canadian_sclera, gen_canadian_iris, None, canadian_json),
    "tweek": (gen_tweek_sclera, gen_tweek_iris, None, tweek_json),
    "towelie": (gen_towelie_sclera, gen_towelie_iris, None, towelie_json),
    "bender": (gen_bender_sclera, gen_bender_iris, None, bender_json),
    "spongebob": (gen_spongebob_sclera, gen_spongebob_iris,
                  gen_spongebob_highlight, spongebob_json),
    "anime": (gen_anime_sclera, gen_anime_iris, None, anime_json),
    "rick": (gen_rick_sclera, gen_rick_iris, None, rick_json),
}


def generate_theme(name: str, out_dir: Path) -> Path:
    gen_sclera, gen_iris, gen_highlight, make_json = TOON_GENERATORS[name]
    spec = make_json()
    G.validate_theme_json(spec)
    tdir = out_dir / name
    tdir.mkdir(parents=True, exist_ok=True)
    # round (not ceil): matches the renderer's round(iris_frac * SIZE), so the
    # baked sprite is always used 1:1 with no bake-time LANCZOS softening
    iris_diam = round(spec["iris_frac"] * G.PANEL)
    print(f"[{name}] sclera ...")
    G.save_rgb(gen_sclera(G.theme_rng(name, "sclera")), tdir / "sclera.png",
               G.SCLERA_SIZE)
    print(f"[{name}] iris {iris_diam}px ...")
    rgb, a = gen_iris(G.theme_rng(name, "iris"), iris_diam)
    G.save_rgba(rgb, a, tdir / "iris.png", iris_diam)
    if gen_highlight is not None and "highlight" in spec["layers"]:
        print(f"[{name}] highlight ...")
        rgb, a = gen_highlight(G.theme_rng(name, "highlight"))
        scale = getattr(gen_highlight, "scale", G.SS)
        G.save_rgba(rgb, a, tdir / "highlight.png", rgb.shape[0] // scale)
    (tdir / "theme.json").write_text(json.dumps(spec, indent=2) + "\n")
    print(f"[{name}] wrote {tdir}/")
    return tdir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the Toons theme collection")
    parser.add_argument("--theme", choices=TOON_NAMES, default=None,
                        help="single theme (default: all)")
    parser.add_argument("--out", type=Path, default=Path("themes"))
    args = parser.parse_args(argv)
    for name in ([args.theme] if args.theme else TOON_NAMES):
        generate_theme(name, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
