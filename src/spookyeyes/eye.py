"""Eye renderer: turn an `EyeState` into a 240x240 RGB frame.

Everything expensive is pre-baked at construction time (iris sprites for each
pupil dilation step, eyelid alpha masks for each openness step, the round
vignette). `render()` is a crop, two or three alpha pastes, and one integer
numpy pass — no per-frame resize/rotate — to stay inside the <= 8 ms per eye
budget on a Pi 3.
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image, ImageDraw

from .model import SIZE, EyeState
from .themes import Theme

logger = logging.getLogger("spookyeyes.eye")

PUPIL_STEPS = 12          # pre-baked iris sprites across the dilation range
LID_STEPS = 24            # pre-baked lid masks quantizing openness 0..1.25
SPIN_STEPS = 36           # pre-baked rotation steps for spinning irises
MAX_OPENNESS = 1.25       # startled-wide upper bound of EyeState.openness

_LID_ARC_RADIUS = 260.0   # px; curvature of the "curved" eyelid edge arcs
_HIGHLIGHT_COUNTER = 0.15  # highlight counter-moves this fraction of gaze_px
_HIGHLIGHT_OFFSET_FRAC = (-0.18, -0.20)  # highlight offset from iris center, in iris diameters


def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


class EyeRenderer:
    """Renders one eye. Construct once per theme (cheap-ish, does the baking);
    call `render(state)` per frame (fast)."""

    PUPIL_STEPS = PUPIL_STEPS
    LID_STEPS = LID_STEPS

    def __init__(self, theme: Theme, spin_dir: int = 1) -> None:
        """``spin_dir=-1`` counter-rotates a spinning iris (the app passes it
        for the right eye when the theme sets ``iris_spin_mirror``)."""
        self._theme = theme
        self._range = float(theme.gaze_range_px)
        self._parallax = float(theme.sclera_parallax)
        self._center = SIZE / 2.0

        self._sclera = theme.sclera
        sw, sh = self._sclera.size
        self._base_x = (sw - SIZE) // 2
        self._base_y = (sh - SIZE) // 2
        self._max_x = sw - SIZE
        self._max_y = sh - SIZE

        self._iris_sprites = self._bake_iris_sprites(theme)
        iris_d = self._iris_sprites[0].width
        self._iris_half = iris_d / 2.0

        # Spinning iris: pre-bake SPIN_STEPS rotated copies of the composed
        # sprite (the loader guarantees pupil shape "none", so all pupil steps
        # are identical and one rotation set suffices). render() then just
        # indexes by time — no per-frame rotation.
        self._spin_sprites: list[Image.Image] | None = None
        self._spin_rate = 0.0
        rpm = float(getattr(theme, "iris_spin_rpm", 0.0))
        if rpm:
            base_sprite = self._iris_sprites[0]
            step_deg = 360.0 / SPIN_STEPS
            self._spin_sprites = [
                base_sprite.rotate(-step_deg * i, resample=Image.BICUBIC)
                for i in range(SPIN_STEPS)
            ]
            direction = 1.0 if spin_dir >= 0 else -1.0
            self._spin_rate = rpm / 60.0 * SPIN_STEPS * direction  # steps/s

        self._highlight = theme.highlight
        if self._highlight is not None:
            self._hl_off = (
                _HIGHLIGHT_OFFSET_FRAC[0] * iris_d - self._highlight.width / 2.0,
                _HIGHLIGHT_OFFSET_FRAC[1] * iris_d - self._highlight.height / 2.0,
            )

        self._lid_masks, self._lid_empty = self._bake_lid_masks(theme)
        self._lid_fill = Image.new("RGB", (SIZE, SIZE), theme.lid_color)
        self._vig_q8 = self._bake_vignette()
        logger.debug(
            "baked renderer for theme %r: iris %dpx, %d pupil sprites, %d lid masks",
            theme.name, iris_d, PUPIL_STEPS, LID_STEPS,
        )

    # ------------------------------------------------------------------ baking

    @staticmethod
    def _bake_iris_sprites(theme: Theme) -> list[Image.Image]:
        """PUPIL_STEPS RGBA sprites: iris art with the pupil drawn at each
        dilation step lerp(min_frac, max_frac, k/(STEPS-1))."""
        iris_d = max(2, round(theme.iris_frac * SIZE))
        base = theme.iris
        if base.size != (iris_d, iris_d):
            logger.debug(
                "theme %r: resizing iris art %s -> %dx%d (once, at bake time)",
                theme.name, base.size, iris_d, iris_d,
            )
            base = base.resize((iris_d, iris_d), Image.LANCZOS)

        sprites: list[Image.Image] = []
        iris_alpha = np.asarray(base.getchannel("A"), dtype=np.uint16)
        c = iris_d / 2.0
        for k in range(PUPIL_STEPS):
            t = k / (PUPIL_STEPS - 1)
            dilation = theme.pupil_min_frac + (theme.pupil_max_frac - theme.pupil_min_frac) * t
            sprite = base.copy()
            if theme.pupil_shape != "none":
                if theme.pupil_shape == "round":
                    half_w = half_h = max(0.5, dilation * iris_d / 2.0)
                else:  # "slit": vertical ellipse, full iris height
                    half_w = max(0.5, theme.slit_width_frac * iris_d * dilation / 2.0)
                    half_h = iris_d / 2.0
                overlay = Image.new("RGBA", (iris_d, iris_d), (0, 0, 0, 0))
                ImageDraw.Draw(overlay).ellipse(
                    (c - half_w, c - half_h, c + half_w, c + half_h),
                    fill=(*theme.pupil_color, 255),
                )
                # Keep the pupil inside the iris art: clip its alpha by the
                # iris alpha (matters for glow-edged irises like "ghost").
                over_alpha = np.asarray(overlay.getchannel("A"), dtype=np.uint16)
                clipped = ((over_alpha * iris_alpha) // 255).astype(np.uint8)
                overlay.putalpha(Image.fromarray(clipped, "L"))
                sprite.alpha_composite(overlay)
            sprites.append(sprite)
        return sprites

    @staticmethod
    def _bake_lid_masks(theme: Theme) -> tuple[list[Image.Image], list[bool]]:
        """LID_STEPS L-mode masks (255 = lid covers) quantizing openness
        0..MAX_OPENNESS. Index 0 is fully closed, the last index fully open.

        The lid edge is modeled as an arc of a large circle ("curved") or a
        horizontal chord ("straight"). The upper lid travels `upper_bias` of
        the closure distance, the lower lid the rest, meeting on full close.
        """
        xs = np.arange(SIZE, dtype=np.float64) - (SIZE - 1) / 2.0
        if theme.lid_style == "curved":
            r = _LID_ARC_RADIUS
            sag = (r - np.sqrt(r * r - xs * xs))[None, :]  # arc drop toward the sides
        else:
            sag = np.zeros((1, SIZE))
        yy = np.arange(SIZE, dtype=np.float64)[:, None]
        bias = _clamp(theme.lid_upper_bias, 0.0, 1.0)

        masks: list[Image.Image] = []
        empty: list[bool] = []
        for i in range(LID_STEPS):
            if i == 0:  # fully closed: exact full cover, no AA seam
                arr = np.full((SIZE, SIZE), 255, dtype=np.uint8)
            elif i == LID_STEPS - 1:  # startled-wide: lids fully retracted
                arr = np.zeros((SIZE, SIZE), dtype=np.uint8)
            else:
                closure = 1.0 - i / (LID_STEPS - 1)
                y_upper = closure * bias * SIZE
                y_lower = SIZE - closure * (1.0 - bias) * SIZE
                upper = np.clip((y_upper + sag) - yy + 0.5, 0.0, 1.0)
                lower = np.clip(yy - (y_lower - sag) + 0.5, 0.0, 1.0)
                arr = (np.clip(upper + lower, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            masks.append(Image.fromarray(arr, "L"))
            empty.append(not arr.any())
        return masks, empty

    @staticmethod
    def _bake_vignette() -> np.ndarray:
        """Q8.8 multiplier (240,240,1) uint32: 256 inside the round face,
        0 outside, antialiased edge."""
        c = (SIZE - 1) / 2.0
        yy, xx = np.mgrid[0:SIZE, 0:SIZE]
        rr = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
        alpha = np.clip(SIZE / 2.0 - rr + 0.5, 0.0, 1.0)
        return (alpha * 256.0 + 0.5).astype(np.uint32)[..., None]

    # ---------------------------------------------------------------- rendering

    def render(self, state: EyeState, t: float = 0.0) -> np.ndarray:
        """Compose one frame; returns a (240, 240, 3) uint8 RGB array.

        ``t`` is the animation clock in seconds (the app passes its running
        frame time); it only affects themes with a spinning iris."""
        gaze_x = _clamp(state.gaze_x, -1.0, 1.0)
        gaze_y = _clamp(state.gaze_y, -1.0, 1.0)
        gpx = gaze_x * self._range
        gpy = -gaze_y * self._range  # screen y grows downward

        # 1. sclera window, offset by -gaze_px * parallax (background lags gaze)
        ox = min(max(self._base_x - round(gpx * self._parallax), 0), self._max_x)
        oy = min(max(self._base_y - round(gpy * self._parallax), 0), self._max_y)
        frame = self._sclera.crop((ox, oy, ox + SIZE, oy + SIZE))

        # 2. iris sprite — rotation step for spinning themes, else the
        #    quantized pupil-dilation step — centered at gaze
        if self._spin_sprites is not None:
            sprite = self._spin_sprites[int(t * self._spin_rate) % SPIN_STEPS]
        else:
            k = round(_clamp(state.pupil, 0.0, 1.0) * (PUPIL_STEPS - 1))
            sprite = self._iris_sprites[k]
        ix = round(self._center + gpx - self._iris_half)
        iy = round(self._center + gpy - self._iris_half)
        frame.paste(sprite, (ix, iy), sprite)

        # 3. highlight: fixed offset from iris center, counter-moving so the
        #    specular stays roughly light-locked
        if self._highlight is not None:
            hx = round(self._center + gpx * (1.0 - _HIGHLIGHT_COUNTER) + self._hl_off[0])
            hy = round(self._center + gpy * (1.0 - _HIGHLIGHT_COUNTER) + self._hl_off[1])
            frame.paste(self._highlight, (hx, hy), self._highlight)

        # 4. eyelids
        openness = _clamp(state.openness, 0.0, MAX_OPENNESS)
        li = round(openness / MAX_OPENNESS * (LID_STEPS - 1))
        if not self._lid_empty[li]:
            frame.paste(self._lid_fill, (0, 0), self._lid_masks[li])

        # 5. vignette + brightness in one integer numpy pass
        factor = self._vig_q8
        brightness = _clamp(state.brightness, 0.0, 1.0)
        if brightness < 1.0:
            factor = (factor * int(round(brightness * 256.0))) >> 8
        out = (np.asarray(frame) * factor) >> 8
        return out.astype(np.uint8)
