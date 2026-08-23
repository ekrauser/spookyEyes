#!/usr/bin/env python3
"""Framebuffer test pattern for the spookyeyes dual-GC9A01 setup.

Writes a distinct RGB565 pattern to each panel so you can verify that both
displays work AND that left/right are not swapped:

  left  (CE0 = spi0.0): red/white bullseye with a white "L"
  right (CE1 = spi0.1): blue/white bullseye with a white "R"

Panels are addressed by stable SPI address and resolved to /dev/fbN via sysfs
— fbdev minor numbers follow async probe order and can swap between boots.
Plain /dev/fbN paths are also accepted.

Standalone on purpose — needs only numpy (``sudo apt install python3-numpy``
or any venv). Reads /sys/class/graphics/fbN/virtual_size and .../stride when
available so writes are stride-aware; falls back to 240x240 / width*2.

Usage:
    python3 pi/test_pattern.py                        # spi0.0 + spi0.1
    python3 pi/test_pattern.py --left /dev/fb2 --right /dev/fb1
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

SIZE = 240

# 5x7 glyphs, scaled up at draw time.
GLYPHS = {
    "L": [
        "10000",
        "10000",
        "10000",
        "10000",
        "10000",
        "10000",
        "11111",
    ],
    "R": [
        "11110",
        "10001",
        "10001",
        "11110",
        "10100",
        "10010",
        "10001",
    ],
}


def build_pattern(color: tuple[int, int, int], letter: str) -> np.ndarray:
    """Bullseye of alternating <color>/white rings + a big centered letter.

    Returns (SIZE, SIZE, 3) uint8, RGB order.
    """
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    r = np.hypot(xx - SIZE / 2 + 0.5, yy - SIZE / 2 + 0.5)

    frame = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    ring = (np.floor_divide(r, 30).astype(int) % 2) == 0
    frame[ring] = color
    frame[~ring] = (255, 255, 255)
    frame[r < 45] = color  # solid center disc under the letter

    glyph = GLYPHS[letter]
    scale = 10  # 5x7 -> 50x70 px
    gw, gh = len(glyph[0]) * scale, len(glyph) * scale
    x0, y0 = (SIZE - gw) // 2, (SIZE - gh) // 2
    for gy, row in enumerate(glyph):
        for gx, bit in enumerate(row):
            if bit == "1":
                frame[
                    y0 + gy * scale : y0 + (gy + 1) * scale,
                    x0 + gx * scale : x0 + (gx + 1) * scale,
                ] = (255, 255, 255)
    return frame


def to_rgb565(frame: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 RGB -> (H, W) uint16 little-endian RGB565."""
    r = (frame[:, :, 0].astype(np.uint16) >> 3) << 11
    g = (frame[:, :, 1].astype(np.uint16) >> 2) << 5
    b = frame[:, :, 2].astype(np.uint16) >> 3
    return (r | g | b).astype("<u2")


def resolve_fb(spec: str) -> str:
    """Resolve "spi0.0"-style specs to /dev/fbN via sysfs; pass paths through."""
    if not re.fullmatch(r"spi\d+\.\d+", spec):
        return spec
    root = "/sys/class/graphics"
    for name in sorted(os.listdir(root)):
        if not re.fullmatch(r"fb\d+", name):
            continue
        target = os.path.realpath(os.path.join(root, name, "device"))
        if os.path.basename(target) == spec:
            return f"/dev/{name}"
    raise FileNotFoundError(
        f"no framebuffer bound to SPI device {spec} (driver not probed?)"
    )


def fb_geometry(path: str) -> tuple[int, int, int]:
    """Return (width, height, stride_bytes) for a fbdev node.

    Reads sysfs when available; otherwise assumes 240x240, stride=width*2.
    """
    width, height = SIZE, SIZE
    name = os.path.basename(path)
    sysdir = f"/sys/class/graphics/{name}"
    try:
        raw = open(os.path.join(sysdir, "virtual_size")).read()
        w, h = raw.strip().split(",")
        width, height = int(w), int(h)
    except (OSError, ValueError):
        print(f"  note: {sysdir}/virtual_size unreadable, assuming {SIZE}x{SIZE}")
    stride = width * 2
    try:
        stride = int(open(os.path.join(sysdir, "stride")).read().strip())
    except (OSError, ValueError):
        print(f"  note: {sysdir}/stride unreadable, assuming {stride}")
    return width, height, stride


def write_fb(path: str, pattern: np.ndarray) -> None:
    """Write the pattern into the top-left of the framebuffer, stride-aware."""
    width, height, stride = fb_geometry(path)
    pix = to_rgb565(pattern)
    w = min(width, pix.shape[1])
    h = min(height, pix.shape[0])
    if (w, h) != (pix.shape[1], pix.shape[0]):
        print(f"  warning: {path} is only {width}x{height}, cropping pattern")
    with open(path, "rb+", buffering=0) as f:
        if w == pix.shape[1] and stride == w * 2 and h == pix.shape[0]:
            f.write(pix.tobytes())  # contiguous fast path
        else:
            for y in range(h):
                f.seek(y * stride)
                f.write(pix[y, :w].tobytes())
    print(f"  wrote {w}x{h} RGB565 to {path} (stride {stride})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--left", default="spi0.0",
                    help="left eye: SPI address or fbdev path (default spi0.0)")
    ap.add_argument("--right", default="spi0.1",
                    help="right eye: SPI address or fbdev path (default spi0.1)")
    args = ap.parse_args(argv)

    jobs = [
        (args.left, (220, 20, 20), "L"),
        (args.right, (20, 40, 220), "R"),
    ]
    failures = 0
    for spec, color, letter in jobs:
        try:
            path = resolve_fb(spec)
        except FileNotFoundError as exc:
            failures += 1
            print(f"{letter}: {spec}\n  ERROR: {exc}\n"
                  "  Check: dmesg | grep -i mipi")
            continue
        print(f"{letter}: {spec} -> {path}" if path != spec else f"{letter}: {path}")
        try:
            write_fb(path, build_pattern(color, letter))
        except FileNotFoundError:
            failures += 1
            print(
                f"  ERROR: {path} does not exist.\n"
                "  The panel-mipi-dbi driver did not create this framebuffer. Check:\n"
                "    dmesg | grep -iE 'mipi|gc9a01'   (driver + firmware load messages)\n"
                "    ls -la /lib/firmware/gc9a01.bin  (run pi/build_firmware.sh if missing)\n"
                "    /boot/firmware/config.txt has both mipi-dbi-spi stanzas "
                "(pi/config.txt.snippet)\n"
                "    and no stray `dtparam=spi=on`; then reboot."
            )
        except PermissionError:
            failures += 1
            print(
                f"  ERROR: no permission to write {path}.\n"
                f"  Add yourself to the video group (`sudo usermod -aG video $USER`,\n"
                "  then re-login) or run with sudo."
            )
        except OSError as exc:
            failures += 1
            print(f"  ERROR: writing {path} failed: {exc}")
    if failures:
        print(f"{failures} of {len(jobs)} framebuffers failed")
    else:
        print("Both patterns written. If L/R appear swapped, swap fb_left/fb_right "
              "in config.toml (or your wiring of CE0/CE1).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
