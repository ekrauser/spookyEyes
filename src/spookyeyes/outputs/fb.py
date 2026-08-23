"""Linux framebuffer output: two GC9A01A panels exposed as fbdev devices.

Frames are converted to RGB565 little-endian (the panel/driver byte order) and
written via mmap into the top-left 240x240 of each device, honouring the
device's line stride when it is wider than the panel.
"""

from __future__ import annotations

import logging
import mmap
import os
import re
import stat
from pathlib import Path

import numpy as np

from spookyeyes.model import SIZE

log = logging.getLogger("spookyeyes.outputs.fb")

_BYTES_PER_PX = 2  # RGB565
_ROW_BYTES = SIZE * _BYTES_PER_PX


def to_rgb565(frame: np.ndarray) -> np.ndarray:
    """Pack an ``(H, W, 3)`` uint8 RGB frame into RGB565.

    Returns an ``(H, W)`` array of dtype ``'<u2'`` (explicit little-endian, so
    ``.tobytes()`` is byte-exact for the framebuffer regardless of host
    endianness). Fully vectorized.
    """
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB array, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        raise ValueError(f"expected a uint8 RGB frame, got dtype {arr.dtype}")
    a16 = arr.astype(np.uint16)
    packed = (
        ((a16[..., 0] & 0xF8) << 8)  # R: top 5 bits -> 15..11
        | ((a16[..., 1] & 0xFC) << 3)  # G: top 6 bits -> 10..5
        | (a16[..., 2] >> 3)  # B: top 5 bits -> 4..0
    )
    return packed.astype("<u2")


def resolve_fb(spec: str, sysfs_root: str | Path = "/sys/class/graphics") -> str:
    """Resolve a framebuffer spec to a ``/dev/fbN`` path.

    fbdev minor numbers follow probe order, which is asynchronous and can swap
    between boots (observed in practice: spi0.1 probing before spi0.0). Config
    should therefore name each panel by its stable SPI bus address:

    - ``"spi0.0"``-style specs are resolved via sysfs to the fb device whose
      ``device`` link points at that SPI address;
    - anything else (e.g. ``"/dev/fb1"``, tmp files in tests) passes through.
    """
    if not re.fullmatch(r"spi\d+\.\d+", spec):
        return spec
    root = Path(sysfs_root)
    try:
        entries = sorted(root.iterdir())
    except OSError as e:
        raise ValueError(
            f"cannot resolve {spec!r}: sysfs unavailable at {root}: {e}"
        ) from e
    for entry in entries:
        if not re.fullmatch(r"fb\d+", entry.name):
            continue
        target = os.path.realpath(entry / "device")
        if os.path.basename(target) == spec:
            return f"/dev/{entry.name}"
    raise ValueError(
        f"no framebuffer found for SPI device {spec!r} under {root} — is the "
        f"panel-mipi-dbi driver bound? (check: dmesg | grep -i mipi)"
    )


def _detect_stride_px(path: str) -> int | None:
    """Best-effort stride autodetection for real fbdev devices.

    The kernel exposes the line length (in bytes) at
    ``/sys/class/graphics/fbN/stride``. Returns the stride in pixels, or None
    if the path is not an fbdev node / sysfs is unavailable.
    """
    name = os.path.basename(path)
    if not re.fullmatch(r"fb\d+", name):
        return None
    sysfs = Path("/sys/class/graphics") / name / "stride"
    try:
        stride_bytes = int(sysfs.read_text().strip())
    except (OSError, ValueError):
        return None
    if stride_bytes <= 0:
        return None
    return stride_bytes // _BYTES_PER_PX


class FramebufferOutput:
    """Writes both eyes to two fbdev devices (or regular files, for tests).

    ``paths`` is ``(left, right)``. ``stride_px`` is the device line stride in
    pixels: an int (applied to both), a per-device ``(left, right)`` tuple, or
    None (the default) to autodetect from ``/sys/class/graphics/fbN/stride``
    where available, falling back to the panel width of 240. If the device's
    virtual size is larger than 240x240, frames land stride-aware in the
    top-left corner.
    """

    def __init__(
        self,
        paths: tuple[str, str],
        stride_px: int | tuple[int, int] | None = None,
    ) -> None:
        if len(paths) != 2:
            raise ValueError(f"expected (left, right) framebuffer paths, got {paths!r}")
        if stride_px is None or isinstance(stride_px, int):
            strides: tuple[int | None, int | None] = (stride_px, stride_px)
        else:
            if len(stride_px) != 2:
                raise ValueError(
                    f"stride_px must be an int or a (left, right) pair, got {stride_px!r}"
                )
            strides = (stride_px[0], stride_px[1])

        self._paths = (resolve_fb(str(paths[0])), resolve_fb(str(paths[1])))
        if self._paths != (str(paths[0]), str(paths[1])):
            log.info(
                "resolved panels: left %s -> %s, right %s -> %s",
                paths[0], self._paths[0], paths[1], self._paths[1],
            )
        self._fds: list[int] = []
        self._maps: list[mmap.mmap] = []
        self._targets: list[np.ndarray] = []
        try:
            for path, sp in zip(self._paths, strides):
                self._open_device(path, sp)
        except Exception:
            self.close()
            raise

    def _open_device(self, path: str, stride_px: int | None) -> None:
        if stride_px is None:
            stride_px = _detect_stride_px(path) or SIZE
        if stride_px < SIZE:
            raise ValueError(
                f"framebuffer {path}: line stride {stride_px}px is narrower than the "
                f"{SIZE}px panel — this device cannot hold a full eye frame"
            )
        stride_bytes = stride_px * _BYTES_PER_PX
        length = stride_bytes * SIZE

        fd = os.open(path, os.O_RDWR)
        self._fds.append(fd)
        st = os.fstat(fd)
        if stat.S_ISREG(st.st_mode) and st.st_size < length:
            raise ValueError(
                f"framebuffer {path} is too small: {st.st_size} bytes, but "
                f"{length} bytes are needed ({SIZE} rows x {stride_bytes}-byte "
                f"stride, RGB565). Check that the display driver is bound and "
                f"the stride ({stride_px}px) matches the device."
            )
        try:
            mm = mmap.mmap(fd, length)
        except (ValueError, OSError) as e:
            raise ValueError(
                f"cannot mmap {length} bytes of framebuffer {path}: {e}. The "
                f"device is smaller than a {SIZE}x{SIZE} RGB565 frame at a "
                f"{stride_px}px stride."
            ) from e
        self._maps.append(mm)
        # A stride-wide '<u2' view over the first SIZE rows; assignment into
        # the left SIZE columns is the stride-aware top-left write.
        view = np.frombuffer(mm, dtype="<u2", count=stride_px * SIZE)
        self._targets.append(view.reshape(SIZE, stride_px)[:, :SIZE])
        log.debug("mapped %s: stride %dpx, %d bytes", path, stride_px, length)

    def show(self, left: np.ndarray, right: np.ndarray) -> None:
        if not self._targets:
            raise RuntimeError("FramebufferOutput is closed")
        for frame, target, path in zip((left, right), self._targets, self._paths):
            arr = np.asarray(frame)
            if arr.shape != (SIZE, SIZE, 3):
                raise ValueError(
                    f"frame for {path} has shape {arr.shape}, expected ({SIZE}, {SIZE}, 3)"
                )
            target[:] = to_rgb565(arr)

    def close(self) -> None:
        # Drop the numpy views first: they hold buffer exports on the mmaps,
        # and mmap.close() raises BufferError while exports exist.
        self._targets = []
        for mm in self._maps:
            try:
                mm.flush()
            except (OSError, ValueError):
                pass
            try:
                mm.close()
            except BufferError:  # a caller kept a view alive; leak, don't crash
                log.warning("framebuffer mmap still referenced at close; not unmapped")
        self._maps = []
        for fd in self._fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds = []
