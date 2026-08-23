"""Record output: saves side-by-side composite PNG frames and an optional GIF."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from spookyeyes.model import SIZE

log = logging.getLogger("spookyeyes.outputs.record")

_GIF_FPS = 15


class RecordOutput:
    """Saves each shown frame pair as ``frame_00000.png``, ``frame_00001.png``…

    Each PNG is a side-by-side composite (left eye, ``GAP`` px black gap,
    right eye). ``every`` keeps only every Nth shown frame; ``max_frames``
    stops saving after that many files (later ``show`` calls are no-ops, not
    errors). If ``gif_path`` is given, ``close()`` additionally assembles the
    saved frames into an animated GIF at ~15 fps playback.
    """

    GAP = 8  # px between the two eyes in the composite

    def __init__(
        self,
        dir: str | Path,
        gif_path: str | Path | None = None,
        max_frames: int | None = None,
        every: int = 1,
    ) -> None:
        if every < 1:
            raise ValueError(f"every must be >= 1, got {every}")
        if max_frames is not None and max_frames < 0:
            raise ValueError(f"max_frames must be >= 0, got {max_frames}")
        self._dir = Path(dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._gif_path = Path(gif_path) if gif_path is not None else None
        self._max_frames = max_frames
        self._every = every
        self._shown = 0
        self._saved: list[Path] = []
        self._closed = False

    @property
    def frames_saved(self) -> int:
        return len(self._saved)

    def show(self, left: np.ndarray, right: np.ndarray) -> None:
        idx = self._shown
        self._shown += 1
        if idx % self._every != 0:
            return
        if self._max_frames is not None and len(self._saved) >= self._max_frames:
            return
        canvas = np.zeros((SIZE, 2 * SIZE + self.GAP, 3), dtype=np.uint8)
        canvas[:, :SIZE] = left
        canvas[:, SIZE + self.GAP :] = right
        path = self._dir / f"frame_{len(self._saved):05d}.png"
        Image.fromarray(canvas).save(path)
        self._saved.append(path)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        log.info("recorded %d frames to %s", len(self._saved), self._dir)
        if self._gif_path is None or not self._saved:
            return
        self._gif_path.parent.mkdir(parents=True, exist_ok=True)
        # Decode each PNG into memory and close its file immediately — holding
        # every file open at once exhausts the fd limit on long recordings.
        frames: list[Image.Image] = []
        for p in self._saved:
            with Image.open(p) as im:
                frames.append(im.copy())
        frames[0].save(
            self._gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=round(1000 / _GIF_FPS),
            loop=0,
        )
        log.info("wrote %s (%d frames, ~%d fps)", self._gif_path, len(frames), _GIF_FPS)
