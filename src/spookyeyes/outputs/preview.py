"""Pygame preview window for development machines.

Shows both eyes side by side, scaled up. Closing the window or pressing ESC
pushes ``Event("quit")`` into the app's event queue (if one was provided);
the app then shuts down cleanly. pygame is imported lazily so the rest of the
package has no hard dependency on it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from spookyeyes.model import SIZE, Event

if TYPE_CHECKING:
    import queue

log = logging.getLogger("spookyeyes.outputs.preview")


class PreviewOutput:
    GAP = 8  # px between the two eyes, before scaling

    def __init__(self, scale: int = 2, events: queue.Queue[Event] | None = None) -> None:
        import pygame  # optional dependency: install spookyeyes[preview]

        self._pygame = pygame
        self._events = events
        self._scale = max(1, int(scale))
        self._closed = False
        pygame.init()
        width = (2 * SIZE + self.GAP) * self._scale
        height = SIZE * self._scale
        self._screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("spookyeyes")

    def _push_quit(self) -> None:
        log.info("preview window closed / ESC pressed")
        if self._events is not None:
            self._events.put(Event("quit"))

    def show(self, left: np.ndarray, right: np.ndarray) -> None:
        if self._closed:
            return
        pg = self._pygame
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                self._push_quit()
            elif ev.type == pg.KEYDOWN and ev.key == pg.K_ESCAPE:
                self._push_quit()
        s = self._scale
        self._screen.fill((0, 0, 0))
        for frame, x in ((left, 0), (right, (SIZE + self.GAP) * s)):
            # pygame surfaces are (width, height); frames are (row, col, rgb).
            surf = pg.surfarray.make_surface(
                np.ascontiguousarray(np.transpose(frame, (1, 0, 2)))
            )
            if s != 1:
                surf = pg.transform.scale(surf, (SIZE * s, SIZE * s))
            self._screen.blit(surf, (x, 0))
        pg.display.flip()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pygame.display.quit()
        self._pygame.quit()
