"""Output backends: where rendered eye frames go.

`Output` is the protocol every backend implements; `make_output` builds the
backend named by ``DisplayConfig.output``. Backends with optional dependencies
(pygame for preview) are imported lazily so the core package works with only
numpy + pillow installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import queue
    from pathlib import Path

    import numpy as np

    from spookyeyes.config import DisplayConfig
    from spookyeyes.model import Event

__all__ = ["Output", "make_output"]


@runtime_checkable
class Output(Protocol):
    """A sink for rendered eye frames.

    ``show`` receives two ``(240, 240, 3)`` uint8 RGB arrays (left and right
    eye, already mirrored by the app if configured). ``close`` releases any
    resources and must be safe to call more than once.
    """

    def show(self, left: np.ndarray, right: np.ndarray) -> None: ...

    def close(self) -> None: ...


def make_output(
    cfg: DisplayConfig,
    *,
    events: queue.Queue[Event] | None = None,
    gif_path: str | Path | None = None,
) -> Output:
    """Build the output backend selected by ``cfg.output``.

    ``events`` is handed to interactive backends (preview) so they can push
    ``Event("quit")`` on window close / ESC. ``gif_path`` is forwarded to the
    record backend. Both are ignored by backends that do not use them.

    Raises ``ValueError`` for an unknown output name; backend constructors may
    raise their own errors (e.g. missing pygame, unusable framebuffer device).
    """
    name = cfg.output
    if name == "null":
        from spookyeyes.outputs.null import NullOutput

        return NullOutput()
    if name == "fb":
        from spookyeyes.outputs.fb import FramebufferOutput

        return FramebufferOutput((cfg.fb_left, cfg.fb_right))
    if name == "record":
        from spookyeyes.outputs.record import RecordOutput

        return RecordOutput(cfg.record_dir, gif_path=gif_path)
    if name == "preview":
        # pygame is an optional dependency: import only when actually needed.
        from spookyeyes.outputs.preview import PreviewOutput

        return PreviewOutput(scale=cfg.scale, events=events)
    raise ValueError(
        f"unknown display output {name!r}; expected one of: preview, fb, record, null"
    )
