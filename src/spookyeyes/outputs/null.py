"""No-op output for benchmarks and headless tests."""

from __future__ import annotations

import numpy as np


class NullOutput:
    def show(self, left: np.ndarray, right: np.ndarray) -> None:
        pass

    def close(self) -> None:
        pass
