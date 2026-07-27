"""Frontend interface: a video frame in, an optional wrist pose out."""

from __future__ import annotations

import abc

import numpy as np

from ..types import WristObservation


class HandFrontend(abc.ABC):
    """Per-frame hand/wrist pose estimator.

    Implementations return `None` for a frame with no confident detection. They
    must never raise on a frame they simply cannot interpret — occlusion and
    subjects leaving the frame are expected inputs, not errors.
    """

    name: str = "abstract"

    @abc.abstractmethod
    def process(self, frame: np.ndarray, frame_index: int) -> WristObservation | None:
        """Estimate the wrist pose in the *camera* frame for one BGR frame."""

    def close(self) -> None:
        """Release any held resources. Safe to call more than once."""

    def __enter__(self) -> HandFrontend:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
