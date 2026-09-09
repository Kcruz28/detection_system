"""Abstract interface for multi-object trackers.

This module intentionally contains NO tracking logic. It exists so that
whichever tracker you build (ByteTrack, DeepSORT, or something else) plugs
into the rest of the repo (frontend/app.py, future eval scripts) through one
consistent shape.

See docs/lessons/tracking_bytetrack_deepsort.md for the full lesson and
src/tracking/README.md for the implementation checklist. You write the
subclasses yourself — that's the point of this exercise.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class Detection:
    """One raw detection for a single frame, straight out of YOLO26.

    `bbox` is (x1, y1, x2, y2) in pixel coordinates.
    """

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int


@dataclass
class Track:
    """One tracked object, as returned by a tracker's `update()`.

    `bbox` is (x1, y1, x2, y2) in pixel coordinates. `track_id` must stay
    stable for the same physical object across consecutive frames — that
    persistent identity is the entire point of tracking-by-detection.
    """

    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    state: str = "tentative"  # e.g. "tentative" | "confirmed" | "lost"


class Tracker(ABC):
    """Common interface every tracker wrapper in this repo should implement.

    Implement this for ByteTrack and/or DeepSORT in your own subclasses
    (e.g. `ByteTrackWrapper`, `DeepSortWrapper`) — see src/tracking/README.md.
    """

    @abstractmethod
    def update(self, detections: Sequence[Detection], frame: Any) -> list[Track]:
        """Advance the tracker by one frame.

        Args:
            detections: this frame's raw detector output (post-YOLO26,
                pre-tracking), already converted to `Detection` objects.
            frame: the raw image/array for this frame (BGR np.ndarray from
                OpenCV, matching what frontend/app.py reads). DeepSORT-style
                trackers need pixels to compute a re-ID embedding; ByteTrack-
                style motion-only trackers can ignore it.

        Returns:
            The current list of live tracks (typically confirmed ones),
            each with a stable `track_id` carried over from previous frames
            when it's the same physical object.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Clear all internal state (e.g. when starting a new video)."""
        raise NotImplementedError
