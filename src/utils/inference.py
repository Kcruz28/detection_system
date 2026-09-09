"""
Shared inference helpers for the drone detection demo.

Keeps the model-loading / single-image-inference logic in one place so it can
be reused by the Streamlit frontend (frontend/app.py) and, eventually, other
entry points (e.g. a CLI or the tracking pipeline) without duplicating code.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class InferenceResult:
    """Container for a single-image inference call."""

    annotated_image: np.ndarray  # BGR numpy array, ready for display/writing
    inference_time_ms: float
    num_detections: int
    confidences: list[float] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    raw: Any = None  # the underlying ultralytics Results object


def load_model(model_path: str):
    """Load a YOLO model from disk.

    Raises FileNotFoundError with a clear message if the checkpoint doesn't
    exist yet (e.g. training hasn't been run), and ImportError if ultralytics
    isn't installed.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model found at '{model_path}'. "
            "Run scripts/train.py first to produce a checkpoint."
        )

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(
            "ultralytics is not installed. Run `pip install -r requirements.txt`."
        ) from e

    return YOLO(str(path))


def run_inference_on_image(model, image: np.ndarray, conf: float = 0.25) -> InferenceResult:
    """Run YOLO inference on a single BGR image (numpy array).

    Returns an InferenceResult with the annotated frame and basic stats.
    """
    start = time.perf_counter()
    results = model(image, conf=conf, verbose=False)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    result = results[0]
    annotated = result.plot()  # BGR numpy array with boxes/labels drawn

    confidences: list[float] = []
    class_names: list[str] = []
    if result.boxes is not None and len(result.boxes) > 0:
        confidences = [float(c) for c in result.boxes.conf.tolist()]
        names = result.names
        class_names = [names[int(c)] for c in result.boxes.cls.tolist()]

    return InferenceResult(
        annotated_image=annotated,
        inference_time_ms=elapsed_ms,
        num_detections=len(confidences),
        confidences=confidences,
        class_names=class_names,
        raw=result,
    )
