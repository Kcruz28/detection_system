"""
Streamlit demo for the Real-Time Drone Detection system.

Lets you upload an image or video (or snap a webcam photo) and run YOLO26
inference against a trained checkpoint, with an adjustable confidence
threshold and basic performance stats (inference time / FPS, detection
counts and confidences).

Run with:
    streamlit run frontend/app.py

Requires a trained checkpoint (default: models/best.pt) produced by
scripts/train.py. Tracking (ByteTrack/DeepSORT) is intentionally NOT
implemented here.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# Make `src` importable when run via `streamlit run frontend/app.py`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.inference import load_model, run_inference_on_image  # noqa: E402

OUTPUTS_DIR = REPO_ROOT / "outputs"
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "best.pt"


st.set_page_config(page_title="Drone Detection Demo", layout="wide")


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _cached_load_model(model_path: str, mtime: float):
    """Cache keyed on path + mtime so a re-trained checkpoint invalidates it."""
    return load_model(model_path)


def get_model(model_path: str):
    path = Path(model_path)
    if not path.exists():
        return None, FileNotFoundError(
            f"No trained model found at '{model_path}'. "
            "Run scripts/train.py first to produce a checkpoint."
        )
    try:
        model = _cached_load_model(str(path), path.stat().st_mtime)
        return model, None
    except Exception as e:  # noqa: BLE001
        return None, e


# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------
st.sidebar.title("Controls")

model_path = st.sidebar.text_input(
    "Model checkpoint path",
    value=str(DEFAULT_MODEL_PATH),
    help="Path to a YOLO26 .pt checkpoint. Defaults to models/best.pt.",
)

conf_threshold = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.01,
)

input_type = st.sidebar.radio(
    "Input type",
    options=["Image upload", "Webcam photo", "Video upload"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Tracking (ByteTrack/DeepSORT) is a separate step and not part of this demo."
)


# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------
st.title("Real-Time Drone Detection — Demo")

model, load_err = get_model(model_path)

if load_err is not None:
    if isinstance(load_err, FileNotFoundError):
        st.warning(str(load_err))
    else:
        st.error(f"Failed to load model: {load_err}")
    st.stop()

st.success(f"Loaded model: `{model_path}`")


def _show_detection_stats(result) -> None:
    col1, col2, col3 = st.columns(3)
    fps = 1000.0 / result.inference_time_ms if result.inference_time_ms > 0 else 0.0
    col1.metric("Inference time", f"{result.inference_time_ms:.1f} ms")
    col2.metric("FPS", f"{fps:.1f}")
    col3.metric("Detections", result.num_detections)

    if result.num_detections > 0:
        st.subheader("Detections")
        rows = [
            {"class": cls, "confidence": f"{conf:.2f}"}
            for cls, conf in zip(result.class_names, result.confidences)
        ]
        st.table(rows)
    else:
        st.info("No detections above the current confidence threshold.")


# ---- Image upload ----
if input_type == "Image upload":
    uploaded = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )
    if uploaded is not None:
        file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image_bgr is None:
            st.error("Could not decode the uploaded image.")
        else:
            result = run_inference_on_image(model, image_bgr, conf=conf_threshold)

            col_left, col_right = st.columns(2)
            with col_left:
                st.subheader("Input")
                st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col_right:
                st.subheader("Detections")
                st.image(
                    cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                )

            _show_detection_stats(result)


# ---- Webcam photo ----
elif input_type == "Webcam photo":
    snapshot = st.camera_input("Take a photo")
    if snapshot is not None:
        file_bytes = np.frombuffer(snapshot.getvalue(), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image_bgr is None:
            st.error("Could not decode the webcam image.")
        else:
            result = run_inference_on_image(model, image_bgr, conf=conf_threshold)
            st.subheader("Detections")
            st.image(
                cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )
            _show_detection_stats(result)


# ---- Video upload ----
else:
    uploaded_video = st.file_uploader(
        "Upload a video", type=["mp4", "avi", "mov", "mkv"]
    )
    if uploaded_video is not None:
        # Write the upload to a temp file so OpenCV can open it by path.
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix) as tmp_in:
            tmp_in.write(uploaded_video.read())
            input_path = tmp_in.name

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            st.error("Could not open the uploaded video.")
        else:
            fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            output_path = OUTPUTS_DIR / f"annotated_{Path(uploaded_video.name).stem}.mp4"

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps_in, (width, height))

            progress_bar = st.progress(0.0, text="Processing video...")

            total_time_ms = 0.0
            total_detections = 0
            frame_idx = 0

            start_all = time.perf_counter()
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                result = run_inference_on_image(model, frame, conf=conf_threshold)
                writer.write(result.annotated_image)

                total_time_ms += result.inference_time_ms
                total_detections += result.num_detections
                frame_idx += 1

                if total_frames > 0:
                    progress_bar.progress(
                        min(frame_idx / total_frames, 1.0),
                        text=f"Processing video... frame {frame_idx}/{total_frames}",
                    )

            cap.release()
            writer.release()
            wall_time_s = time.perf_counter() - start_all
            progress_bar.progress(1.0, text="Done")

            st.subheader("Annotated video")
            st.video(str(output_path))

            with open(output_path, "rb") as f:
                st.download_button(
                    "Download annotated video",
                    data=f.read(),
                    file_name=output_path.name,
                    mime="video/mp4",
                )

            avg_ms = total_time_ms / frame_idx if frame_idx else 0.0
            avg_fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
            overall_fps = frame_idx / wall_time_s if wall_time_s > 0 else 0.0

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Frames processed", frame_idx)
            col2.metric("Avg inference/frame", f"{avg_ms:.1f} ms")
            col3.metric("Avg inference FPS", f"{avg_fps:.1f}")
            col4.metric("Overall throughput", f"{overall_fps:.1f} FPS")
            st.metric("Total detections (all frames)", total_detections)
