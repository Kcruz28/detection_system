# Real-Time Drone Detection & Tracking System

## Overview

## Setup

## Dataset

Dataset comes from [Roboflow Universe](https://universe.roboflow.com) — search
"drone detection" or "UAV detection" for a pre-labeled dataset.

1. Get a free API key from https://app.roboflow.com/settings/api and export it:
   ```bash
   export ROBOFLOW_API_KEY="your_key_here"
   ```
2. Pick a dataset on Roboflow Universe and note its workspace/project/version
   (from the dataset URL), then download it in YOLOv8-compatible format (still correct for YOLO26) into `data/raw/`:
   ```bash
   python scripts/download_dataset.py --workspace <ws> --project <project> --version <n>
   ```
3. Roboflow's export includes its own `data.yaml`. Update `configs/data.yaml`
   (paths + class names) to match it — that file is used by both training and
   evaluation.

## Training

Hyperparameters (model size, epochs, batch, image size, etc.) live in
`configs/train_config.yaml` so they're decoupled from code.

```bash
python scripts/train.py                                   # use config as-is
python scripts/train.py --epochs 50 --batch 32 --model yolo26s.pt   # CLI overrides
```

Training logs, plots, and checkpoints land under `outputs/train/`; the best
checkpoint is also copied to `models/best.pt`.

To evaluate a trained checkpoint (mAP, precision, recall, inference FPS):

```bash
python scripts/evaluate.py --weights models/best.pt --split val
```

Results print to the console and save to `outputs/eval_metrics.json`.

## Demo

A Streamlit app (`frontend/app.py`) provides an interactive way to run the trained
detector without touching code.

**Prerequisite:** a trained checkpoint. Run `scripts/train.py` first — it should
produce a weights file (e.g. `models/best.pt`). If no checkpoint exists yet, the
app will tell you to train one instead of erroring out.

**Run it:**

```bash
pip install -r requirements.txt
streamlit run frontend/app.py
```

**Features:**
- Sidebar controls: model checkpoint path (defaults to `models/best.pt`), a
  confidence threshold slider, and an input-type selector.
- Image upload, webcam snapshot (`st.camera_input`), or video file upload.
- Annotated output (bounding boxes, class labels, confidence scores) via
  Ultralytics' `results[0].plot()`.
- Inference stats: per-frame inference time / FPS and detection counts/confidences.
- For video: frames are processed with OpenCV, written to an annotated `.mp4`
  in `outputs/`, then shown in-app with a download button.

Note: this demo covers detection only. Multi-object tracking (ByteTrack/DeepSORT)
is a separate, hand-built step and is not included here.

## Tracking (Learning Path)

Multi-object tracking (ByteTrack/DeepSORT) is a **self-directed learning
module**, not yet implemented. The goal is to actually learn IoU
association, the Hungarian algorithm, Kalman filter motion prediction, and
track lifecycle management by building it, not to drop in a finished
library call.

- Full lesson (concepts + setup + wiring walkthrough):
  [`docs/lessons/tracking_bytetrack_deepsort.md`](docs/lessons/tracking_bytetrack_deepsort.md)
- Scaffold: `src/tracking/base_tracker.py` (abstract `Tracker` interface,
  `Detection`/`Track` dataclasses — no tracking logic implemented)
- Implementation checklist: [`src/tracking/README.md`](src/tracking/README.md)

`frontend/app.py` and `scripts/` are unaffected until a tracker is built
and wired in deliberately.

## Results
