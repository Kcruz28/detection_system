"""
Evaluate a trained YOLO26 checkpoint on the validation (or test) split and
report mAP, precision, recall, and inference speed (FPS).

Usage
-----
    python scripts/evaluate.py
    python scripts/evaluate.py --weights models/best.pt --split test
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = REPO_ROOT / "models" / "best.pt"
DEFAULT_DATA_CONFIG = REPO_ROOT / "configs" / "data.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS),
        help="Path to trained checkpoint (default: models/best.pt).",
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_CONFIG),
        help="Path to YOLO dataset config (default: configs/data.yaml).",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on (default: val).",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", default=0, help="Device (e.g. 0, cpu).")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "outputs" / "eval_metrics.json"),
        help="Where to save the metrics summary as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {weights_path}. Run scripts/train.py first."
        )

    model = YOLO(str(weights_path))

    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )

    # Ultralytics reports per-image speed in milliseconds for preprocess,
    # inference, loss, and postprocess. FPS is derived from inference time.
    inference_ms = metrics.speed.get("inference", 0.0)
    fps = 1000.0 / inference_ms if inference_ms > 0 else float("nan")

    summary = {
        "split": args.split,
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "inference_ms_per_image": inference_ms,
        "fps": fps,
    }

    print("\n=== Evaluation results ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMetrics saved to: {output_path}")


if __name__ == "__main__":
    main()
