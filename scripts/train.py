"""
Fine-tune a YOLO26 model on the drone-detection dataset.

Hyperparameters live in configs/train_config.yaml so runs are reproducible
without touching code. Common ones can be overridden from the CLI.

Usage
-----
    python scripts/train.py
    python scripts/train.py --epochs 50 --batch 32 --model yolo26s.pt
"""

import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "train_config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to train_config.yaml (default: configs/train_config.yaml).",
    )
    parser.add_argument("--epochs", type=int, help="Override epochs.")
    parser.add_argument("--batch", type=int, help="Override batch size.")
    parser.add_argument("--imgsz", type=int, help="Override image size.")
    parser.add_argument(
        "--model", dest="model_name", help="Override base model (e.g. yolo26s.pt)."
    )
    parser.add_argument("--device", help="Override device (e.g. 0, cpu).")
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    overrides = {
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "model": args.model_name,
        "device": args.device,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    config = apply_overrides(config, args)

    data_path = (REPO_ROOT / config["data"]).resolve()
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset config not found at {data_path}. Run "
            "scripts/download_dataset.py first, and make sure configs/data.yaml "
            "points at your actual dataset."
        )

    model = YOLO(config["model"])

    results = model.train(
        data=str(data_path),
        epochs=config.get("epochs", 100),
        imgsz=config.get("imgsz", 640),
        batch=config.get("batch", 16),
        patience=config.get("patience", 20),
        device=config.get("device", 0),
        workers=config.get("workers", 8),
        project=config.get("project", "outputs/train"),
        name=config.get("name", "drone_yolo26"),
        optimizer=config.get("optimizer", "auto"),
        lr0=config.get("lr0", 0.01),
        seed=config.get("seed", 0),
    )

    # Report the headline metrics tracked in vision.md step 2.
    metrics = model.trainer.validator.metrics
    print("\n=== Training complete ===")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

    # Copy the best checkpoint out of the Ultralytics run dir into models/.
    run_dir = Path(results.save_dir)
    best_weights = run_dir / "weights" / "best.pt"
    export_path = REPO_ROOT / config.get("export_weights_path", "models/best.pt")
    export_path.parent.mkdir(parents=True, exist_ok=True)

    if best_weights.exists():
        shutil.copy2(best_weights, export_path)
        print(f"Best weights copied to: {export_path}")
    else:
        print(f"WARNING: expected best weights at {best_weights}, not found.")


if __name__ == "__main__":
    main()
