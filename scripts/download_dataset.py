"""
Download a drone-detection dataset from Roboflow Universe in YOLOv8 format.

Setup
-----
1. Create a free account at https://universe.roboflow.com and grab an API key
   from https://app.roboflow.com/settings/api.
2. Export it as an environment variable:
       export ROBOFLOW_API_KEY="your_key_here"
3. Search Roboflow Universe for "drone detection" or "UAV detection" to find
   a labeled dataset you like, and note its workspace slug, project slug,
   and version number (visible in the dataset URL and on its "Download"
   dialog, e.g. https://universe.roboflow.com/<workspace>/<project>/dataset/<version>).
4. Pass those in as CLI args (or edit the DEFAULT_* placeholders below).

Usage
-----
    python scripts/download_dataset.py \
        --workspace some-workspace \
        --project drone-detection-xyz \
        --version 1
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = REPO_ROOT / "data" / "raw"

# Placeholder defaults -- replace with a real dataset you've picked out on
# Roboflow Universe (search "drone detection" / "UAV detection").
DEFAULT_WORKSPACE = "your-roboflow-workspace"
DEFAULT_PROJECT = "drone-detection-dataset"
DEFAULT_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        default=DEFAULT_WORKSPACE,
        help="Roboflow workspace slug (from the dataset URL).",
    )
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help="Roboflow project slug (from the dataset URL).",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=DEFAULT_VERSION,
        help="Dataset version number to download.",
    )
    parser.add_argument(
        "--format",
        default="yolov8",
        help="Export format (default: yolov8).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DATA_RAW_DIR),
        help="Directory to download the dataset into (default: data/raw).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit(
            "ROBOFLOW_API_KEY environment variable is not set.\n"
            "Get a key from https://app.roboflow.com/settings/api and run:\n"
            '    export ROBOFLOW_API_KEY="your_key_here"'
        )

    if args.workspace == DEFAULT_WORKSPACE or args.project == DEFAULT_PROJECT:
        print(
            "WARNING: using placeholder workspace/project values. "
            "Pick a real dataset from https://universe.roboflow.com "
            "(search 'drone detection' or 'UAV detection') and pass "
            "--workspace/--project/--version, or edit the DEFAULT_* "
            "constants in this script.",
            file=sys.stderr,
        )

    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit(
            "The 'roboflow' package is not installed. Run:\n"
            "    pip install -r requirements.txt"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)
    dataset = project.version(args.version).download(
        args.format, location=str(output_dir)
    )

    print(f"Dataset downloaded to: {dataset.location}")
    print(
        "Roboflow exports usually include their own data.yaml. Compare it "
        "against configs/data.yaml and update configs/data.yaml's paths "
        "(and class names) to match, or point train.py at the exported one."
    )


if __name__ == "__main__":
    main()
