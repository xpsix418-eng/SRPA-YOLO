"""Run the public evaluation and optional paired-latency reproduction entry points."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ultralytics.utils import YAML

from srpa_yolo import ROOT


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--run-latency", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    if not config_path.is_file():
        raise FileNotFoundError(f"Reproduction configuration does not exist: {config_path}")
    config = dict(YAML.load(config_path))
    if not args.skip_evaluation:
        run([
            sys.executable, "scripts/evaluate.py", "--weights", str(config["model"]), "--data", str(config["data"]),
            "--imgsz", str(config.get("imgsz", 640)), "--batch", str(config.get("batch", 32)),
            "--conf", str(config.get("conf", 0.001)), "--iou", str(config.get("iou", 0.7)),
            "--max-det", str(config.get("max_det", 300)), "--device", str(config.get("device", 0)),
        ])
    run([sys.executable, "scripts/verify_fusion.py", "--weights", str(config["model"])])
    if args.run_latency:
        latency = config["latency"]
        run([
            sys.executable, "scripts/benchmark_latency.py", "--reference", str(latency["reference"]),
            "--candidate", str(config["model"]), "--candidate-name", "SRPA-YOLO",
            "--imgsz", str(config.get("imgsz", 640)), "--warmup", str(latency.get("warmup", 80)),
            "--iters", str(latency.get("iterations", 300)), "--out", "results/latency/reproduced_pair.json",
        ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
