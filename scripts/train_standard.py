"""Train or resume a standard Ultralytics detector from a YAML configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.utils import YAML


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if not args.config.is_file():
        raise FileNotFoundError(f"Experiment configuration does not exist: {args.config}")
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {args.resume}")
    config = dict(YAML.load(args.config))
    config.pop("task", None)
    config.pop("mode", None)
    if args.resume:
        model = YOLO(args.resume)
        overrides = {
            key: config[key]
            for key in ("data", "workers", "cache", "batch", "device", "epochs", "patience", "plots", "val")
            if key in config
        }
        model.train(resume=True, **overrides)
    else:
        model = YOLO(config.pop("model"))
        model.train(**config)


if __name__ == "__main__":
    main()
