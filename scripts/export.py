"""Fuse and export an SRPA-YOLO checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from srpa_yolo import ROOT, load_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("weights/srpa_yolo_dut_seed42_best.pt"))
    parser.add_argument("--format", choices=("onnx", "torchscript", "engine"), default="onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true")
    args = parser.parse_args()
    weights = args.weights if args.weights.is_absolute() else ROOT / args.weights
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {weights}")
    model = load_model(weights)
    model.model.fuse(verbose=False)
    exported = model.export(format=args.format, imgsz=args.imgsz, half=args.half)
    print(exported)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

