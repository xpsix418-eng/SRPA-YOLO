"""Evaluate a detector with the formal DUT Anti-UAV protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from srpa_yolo import ROOT, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, default=Path("results/main_results/evaluation.json"))
    return parser.parse_args()


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    """Return a portable path for repository files and a resolved path otherwise."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> int:
    args = parse_args()
    weights, data, output = resolve(args.weights), resolve(args.data), resolve(args.output)
    for label, path in (("weights", weights), ("data", data)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file does not exist: {path}")
    model = load_model(weights)
    model.model.eval()
    before = {name: value.detach().cpu().clone() for name, value in model.model.state_dict().items()}
    with torch.inference_mode():
        metrics = model.val(
            data=str(data), imgsz=args.imgsz, batch=args.batch, conf=args.conf, iou=args.iou,
            max_det=args.max_det, device=args.device, plots=False, save_json=False,
        )
    after = model.model.state_dict()
    if any(not torch.equal(before[name], value.detach().cpu()) for name, value in after.items()):
        raise RuntimeError("Model parameters or buffers changed during evaluation")
    ap_by_iou = metrics.box.all_ap.mean(0).tolist()
    result = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "ap50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "ap75": float(ap_by_iou[5]),
        "imgsz": args.imgsz, "batch": args.batch, "conf": args.conf, "iou": args.iou,
        "max_det": args.max_det, "weights": display_path(weights),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
