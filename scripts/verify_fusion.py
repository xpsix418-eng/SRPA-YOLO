"""Verify numerical equivalence before and after SRPA structural fusion."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from srpa_yolo import ROOT, load_model, set_reproducible_seed


def prediction(output: object) -> torch.Tensor:
    return output[0] if isinstance(output, (tuple, list)) else output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("weights/srpa_yolo_dut_seed42_best.pt"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--atol", type=float, default=3e-4,
                        help="Maximum tolerated absolute error from floating-point kernel fusion.")
    args = parser.parse_args()
    path = args.weights if args.weights.is_absolute() else ROOT / args.weights
    model = load_model(path).model.eval().cpu()
    fused = copy.deepcopy(model).eval().cpu().fuse(verbose=False)
    set_reproducible_seed(42)
    sample = torch.randn(1, 3, args.imgsz, args.imgsz)
    with torch.inference_mode():
        before = prediction(model(sample))
        after = prediction(fused(sample))
    diff = (before - after).abs()
    result = {"max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()), "atol": args.atol}
    result["passed"] = result["max_abs_diff"] <= args.atol
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError(f"Fusion equivalence failed: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
