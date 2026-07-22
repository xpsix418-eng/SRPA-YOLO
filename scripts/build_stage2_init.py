"""Build a seed-controlled SRPA private-stage initialization from the frozen Stage-I model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


SOURCE = ROOT / "weights/srpa_stage1_shared64_dut_seed42_best.pt"
MODEL = ROOT / "configs/models/srpa_yolon.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--model", type=Path, default=MODEL)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    source_path = args.source if args.source.is_absolute() else ROOT / args.source
    model_path = args.model if args.model.is_absolute() else ROOT / args.model
    if not source_path.is_file():
        raise FileNotFoundError(f"Stage-I checkpoint does not exist: {source_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"SRPA model configuration does not exist: {model_path}")
    source = YOLO(source_path)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    target = YOLO(model_path)
    src, dst = source.model.state_dict(), target.model.state_dict()
    with torch.no_grad():
        for key, value in dst.items():
            if not key.startswith("model.28.") and key in src and src[key].shape == value.shape:
                value.copy_(src[key])
        for i in range(3):
            for branch in ("conv1", "conv2"):
                old = f"model.28.stem.{i}.{branch}"
                new = f"model.28.shared_stem.{i}.{branch}"
                for suffix in (
                    "conv.weight", "bn.weight", "bn.bias", "bn.running_mean", "bn.running_var",
                    "bn.num_batches_tracked",
                ):
                    dst[f"{new}.{suffix}"].copy_(src[f"{old}.{suffix}"])
            old_identity = f"model.28.stem.{i}.bn"
            new_identity = f"model.28.shared_stem.{i}.bn"
            if f"{old_identity}.weight" in src:
                for suffix in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
                    dst[f"{new_identity}.{suffix}"].copy_(src[f"{old_identity}.{suffix}"])
            for suffix in ("weight", "bias"):
                dst[f"model.28.cv2.{i}.{suffix}"].copy_(src[f"model.28.cv2.{i}.{suffix}"])
                dst[f"model.28.cv3_shared.{i}.{suffix}"].copy_(src[f"model.28.cv3.{i}.{suffix}"])
            dst[f"model.28.cv3_private.{i}.weight"].zero_()
        dst["model.28.dfl.conv.weight"].copy_(src["model.28.dfl.conv.weight"])
    target.model.load_state_dict(dst, strict=True)
    source.model.eval()
    target.model.eval()
    generator = torch.Generator().manual_seed(20260714)
    sample = torch.randn(1, 3, 640, 640, generator=generator)
    with torch.no_grad():
        source_output, target_output = source.model(sample), target.model(sample)
    decoded_diff = float((source_output[0] - target_output[0]).abs().max())
    raw_diff = max(float((a - b).abs().max()) for a, b in zip(source_output[1], target_output[1]))
    if decoded_diff > 1e-5 or raw_diff > 1e-5:
        raise RuntimeError(f"initial equivalence failed: decoded={decoded_diff}, raw={raw_diff}")
    output.parent.mkdir(parents=True, exist_ok=True)
    target.save(output)
    metadata = {
        "seed": args.seed,
        "source": str(source_path.relative_to(ROOT)),
        "model": str(model_path.relative_to(ROOT)),
        "output": str(output.relative_to(ROOT)),
        "decoded_max_diff": decoded_diff,
        "raw_max_diff": raw_diff,
        "private_projection_zero": True,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
