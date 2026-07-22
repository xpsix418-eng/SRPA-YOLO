"""Order-balanced paired latency benchmark for SRPA-YOLO and YOLOv8n."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression


def percentile(values: list[float], q: float) -> float:
    """Return a linearly interpolated percentile."""
    values = sorted(values)
    index = (len(values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def prediction(output):
    """Extract decoded predictions from an Ultralytics model output."""
    return output[0] if isinstance(output, (tuple, list)) else output


def load_model(path: Path, device: torch.device, half: bool):
    """Load, fuse, and place one model on the benchmark device."""
    model = YOLO(str(path)).model.eval()
    model.fuse(verbose=False)
    model.to(device)
    if half and device.type == "cuda":
        model.half()
    return model


def timed_step(model, image: torch.Tensor, device: torch.device) -> tuple[float, float, float]:
    """Measure synchronized forward, NMS, and end-to-end latency."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    output = prediction(model(image))
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    non_max_suppression(output, conf_thres=0.25, iou_thres=0.7, max_det=300)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    return (t1 - t0) * 1000, (t2 - t1) * 1000, (t2 - t0) * 1000


def summarize(name: str, path: Path, model, samples: list[tuple[float, float, float]], imgsz: int) -> dict:
    """Build one benchmark result row."""
    forward = [sample[0] for sample in samples]
    nms = [sample[1] for sample in samples]
    end_to_end = [sample[2] for sample in samples]
    mean_ms = statistics.fmean(end_to_end)
    return {
        "name": name,
        "path": path.as_posix(),
        "params_m": round(sum(parameter.numel() for parameter in model.parameters()) / 1_000_000, 3),
        "imgsz": imgsz,
        "device": str(next(model.parameters()).device),
        "precision": "fp16" if next(model.parameters()).dtype == torch.float16 else "fp32",
        "fused": True,
        "forward_mean_ms": round(statistics.fmean(forward), 4),
        "nms_mean_ms": round(statistics.fmean(nms), 4),
        "mean_ms": round(mean_ms, 4),
        "p50_ms": round(percentile(end_to_end, 0.50), 4),
        "p95_ms": round(percentile(end_to_end, 0.95), 4),
        "fps_mean": round(1000.0 / mean_ms, 2),
        "iters": len(samples),
        "status": "ok",
    }


def main() -> int:
    """Run interleaved AB/BA warmup and measurement pairs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=80)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    half = not args.fp32 and device.type == "cuda"
    reference_path = Path(args.reference).expanduser()
    candidate_path = Path(args.candidate).expanduser()
    reference_path = reference_path if reference_path.is_absolute() else ROOT / reference_path
    candidate_path = candidate_path if candidate_path.is_absolute() else ROOT / candidate_path
    for label, path in (("reference", reference_path), ("candidate", candidate_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} checkpoint does not exist: {path}")
    reference = load_model(reference_path, device, half)
    candidate = load_model(candidate_path, device, half)
    image = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)
    if half:
        image = image.half()

    models = (reference, candidate)
    with torch.inference_mode():
        for index in range(args.warmup):
            order = (0, 1) if index % 2 == 0 else (1, 0)
            for model_index in order:
                output = prediction(models[model_index](image))
                non_max_suppression(output, conf_thres=0.25, iou_thres=0.7, max_det=300)
        if device.type == "cuda":
            torch.cuda.synchronize()

        samples = [[], []]
        for index in range(args.iters):
            order = (0, 1) if index % 2 == 0 else (1, 0)
            for model_index in order:
                samples[model_index].append(timed_step(models[model_index], image, device))

    rows = [
        summarize("yolov8n", reference_path, reference, samples[0], args.imgsz),
        summarize(args.candidate_name, candidate_path, candidate, samples[1], args.imgsz),
    ]
    rows.append(
        {
            "paired_delta_ms": round(rows[1]["mean_ms"] - rows[0]["mean_ms"], 4),
            "passed": rows[1]["mean_ms"] < rows[0]["mean_ms"],
            "protocol": "interleaved_ab_ba",
            "warmup_per_model": args.warmup,
            "iterations_per_model": args.iters,
        }
    )
    output_path = Path(args.out).expanduser()
    output_path = output_path if output_path.is_absolute() else ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
