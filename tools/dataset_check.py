"""Validate a one-class YOLO detection dataset without modifying it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
from ultralytics.utils import YAML


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def iter_images(path: Path) -> list[Path]:
    if path.is_file():
        return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted(item for item in path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output", type=Path, default=Path("results/manifests/dataset_audit.json"))
    args = parser.parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {args.data}")
    cfg = dict(YAML.load(args.data))
    root = Path(cfg.get("path", args.data.parent))
    if not root.is_absolute():
        root = (args.data.parent / root).resolve()
    split = Path(cfg[args.split])
    split = split if split.is_absolute() else root / split
    images = iter_images(split)
    report = {"split": args.split, "images": len(images), "corrupt_images": [], "missing_labels": [],
              "empty_labels": [], "invalid_boxes": []}
    for image_path in images:
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as exc:
            report["corrupt_images"].append({"path": str(image_path), "error": str(exc)})
            continue
        label_path = Path(str(image_path).replace("images", "labels")).with_suffix(".txt")
        if not label_path.is_file():
            report["missing_labels"].append(str(label_path))
            continue
        lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            report["empty_labels"].append(str(label_path))
        for index, line in enumerate(lines, 1):
            try:
                cls, x, y, w, h = map(float, line.split())
                valid = cls == 0 and 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
                valid = valid and x - w / 2 >= 0 and x + w / 2 <= 1 and y - h / 2 >= 0 and y + h / 2 <= 1
            except ValueError:
                valid = False
            if not valid:
                report["invalid_boxes"].append({"path": str(label_path), "line": index, "value": line})
    report["passed"] = (
        bool(images)
        and not report["corrupt_images"]
        and not report["missing_labels"]
        and not report["invalid_boxes"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
