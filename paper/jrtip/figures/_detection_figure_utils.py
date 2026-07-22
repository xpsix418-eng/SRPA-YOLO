"""Shared drawing utilities for editable qualitative detection figures."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("DUT_DATASET_ROOT", ROOT / "datasets/DUT-Anti-UAV"))
ASSETS = Path(__file__).resolve().parent
SOURCE = Path(__file__).resolve().parent
ANALYSIS = SOURCE / "qualitative_case_analysis.json"
RED = "#D62728"
GOLD = "#F2B134"
INK = "#202A33"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 9,
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def load_cases(names: list[str]) -> list[dict]:
    payload = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    by_name = {case["file_name"]: case for case in payload["candidates"]}
    return [by_name[name] for name in names]


def image_array(name: str) -> np.ndarray:
    return np.asarray(Image.open(DATA / "images/test" / name).convert("RGB"))


def crop_bounds(box: list[float], width: int, height: int, context: float = 12.0) -> tuple[int, int, int, int]:
    x, y, w, h = box
    size = max(72.0, context * max(w, h))
    cx, cy = x + w / 2, y + h / 2
    x1 = max(0, int(round(cx - size / 2)))
    y1 = max(0, int(round(cy - size / 2)))
    x2 = min(width, int(round(cx + size / 2)))
    y2 = min(height, int(round(cy + size / 2)))
    return x1, y1, x2, y2


def add_box(axis, box: list[float], color: str, linestyle: str = "-", linewidth: float = 1.6) -> None:
    x, y, w, h = box
    axis.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=linewidth,
                             linestyle=linestyle, zorder=5))


def draw_pair(
    figure,
    spec,
    case: dict,
    prediction: bool,
    title: str,
    show_ground_truth: bool = True,
) -> None:
    sub = spec.subgridspec(1, 2, width_ratios=[2.35, 1.0], wspace=0.035)
    full_axis = figure.add_subplot(sub[0, 0])
    crop_axis = figure.add_subplot(sub[0, 1])
    image = image_array(case["file_name"])
    height, width = image.shape[:2]
    gt = case["gt_bbox_xywh"]
    x1, y1, x2, y2 = crop_bounds(gt, width, height)

    full_axis.imshow(image)
    full_axis.set_xlim(0, width)
    full_axis.set_ylim(height, 0)
    full_axis.axis("off")
    full_axis.set_title(title, loc="left", fontsize=9.5, fontweight="bold", color=INK, pad=3)
    if show_ground_truth:
        add_box(full_axis, gt, GOLD, "--", 1.4)
    if prediction:
        add_box(full_axis, case["srpa_bbox_xywh"], RED, "-", 1.6)

    crop_axis.imshow(image)
    crop_axis.set_xlim(x1, x2)
    crop_axis.set_ylim(y2, y1)
    crop_axis.set_xticks([])
    crop_axis.set_yticks([])
    for spine in crop_axis.spines.values():
        spine.set_color(RED if prediction else GOLD)
        spine.set_linewidth(1.5)
    if show_ground_truth:
        add_box(crop_axis, gt, GOLD, "--", 1.7)
    if prediction:
        add_box(crop_axis, case["srpa_bbox_xywh"], RED, "-", 1.9)
        label = f"UAV {100 * case['srpa_score']:.1f}%"
        color = RED
    else:
        label = "Missed"
        color = INK
    crop_axis.text(0.03, 0.96, label, transform=crop_axis.transAxes, ha="left", va="top",
                   fontsize=8.5, fontweight="bold", color=color,
                   bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.5})

    full_axis.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=GOLD,
                                  linewidth=1.0, linestyle=(0, (3, 2)), zorder=4))
    figure.add_artist(ConnectionPatch(xyA=(x2, y1), coordsA=full_axis.transData,
                                      xyB=(0, 1), coordsB=crop_axis.transAxes,
                                      color=GOLD, linewidth=0.8))
    figure.add_artist(ConnectionPatch(xyA=(x2, y2), coordsA=full_axis.transData,
                                      xyB=(0, 0), coordsB=crop_axis.transAxes,
                                      color=GOLD, linewidth=0.8))


def save_figure(figure, name: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    figure.savefig(ASSETS / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    figure.savefig(ASSETS / f"{name}.svg", bbox_inches="tight", pad_inches=0.03)
    figure.savefig(ASSETS / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)
