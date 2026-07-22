"""Generate publication-ready vector figures for the SRPA-YOLO manuscript."""

from __future__ import annotations

import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
METHOD_OUT = Path(__file__).resolve().parent
METHOD_SOURCE = METHOD_OUT
EXP_OUT = METHOD_OUT
EXP_SOURCE = METHOD_OUT
for directory in (METHOD_OUT, METHOD_SOURCE, EXP_OUT, EXP_SOURCE):
    directory.mkdir(parents=True, exist_ok=True)

INK = "#1F2A35"
EDGE = "#49637A"
BLUE = "#A9C9EA"
BLUE_DARK = "#4C78A8"
ORANGE = "#F3B184"
GOLD = "#F2C879"
PINK = "#E8B6CF"
GREEN = "#A8D5BA"
GRAY = "#E4E8EC"
RED = "#C94747"


def save_vector(fig: plt.Figure, directory: Path, source: Path, name: str) -> None:
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(directory / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(source / f"{name}.svg", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def write_csv(name: str, header: list[str], rows: list[list[object]]) -> None:
    with (EXP_SOURCE / name).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def setup(ax, xlim=(0, 1), ylim=(0, 1)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")


def frame(ax, x, y, w, h, title, color=EDGE):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=1.3, linestyle=(0, (4, 3))))
    ax.text(x + w / 2, y + h - 0.025, title, ha="center", va="top", fontsize=10.5,
            fontweight="bold", color=INK)


def box(ax, x, y, w, h, label, color=GRAY, fontsize=9, weight="normal"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        facecolor=color, edgecolor=EDGE, linewidth=1.3,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize,
            fontweight=weight, color=INK, linespacing=1.2)
    return {"x": x, "y": y, "w": w, "h": h, "patch": patch}


def arrow(ax, x1, y1, x2, y2, color=INK, linewidth=1.3):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
                                 linewidth=linewidth, color=color, shrinkA=0, shrinkB=0))


def anchor(node, side):
    x, y, w, h = node["x"], node["y"], node["w"], node["h"]
    return {
        "left": (x, y + h / 2), "right": (x + w, y + h / 2),
        "top": (x + w / 2, y + h), "bottom": (x + w / 2, y),
    }[side]


def connect(ax, source, target, source_side="right", target_side="left", color=INK, linewidth=1.3):
    x1, y1 = anchor(source, source_side)
    x2, y2 = anchor(target, target_side)
    arrow(ax, x1, y1, x2, y2, color=color, linewidth=linewidth)


def orthogonal(ax, source, target, source_side="right", target_side="left", color=INK, linewidth=1.3, mid=None):
    x1, y1 = anchor(source, source_side)
    x2, y2 = anchor(target, target_side)
    if mid is None:
        mid = (x1 + x2) / 2
    ax.plot([x1, mid, mid], [y1, y1, y2], color=color, linewidth=linewidth)
    arrow(ax, mid, y2, x2, y2, color=color, linewidth=linewidth)


def fig1_network():
    fig, ax = plt.subplots(figsize=(16.0, 7.0))
    setup(ax)
    frame(ax, 0.075, 0.08, 0.205, 0.84, "Backbone")
    frame(ax, 0.305, 0.08, 0.270, 0.84, "Bidirectional Neck")
    frame(ax, 0.600, 0.08, 0.175, 0.84, "SRPA Head (Training)")
    frame(ax, 0.800, 0.08, 0.175, 0.84, "SRPA Head (Deployment)")

    inp = box(ax, 0.008, 0.69, 0.055, 0.11, "Input\n640 x 640", "#FFFFFF", 8.8, "bold")
    backbone_specs = [
        ("Conv / 2", GOLD), ("C2f", "#D6E9C5"), ("Conv / 2", GOLD),
        ("C2f + CBAM", "#D6E9C5"), ("Conv / 2", GOLD), ("C2f", "#D6E9C5"),
        ("SPPF +\nContext", GREEN),
    ]
    bb = []
    for i, (label, color) in enumerate(backbone_specs):
        bb.append(box(ax, 0.113, 0.79 - i * 0.102, 0.128, 0.066, label, color, 8.2,
                      "bold" if i in (3, 6) else "normal"))
    connect(ax, inp, bb[0])
    for a, b in zip(bb[:-1], bb[1:]):
        connect(ax, a, b, "bottom", "top")

    neck_left, neck_right = [], []
    neck_labels = ["Upsample", "Concat", "GhostConv", "Upsample", "Concat", "C2f"]
    for i, label in enumerate(neck_labels):
        y = 0.79 - i * 0.118
        neck_left.append(box(ax, 0.327, y, 0.095, 0.070, label, "#F9E2A6", 8.0))
        neck_right.append(box(ax, 0.455, y, 0.095, 0.070,
                              "DWConv" if i in (0, 3) else label,
                              "#F9E2A6" if i not in (0, 3) else ORANGE, 8.0))
    for col in (neck_left, neck_right):
        for a, b in zip(col[:-1], col[1:]):
            connect(ax, a, b, "bottom", "top")
    for i in (1, 4):
        connect(ax, neck_left[i], neck_right[i])
    connect(ax, bb[6], neck_left[0])
    orthogonal(ax, bb[3], neck_left[4], mid=0.292)
    orthogonal(ax, bb[5], neck_left[1], mid=0.292)

    train_nodes, deploy_nodes = [], []
    scale_specs = [("P2\nPA Head\nStride 4", "80 x 80"),
                   ("P3\nPA Head\nStride 8", "40 x 40"),
                   ("P4\nPA Head\nStride 16", "20 x 20")]
    for i, (label, size) in enumerate(scale_specs):
        y = 0.69 - i * 0.245
        train_nodes.append(box(ax, 0.625, y, 0.125, 0.145,
                               f"{label}\n{size}", BLUE, 8.4, "bold"))
        deploy_nodes.append(box(ax, 0.825, y, 0.125, 0.145,
                                f"P{i + 2}\nSingle Conv\n{size}", "#DED3ED", 8.4, "bold"))
        connect(ax, train_nodes[-1], deploy_nodes[-1])
    connect(ax, neck_right[1], train_nodes[0])
    connect(ax, neck_right[3], train_nodes[1])
    connect(ax, neck_right[5], train_nodes[2])
    ax.text(0.687, 0.125, r"$z_{cls}=W_sH_s+W_pH_p$", ha="center", va="center", fontsize=9.5)
    ax.text(0.887, 0.125, r"$z_{box}=W_bH_s$", ha="center", va="center", fontsize=9.5)
    save_vector(fig, METHOD_OUT, METHOD_SOURCE, "fig1_overall_framework")


def fig2_feature_allocation():
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.8))
    for ax in axes:
        setup(ax)
    axes[0].set_title("(a) Traditional Shared Head", fontsize=12, fontweight="bold", color=INK)
    sf = box(axes[0], 0.29, 0.76, 0.42, 0.12, "Shared Feature\n(high-level representation)", BLUE, 10, "bold")
    sh = box(axes[0], 0.32, 0.52, 0.36, 0.12, "Shared Head\nConvolution", GRAY, 10)
    cl = box(axes[0], 0.08, 0.23, 0.34, 0.13, "Classification\nOutput", "#D6E9C5", 10)
    bx = box(axes[0], 0.58, 0.23, 0.34, 0.13, "Bounding-box\nOutput", ORANGE, 10)
    connect(axes[0], sf, sh, "bottom", "top")
    connect(axes[0], sh, cl, "bottom", "top")
    connect(axes[0], sh, bx, "bottom", "top")
    axes[0].text(0.50, 0.08, "Classification and localization update\nthe same feature dictionary.",
                 ha="center", va="center", fontsize=9.2, color=EDGE)

    axes[1].set_title("(b) SRPA-YOLO Head", fontsize=12, fontweight="bold", color=INK)
    sf = box(axes[1], 0.29, 0.80, 0.42, 0.11, "Shared Feature\n(high-level representation)", BLUE, 10, "bold")
    sb = box(axes[1], 0.08, 0.56, 0.31, 0.13, "Shared Branch\n64 channels", "#D6E9C5", 10)
    pb = box(axes[1], 0.48, 0.56, 0.31, 0.13, "Private Branch\n16 channels\n(classification only)", "#F9E2A6", 9.2)
    pj = box(axes[1], 0.31, 0.36, 0.28, 0.10, "Task Projection", BLUE, 10)
    cl = box(axes[1], 0.08, 0.12, 0.32, 0.12, "Classification\nOutput", "#D6E9C5", 10)
    bx = box(axes[1], 0.55, 0.12, 0.32, 0.12, "Bounding-box\nOutput", ORANGE, 10)
    connect(axes[1], sf, sb, "bottom", "top")
    connect(axes[1], sf, pb, "bottom", "top")
    connect(axes[1], sb, pj, "bottom", "top")
    connect(axes[1], pb, pj, "bottom", "top", RED)
    connect(axes[1], pj, cl, "bottom", "top")
    x1, y1 = anchor(sb, "top")
    x2, y2 = anchor(bx, "top")
    axes[1].plot([x1, x1, 0.985, 0.985, x2], [y1, 0.94, 0.94, 0.30, 0.30], color=INK, linewidth=1.3)
    arrow(axes[1], x2, 0.30, x2, y2, color=INK, linewidth=1.3)
    axes[1].text(0.88, 0.49, "Private rank affects\nclassification only", ha="center", va="center",
                 fontsize=9, color=RED)
    axes[1].text(0.71, 0.035, r"$\partial z_{box}/\partial H_p=0$", ha="center", fontsize=11, color=RED)
    fig.tight_layout(w_pad=2.0)
    save_vector(fig, METHOD_OUT, METHOD_SOURCE, "fig2_feature_allocation")


def fig3_reparameterization():
    fig, ax = plt.subplots(figsize=(13.6, 5.6))
    setup(ax)
    frame(ax, 0.03, 0.07, 0.39, 0.86, "(a) Training Stage")
    frame(ax, 0.58, 0.07, 0.39, 0.86, "(b) Deployment Stage")
    inp1 = box(ax, 0.145, 0.72, 0.16, 0.10, "Input Feature", "#E5EFF9", 10, "bold")
    shared = box(ax, 0.055, 0.55, 0.15, 0.12, "Shared RepConv\n64 channels", "#D6E9C5", 9.5)
    private = box(ax, 0.245, 0.55, 0.15, 0.12, "Private RepConv\n16 channels", "#F9E2A6", 9.5)
    concat = box(ax, 0.145, 0.33, 0.16, 0.10, "Concatenate\n80 channels", GRAY, 9.5)
    proj = box(ax, 0.145, 0.14, 0.16, 0.10, "Output Projection\n1 x 1 Conv", BLUE, 9.5)
    connect(ax, inp1, shared, "bottom", "top")
    connect(ax, inp1, private, "bottom", "top")
    connect(ax, shared, concat, "bottom", "top")
    connect(ax, private, concat, "bottom", "top")
    connect(ax, concat, proj, "bottom", "top")
    ax.text(0.50, 0.50, "Exact\nFusion", ha="center", va="center", fontsize=11, fontweight="bold", color=EDGE)
    arrow(ax, 0.43, 0.50, 0.57, 0.50, EDGE, 1.5)
    inp2 = box(ax, 0.695, 0.72, 0.16, 0.10, "Input Feature", "#E5EFF9", 10, "bold")
    fused = box(ax, 0.675, 0.49, 0.20, 0.16, "Equivalent Convolution\n80 channels, 3 x 3", "#DED3ED", 10, "bold")
    proj2 = box(ax, 0.695, 0.23, 0.16, 0.11, "Output Projection\n1 x 1 Conv", BLUE, 9.5)
    out = box(ax, 0.715, 0.09, 0.12, 0.08, "Output", "#FFFFFF", 10, "bold")
    connect(ax, inp2, fused, "bottom", "top")
    connect(ax, fused, proj2, "bottom", "top")
    connect(ax, proj2, out, "bottom", "top")
    ax.text(0.89, 0.56, "Two branches are fused\ninto one equivalent kernel.", ha="left", va="center",
            fontsize=8.6, color=EDGE)
    ax.text(0.89, 0.29, "Functionally identical\nto the training graph.", ha="left", va="center",
            fontsize=8.6, color=EDGE)
    save_vector(fig, METHOD_OUT, METHOD_SOURCE, "fig3_reparameterization")


def fig4_srpa_head():
    fig, ax = plt.subplots(figsize=(14.2, 5.6))
    setup(ax)
    frame(ax, 0.18, 0.24, 0.46, 0.68, "SRPA Head (per scale)")
    inp = box(ax, 0.025, 0.52, 0.13, 0.13, "Input Feature\n$C_{in}\\times H\\times W$", "#E5EFF9", 9.5, "bold")
    shared = box(ax, 0.23, 0.66, 0.17, 0.13, "Shared RepConv\n64 channels, 3 x 3", "#D6E9C5", 9.5)
    private = box(ax, 0.23, 0.37, 0.17, 0.13, "Private RepConv\n16 channels, 3 x 3\n(classification only)", "#F9E2A6", 8.7)
    concat = box(ax, 0.47, 0.52, 0.13, 0.13, "Concatenate\n80 channels", GRAY, 9.5)
    proj = box(ax, 0.69, 0.52, 0.16, 0.13, "Task Projections\n1 x 1 Conv", BLUE, 9.5, "bold")
    cls = box(ax, 0.87, 0.67, 0.12, 0.12, "Classification\nOutput", "#D6E9C5", 9.2)
    reg = box(ax, 0.87, 0.35, 0.12, 0.12, "Bounding-box\nOutput", ORANGE, 9.2)
    connect(ax, inp, shared)
    connect(ax, inp, private, color=RED)
    connect(ax, shared, concat)
    connect(ax, private, concat, color=RED)
    connect(ax, concat, proj)
    connect(ax, proj, cls)
    connect(ax, proj, reg)
    note_specs = [
        (0.19, "Shared representation\nserves both tasks."),
        (0.40, "Additional private rank\nis assigned only to classification."),
        (0.64, "Channel-wise\nconcatenation."),
        (0.82, "Task-specific\nprojections."),
    ]
    for x, label in note_specs:
        box(ax, x, 0.055, 0.16, 0.11, label, "#FFFFFF", 8.8)
    ax.text(0.445, 0.205, r"$\partial z_{box}/\partial H_p=0$", ha="center", fontsize=10.5, color=RED)
    save_vector(fig, METHOD_OUT, METHOD_SOURCE, "fig4_srpa_head_detail")


METHODS = ["YOLO11n", "YOLOv8n", "EDGS-YOLOv8", "DRBD-YOLOv8", "YOLOv8n-P2", "RLRD-YOLO", "SRPA-YOLO"]
PARAMS = np.array([2.582, 3.006, 3.091, 3.047, 2.921, 2.926, 2.164])
PRECISION = np.array([92.682, 92.762, 94.266, 92.658, 96.114, 95.781, 95.850])
RECALL = np.array([83.868, 83.959, 83.244, 83.601, 90.375, 90.107, 92.201])
MAP = np.array([59.280, 60.176, 60.347, 60.689, 65.407, 65.436, 67.071])
AP75 = np.array([66.383, 67.579, 66.415, 67.459, 76.290, 74.757, 77.329])
LATENCY = np.array([(21.6357 + 19.1505) / 2, (9.8859 + 9.9893) / 2,
                    (19.5146 + 22.0043) / 2, (29.0158 + 31.7115) / 2,
                    (19.1953 + 16.9629) / 2, (22.5644 + 22.7773) / 2,
                    (9.7435 + 9.8574) / 2])


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8)


def comparison():
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2))
    colors = ["#98A1A9"] * 6 + [RED]
    y = np.arange(len(METHODS))
    axes[0].barh(y, MAP, color=colors, edgecolor="white")
    axes[0].set_yticks(y, METHODS)
    axes[0].invert_yaxis()
    axes[0].set_xlim(57.5, 68)
    axes[0].set_xlabel("mAP@0.50:0.95 (%)")
    style_axis(axes[0])
    axes[1].scatter(RECALL, PRECISION, c=colors, s=65, edgecolors=INK, linewidth=0.6)
    for x, yv, name in zip(RECALL, PRECISION, METHODS):
        if name in {"YOLOv8n", "YOLOv8n-P2", "RLRD-YOLO", "SRPA-YOLO"}:
            offset = (5, 5) if name != "YOLOv8n" else (5, -13)
            axes[1].annotate(name, (x, yv), xytext=offset, textcoords="offset points", fontsize=7.5)
    axes[1].set_xlabel("Recall (%)")
    axes[1].set_ylabel("Precision (%)")
    style_axis(axes[1])
    fig.tight_layout(w_pad=2.2)
    write_csv("comparison.csv", ["method", "parameters_M", "precision_percent", "recall_percent", "map5095_percent", "ap75_percent"],
              [[m, p, pr, r, ma, a] for m, p, pr, r, ma, a in zip(METHODS, PARAMS, PRECISION, RECALL, MAP, AP75)])
    save_vector(fig, EXP_OUT, EXP_SOURCE, "comparison")


def complexity_tradeoff():
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    colors = ["#98A1A9"] * 6 + [RED]
    ax.scatter(PARAMS, MAP, c=colors, s=75, edgecolors=INK, linewidth=0.7, zorder=3)
    offsets = {"YOLO11n": (-10, -20), "YOLOv8n": (-52, -17), "EDGS-YOLOv8": (8, 12), "DRBD-YOLOv8": (-62, 18),
               "YOLOv8n-P2": (-82, 12), "RLRD-YOLO": (10, -18), "SRPA-YOLO": (5, 5)}
    for x, y, name in zip(PARAMS, MAP, METHODS):
        ax.annotate(name, (x, y), xytext=offsets[name], textcoords="offset points", fontsize=7.5,
                    fontweight="bold" if name == "SRPA-YOLO" else "normal")
    ax.annotate("Favorable", xy=(2.18, 66.8), xytext=(2.55, 63.7), arrowprops={"arrowstyle": "->", "color": EDGE},
                color=EDGE, fontsize=8.5)
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("mAP@0.50:0.95 (%)")
    ax.set_xlim(2.05, 3.18)
    ax.set_ylim(58.2, 68.0)
    style_axis(ax)
    fig.tight_layout()
    save_vector(fig, EXP_OUT, EXP_SOURCE, "complexity_tradeoff")


def accuracy_speed_tradeoff():
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    colors = ["#98A1A9"] * 6 + [RED]
    ax.scatter(LATENCY, MAP, c=colors, s=75, edgecolors=INK, linewidth=0.7, zorder=3)
    for x, y, name in zip(LATENCY, MAP, METHODS):
        offset = (5, 5)
        if name in {"YOLO11n", "DRBD-YOLOv8", "RLRD-YOLO"}:
            offset = (5, -13)
        ax.annotate(name, (x, y), xytext=offset, textcoords="offset points", fontsize=7.5,
                    fontweight="bold" if name == "SRPA-YOLO" else "normal")
    ax.annotate("Favorable Pareto region", xy=(9.9, 66.8), xytext=(14.0, 63.5),
                arrowprops={"arrowstyle": "->", "color": EDGE}, color=EDGE, fontsize=8.5)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("mAP@0.50:0.95 (%)")
    ax.set_xlim(8.2, 32.0)
    ax.set_ylim(58.0, 68.0)
    style_axis(ax)
    fig.tight_layout()
    write_csv("accuracy_speed.csv", ["method", "mean_latency_ms", "map5095_percent"],
              [[m, l, ma] for m, l, ma in zip(METHODS, LATENCY, MAP)])
    save_vector(fig, EXP_OUT, EXP_SOURCE, "accuracy_speed_tradeoff")


def training_curves():
    path = METHOD_OUT / "training_curves_data.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    epoch = np.array([int(float(row["epoch"])) for row in rows])
    values = {key: np.array([float(row[key]) for row in rows]) for key in rows[0] if key not in {"epoch", "time"}}
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8))
    for key, label, color in [("metrics/mAP50(B)", "AP@0.50", BLUE_DARK), ("metrics/mAP50-95(B)", "mAP@0.50:0.95", RED),
                              ("metrics/precision(B)", "Precision", "#D49A35"), ("metrics/recall(B)", "Recall", "#3A8D5D")]:
        axes[0].plot(epoch, 100 * values[key], label=label, color=color, linewidth=1.3)
    axes[0].set_xlabel("Private-stage epoch")
    axes[0].set_ylabel("Metric (%)")
    axes[0].legend(ncol=2, fontsize=8, frameon=False)
    style_axis(axes[0])
    for key, label, color in [("train/box_loss", "Box", BLUE_DARK), ("train/cls_loss", "Classification", "#D49A35"), ("train/dfl_loss", "DFL", RED)]:
        axes[1].plot(epoch, values[key], label=label, color=color, linewidth=1.3)
    axes[1].set_xlabel("Private-stage epoch")
    axes[1].set_ylabel("Training loss")
    axes[1].legend(fontsize=8, frameon=False)
    style_axis(axes[1])
    fig.tight_layout(w_pad=2.0)
    write_csv("training_curves.csv", ["epoch", "precision_percent", "recall_percent", "ap50_percent", "map5095_percent", "box_loss", "cls_loss", "dfl_loss"],
              [[e, 100 * values["metrics/precision(B)"][i], 100 * values["metrics/recall(B)"][i],
                100 * values["metrics/mAP50(B)"][i], 100 * values["metrics/mAP50-95(B)"][i],
                values["train/box_loss"][i], values["train/cls_loss"][i], values["train/dfl_loss"][i]] for i, e in enumerate(epoch)])
    save_vector(fig, EXP_OUT, EXP_SOURCE, "training_curves")


def scale_iou():
    iou = np.arange(50, 100, 5)
    v8 = 100 * np.array([.901282, .887038, .860783, .830963, .765785, .675788, .542495, .362682, .171144, .019605])
    ours = 100 * np.array([.951845, .941602, .925651, .906905, .862261, .773293, .639211, .450710, .219911, .035740])
    metrics = ["Precision", "Recall", "AP@0.50", "mAP@0.50:0.95", "AP@0.75"]
    v8_metrics = np.array([92.762, 83.959, 90.128, 60.176, 67.579])
    our_metrics = np.array([95.850, 92.201, 95.185, 67.071, 77.329])
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8))
    axes[0].plot(iou, v8, "o-", label="YOLOv8n", color=BLUE_DARK, linewidth=1.4, markersize=4)
    axes[0].plot(iou, ours, "o-", label="SRPA-YOLO", color=RED, linewidth=1.4, markersize=4)
    axes[0].set_xlabel("IoU threshold (%)")
    axes[0].set_ylabel("Average precision (%)")
    axes[0].legend(frameon=False)
    style_axis(axes[0])
    x = np.arange(len(metrics)); width = 0.36
    axes[1].bar(x - width / 2, v8_metrics, width, label="YOLOv8n", color=BLUE_DARK)
    axes[1].bar(x + width / 2, our_metrics, width, label="SRPA-YOLO", color=RED)
    axes[1].set_xticks(x, metrics, rotation=18, ha="right")
    axes[1].set_ylabel("Metric (%)")
    axes[1].set_ylim(55, 100)
    axes[1].legend(fontsize=8, frameon=False)
    style_axis(axes[1])
    fig.tight_layout(w_pad=2.0)
    write_csv("localization_comparison.csv", ["iou_percent", "yolov8n_ap_percent", "srpa_yolo_ap_percent"], list(map(list, zip(iou, v8, ours))))
    save_vector(fig, EXP_OUT, EXP_SOURCE, "scale_iou")


def qualitative():
    pred_path = ROOT / "results/manifests/srpa_test_predictions.json"
    predictions = json.loads(pred_path.read_text(encoding="utf-8"))
    by_name = defaultdict(list)
    for prediction in predictions:
        if prediction["score"] >= 0.25:
            by_name[prediction["file_name"]].append(prediction)
    selected = ["00005.jpg", "00042.jpg", "02179.jpg", "02197.jpg"]
    input_dir = EXP_SOURCE / "qualitative_input"
    input_dir.mkdir(exist_ok=True)
    annotations = {}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.5))
    for index, (axis, name) in enumerate(zip(axes.flat, selected)):
        dataset_root = Path(os.environ.get("DUT_DATASET_ROOT", ROOT / "datasets/DUT-Anti-UAV"))
        source = dataset_root / "images/test" / name
        shutil.copy2(source, input_dir / name)
        image = np.asarray(Image.open(source).convert("RGB"))
        axis.imshow(image)
        axis.axis("off")
        axis.set_title(f"({chr(97 + index)}) {name}", loc="left", fontsize=9)
        annotations[name] = []
        for prediction in by_name.get(name, [])[:4]:
            x, y, w, h = prediction["bbox"]
            axis.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=RED, linewidth=1.3))
            axis.text(x, max(0, y - 4), f"UAV {100 * prediction['score']:.1f}%", color=RED, fontsize=7,
                      bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1})
            annotations[name].append({"bbox_xywh": [x, y, w, h], "score_percent": 100 * prediction["score"]})
    fig.tight_layout(pad=0.8)
    (EXP_SOURCE / "qualitative_annotations.json").write_text(json.dumps(annotations, indent=2), encoding="utf-8")
    save_vector(fig, EXP_OUT, EXP_SOURCE, "qualitative")


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 9,
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "mathtext.fontset": "stix",
    })
    fig1_network()
    fig2_feature_allocation()
    fig3_reparameterization()
    fig4_srpa_head()
    comparison()
    complexity_tradeoff()
    accuracy_speed_tradeoff()
    training_curves()
    scale_iou()
    shutil.copy2(__file__, EXP_SOURCE / "generate_srpa_paper_figures.py")
    shutil.copy2(__file__, METHOD_SOURCE / "generate_srpa_paper_figures.py")
    print(f"Generated method figures in {METHOD_OUT}")
    print(f"Generated experimental figures in {EXP_OUT}")


if __name__ == "__main__":
    main()
