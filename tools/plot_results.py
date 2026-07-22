"""Plot the public DUT Anti-UAV accuracy and complexity comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from srpa_yolo import ROOT


def load_models(path: Path) -> dict[str, dict[str, object]]:
    """Load successful model records that contain mAP and parameter counts."""
    if not path.is_file():
        raise FileNotFoundError(f"Result file does not exist: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))["models"]
    return {
        name: record
        for name, record in records.items()
        if record.get("status") == "ok" and record.get("params") and record.get("map5095") is not None
    }


def plot_complexity(models: dict[str, dict[str, object]], output: Path) -> None:
    """Create an editable vector parameter-versus-accuracy plot."""
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10})
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    for name, record in models.items():
        x = float(record["params"]) / 1e6
        y = float(record["map5095"]) * 100
        highlight = name == "SRPA-YOLO"
        ax.scatter(x, y, s=90 if highlight else 48, marker="*" if highlight else "o",
                   color="#C94747" if highlight else "#4C78A8", zorder=3)
        ax.annotate(name, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("mAP@0.50:0.95 (%)")
    ax.spines[["top", "right"]].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight")
    if output.suffix.lower() != ".svg":
        fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/main_results/dut_main_comparison.json")
    parser.add_argument("--output", type=Path, default=ROOT / "assets/README_images/complexity_tradeoff.pdf")
    args = parser.parse_args()
    result_path = args.results if args.results.is_absolute() else ROOT / args.results
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    plot_complexity(load_models(result_path), output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
