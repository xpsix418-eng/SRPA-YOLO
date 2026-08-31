"""Guard the public evidence files against drift from the reported experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(relative_path: str):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def sha256(relative_path: str) -> str:
    digest = hashlib.sha256()
    with (ROOT / relative_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_reported_result_lineage() -> None:
    provenance = load_json("results/manifests/result_provenance.json")
    comparison = load_json("results/main_results/dut_main_comparison.json")
    canonical = provenance["canonical_main_result"]
    reported = comparison["models"][canonical["result_key"]]

    for metric, value in canonical["metrics"].items():
        actual = reported["ap_by_iou"]["AP75"] if metric == "ap75" else reported[metric]
        assert actual == value

    assert sha256(canonical["checkpoint"]) == canonical["checkpoint_sha256"]
    control = provenance["yolov8n_control"]
    assert sha256(control["checkpoint"]) == control["checkpoint_sha256"]

    matched_seed42 = load_json("results/multiseed/srpa_seed42.json")
    assert matched_seed42["canonical_main_result"] is False
    assert matched_seed42["run_id"] != canonical["run_id"]
    assert matched_seed42["checkpoint_sha256"] != canonical["checkpoint_sha256"]


def test_reported_latency_rounds() -> None:
    round1 = load_json("results/latency/srpa_yolo_round1.json")
    round2 = load_json("results/latency/srpa_yolo_round2.json")

    assert (round1[1]["mean_ms"], round1[0]["mean_ms"]) == (9.7435, 9.8859)
    assert (round2[1]["mean_ms"], round2[0]["mean_ms"]) == (9.8574, 9.9893)
    assert round1[2]["passed"] is True
    assert round2[2]["passed"] is True
