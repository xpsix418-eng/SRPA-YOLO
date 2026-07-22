# SRPA-YOLO Project Cleanup Report

## Scope and safety

- Original workspace: retained locally on branch `archive/pre-cleanup` at commit `ee61be8`.
- Backup tag: annotated tag `pre-cleanup-backup` (tag object `14f832e77655fcf52d3acc43fdc692d750656a97`).
- Public release worktree: `cleanup/reproducible-release`.
- No force push, history rewrite, raw dataset upload, or modification of the archived source workspace was performed.
- The cleanup inventory was created before files were excluded from the public worktree: `cleanup_inventory.md`.

## Cleanup outcome

The archived workspace contains 1,363 tracked paths and approximately 827.0 MB of local files. After separating the manuscript from GitHub, the public release contains 326 tracked paths and approximately 28.5 MB, a physical-size reduction of about 96.6%. The public branch contains only the implementation, reproducibility configurations, formal experimental evidence, compact required checkpoints, tests, and documentation.

### Excluded content

- Complete MDPI/Drones manuscript, template, cover-letter, and submission-package trees.
- Superseded JRTIP snapshots, backup manuscripts, duplicate PDFs/DOCX/TEX files, and rendered QA folders.
- Failed candidate experiment directories, cancelled cross-dataset experiments, queue launchers, diagnostic scripts, and historical model YAML trees.
- Duplicate and invalid checkpoints, `last.pt` files, caches, TensorBoard/W&B logs, screenshots, temporary CSV outputs, and local absolute-path configurations.
- Python/pytest caches, IDE metadata, LaTeX logs/intermediates, package build metadata, and raw datasets.

The detailed category inventory is in `cleanup_inventory.md`; all excluded material remains recoverable from the archive branch and tag.

### Retained content

- Final `RepSharedPrivateResidualDetect` implementation and the compatible Ultralytics 8.3.239 runtime required for checkpoint deserialization, training, validation, fusion, and export.
- Canonical Stage-I and Stage-II model/experiment configurations.
- Four compact checkpoints needed by the two-stage protocol and their SHA-256 values.
- Formal main results, ablations, matched-seed records, paired-latency records, deduplicated test membership, and scale manifest.
- Public training, evaluation, export, fusion, latency, plotting, dataset-audit, duplicate-audit, and result-reproduction entry points.

### Main moves and renames

| Archived role | Public location |
|---|---|
| final model YAML | `configs/models/srpa_yolon.yaml` |
| shared Stage-I YAML | `configs/models/srpa_shared64.yaml` |
| formal experiment settings | `configs/experiments/` |
| dataset template | `configs/datasets/dut_anti_uav.example.yaml` |
| stable public API | `srpa_yolo/` |
| compatible runtime | `ultralytics/` |
| operational entry points | `scripts/` and `tools/` |
| formal evidence | `results/main_results/`, `results/ablations/`, `results/multiseed/`, and `results/latency/` |

## Final directory tree

```text
SRPA-YOLO/
|-- README.md
|-- LICENSE
|-- CITATION.cff
|-- requirements.txt
|-- environment.yml
|-- configs/
|   |-- datasets/
|   |-- experiments/
|   `-- models/
|-- srpa_yolo/
|   |-- models/
|   |-- modules/
|   |-- losses/
|   `-- utils/
|-- ultralytics/                 # compatible detection runtime
|-- scripts/
|   |-- train.py
|   |-- train_standard.py
|   |-- build_stage2_init.py
|   |-- evaluate.py
|   |-- verify_fusion.py
|   |-- export.py
|   |-- benchmark_latency.py
|   `-- reproduce_results.py
|-- tools/
|   |-- dataset_check.py
|   |-- duplicate_check.py
|   `-- plot_results.py
|-- tests/
|-- results/
|   |-- main_results/
|   |-- ablations/
|   |-- multiseed/
|   |-- latency/
|   `-- manifests/
|-- weights/
|-- assets/README_images/
`-- docs/
```

## Reproduction entry points

Install:

```bash
pip install -r requirements.txt
```

Stage I, function-preserving Stage-II initialization, and Stage II:

```bash
python scripts/train.py --config configs/experiments/srpa_stage1_seed42.yaml
python scripts/build_stage2_init.py --seed 42 --source runs/srpa_stage1/seed42/weights/best.pt --output runs/srpa_stage2/init/srpa_seed42_init.pt
python scripts/train.py --config configs/experiments/srpa_stage2_seed42.yaml
```

Evaluation and complete result reproduction:

```bash
python scripts/evaluate.py --weights weights/srpa_yolo_dut_seed42_best.pt --data configs/datasets/dut_anti_uav.yaml --imgsz 640 --batch 32 --conf 0.001 --iou 0.7 --max-det 300
python scripts/reproduce_results.py --config configs/experiments/srpa_yolo.yaml
```

Fusion and paired latency:

```bash
python scripts/verify_fusion.py --weights weights/srpa_yolo_dut_seed42_best.pt
python scripts/benchmark_latency.py --reference weights/yolov8n.pt --candidate weights/srpa_yolo_dut_seed42_best.pt --candidate-name SRPA-YOLO --imgsz 640 --warmup 80 --iters 300 --out results/latency/reproduced_pair.json
```

## Verification results

| Check | Result |
|---|---|
| package installation | editable installation succeeded in an external Python 3.12 validation environment |
| Python import/smoke/fusion tests | `3 passed` |
| minimal model forward | passed with finite predictions |
| Stage-II initialization | exact decoded/raw difference `0.0`; private projection zero verified |
| final checkpoint fusion at 640 | max absolute difference `2.44140625e-4`, mean `6.7095716e-6`, threshold `3e-4`; passed |
| script argument parsing | all eight public scripts returned valid `--help` output |
| Python syntax compilation | `compileall` passed for API, scripts, tools, and runtime |
| absolute-path scan | no local `E:/Ayolo`, `E:\\Ayolo`, or `C:\\Users` paths in public project content |
| credential scan | no GitHub token prefix, AWS access-key signature, or private-key block found |
| checkpoint hashes | all four files match `weights/SHA256SUMS` |
| raw datasets | none included |
| largest committed file | 9.06 MB Stage-I checkpoint; no file exceeds GitHub's 100 MB limit |

The current machine did not retain the original CUDA training environment. Dynamic smoke and fusion checks were therefore rerun with CPU PyTorch in an external disposable environment. GPU latency was not remeasured during cleanup; the two original same-process paired records are preserved unchanged under `results/latency/`.

## Manuscript distribution

The manuscript is intentionally excluded from GitHub. The independently delivered English submission ZIP contains the LaTeX source, compiled PDF, bibliography, all cited figures, and editable figure sources. Its clean-source build completed with 12 pages, no LaTeX errors, and no undefined citations or references.

## Large-file policy

No dataset, full training cache, TensorBoard history, video, or duplicate checkpoint is committed. Each of the four required checkpoints is below 10 MB and is committed directly; `weights/SHA256SUMS` and `results/manifests/checkpoint_manifest.json` provide integrity metadata. The YOLOv8n latency-reference checkpoint is not redistributed and must be placed at `weights/yolov8n.pt` by the reproducer.

## Git status

- Release branch: `cleanup/reproducible-release`.
- Archive branch: `archive/pre-cleanup`.
- Archive tag: `pre-cleanup-backup`.
- Commits before the release-evidence commit:
  - `173b8af chore: remove obsolete manuscripts and figures`
  - `c4a28f3 refactor: organize SRPA-YOLO source and configs`
  - `848ca40 docs: add reproducibility and dataset instructions`
  - `55b0bda test: add smoke and fusion-equivalence checks`
- Remote: `https://github.com/xpsix418-eng/SRPA-YOLO.git`.
- Initial branch push: successful; no force push was used.
- Final release commit: `chore: prepare public GitHub release` (this report and formal result evidence).

## Manual follow-up

1. Obtain DUT Anti-UAV under its license and create `configs/datasets/dut_anti_uav.yaml` from the example.
2. Verify that local test membership matches `results/manifests/dut_test_dedup.txt`.
3. Place the independently obtained YOLOv8n checkpoint at `weights/yolov8n.pt` before paired latency reproduction.
4. GitHub was empty before this release branch was pushed; create or designate a default `main` branch before opening a pull request.
