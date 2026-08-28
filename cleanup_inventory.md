# SRPA-YOLO Cleanup Inventory

This inventory records the final public-release boundary. The immutable safety references are branch `archive/pre-cleanup` and tag `pre-cleanup-backup`; the GitHub release is maintained on `cleanup/reproducible-release`.

## Retained in the public repository

- SRPA-YOLO source code and the compatible Ultralytics runtime required for training, evaluation, fusion, export, and checkpoint loading.
- Portable model, dataset-example, and Stage-I/Stage-II experiment configurations.
- Public training, evaluation, Stage-II initialization, export, paired-latency, fusion-verification, and result-reproduction scripts.
- Formal DUT Anti-UAV result summaries, mechanism/rank ablations, matched-seed summaries, latency JSON records, deduplicated test manifest, and scale manifest.
- Four compact protocol checkpoints and their SHA-256 values.
- README, license, citation metadata, environment specifications, reproducibility documentation, and smoke tests.

## Archived outside Git

- Final English LaTeX submission package, compiled manuscript, editable figure sources, and Cover Letter.
- Formal raw evidence for the final SRPA-YOLO run, the Stage-I shared detector, SRPA/RLRD matched seeds, controlled DUT evaluations, and paired latency.
- The local external archive is intentionally not referenced by runtime code or committed documentation.
- A SHA-256 manifest is stored at the archive root.

## Deleted from the active workspace

- MDPI/Drones manuscripts and templates, superseded JRTIP snapshots, duplicate PDFs/DOCX/TEX files, rendered QA images, and obsolete submission bundles.
- Failed or superseded experimental candidate runs, duplicate and invalid checkpoints, `last.pt` copies not needed by the formal protocol, and cancelled cross-dataset artifacts.
- Historical launchers, diagnostic scripts, discarded model configurations, temporary CSV files, screenshots, and spreadsheet-reader output.
- `tmp`, smoke outputs, Python/pytest caches, IDE metadata, LaTeX intermediates, and package build artifacts.
- Raw datasets and third-party baseline checkpoints that are not redistributed.

## Public moves and names

| Archived role | Public location |
|---|---|
| final SRPA-YOLO model | `configs/models/srpa_yolon.yaml` |
| shared Stage-I model | `configs/models/srpa_shared64.yaml` |
| formal protocol settings | `configs/experiments/` |
| dataset template | `configs/datasets/dut_anti_uav.example.yaml` |
| stable package API | `srpa_yolo/` |
| compatible runtime | `ultralytics/` |
| operational entry points | `scripts/` and `tools/` |
| formal evidence | `results/` and `weights/` |

## Items requiring user-provided assets

- DUT Anti-UAV images and labels must be obtained under the dataset license.
- The official Ultralytics YOLOv8n checkpoint used as the paired-latency reference is not redistributed.
- Manuscript and submission files remain intentionally separate from the reproducibility repository.
