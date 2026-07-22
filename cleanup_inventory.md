# SRPA-YOLO Cleanup Inventory

Prepared before any destructive cleanup. The original working directory is retained as the local archival worktree; the public release is assembled in a separate Git worktree so that no source artifact is lost during cleanup.

## Planned removal from the public release

- `paper_source/`: original Word manuscript, extracted text, and the complete MDPI template tree.
- `SRPA_YOLO_Drones/backup_before_academic_polishing/`, `JRTIP_Academic_Polished_20260719/`, `JRTIP_Bilingual_Upload_20260719/`, `JRTIP_Official_Template_20260719/`, and `JRTIP_Upload_20260719/`: superseded submission snapshots.
- MDPI/Drones-era manuscripts, archives, cover letters, and submission bundles, including `SRPA_YOLO_Drones_Submission.zip`.
- LaTeX intermediates (`*.aux`, `*.log`, `*.out`, `*.toc`, `*.synctex.gz`, `*.fls`, `*.fdb_latexmk`) and rendered QA directories (`tmp/`, `rendered/`, `layout_check/`).
- Failed or superseded AFNet experiment directories under `runs/afnet_hybrid_v20`, `v34`, `v42`, `v86`, `v92`, `v96`, `v97`, `v99`, and non-final candidate material in `runs/afnet_hybrid_v107`.
- Cancelled cross-dataset robustness outputs under `results/cross_dataset_direct/` and unrelated Det-Fly/TIB-Net training artifacts.
- Obsolete AFNet queue launchers, diagnostic scripts, document-generation scripts, historical model YAMLs, and modules unrelated to the final SRPA-YOLO path.
- Duplicate checkpoints, `last.pt`, initialization checkpoints, caches, IDE settings, screenshots, and local-only absolute-path configurations.
- Root-level `yolo11n.pt` and other third-party or baseline weights.

## Planned retention in the public release

- Final SRPA-YOLO implementation: the `RepSharedPrivateResidualDetect` training and fusion behavior plus required Ultralytics runtime dependencies.
- Canonical model definition derived from `ultralytics/cfg/models/v8/spra-yolon.yaml`, renamed and documented as SRPA-YOLO.
- Stage-I and Stage-II formal experiment configurations for seed 42, plus the preserved matched-seed 40/41/42 result records.
- Dataset YAML examples with repository-relative or environment-variable paths; no DUT Anti-UAV images or labels.
- Training, evaluation, export, paired-latency, fusion-verification, dataset-check, duplicate-check, plotting, and result-reproduction entry points.
- Formal independent-test summaries, rank/mechanism ablations, matched-seed summaries, latency JSON files, deduplication manifest, and checkpoint metadata/SHA-256.
- License, citation metadata, environment specifications, README, reproducibility documentation, and smoke tests.

## Planned moves and renames

- `ultralytics/cfg/models/v8/spra-yolon.yaml` -> `configs/models/srpa_yolon.yaml`.
- Final experiment YAMLs -> `configs/experiments/` with SRPA naming and portable paths.
- Dataset definitions -> `configs/datasets/`.
- Required implementation modules -> `srpa_yolo/` and a vendored minimal `ultralytics/` runtime where compatibility requires it.
- Selected operational scripts -> stable names under `scripts/` and `tools/`.
- Formal result summaries -> `results/main_results/`, `results/ablations/`, `results/latency/`, and `results/manifests/`.

## Post-cleanup publication split

- The current English manuscript, bibliography, compiled PDF, cited figures, and editable figure sources are distributed as a separate submission ZIP.
- No manuscript source, journal template, or paper figure is retained in the public reproducibility-code repository.

## Uncertain items retained outside the public release pending manual review

- Historical third-party comparison implementations embedded in the modified Ultralytics tree.
- Full training logs and visualization images for failed candidates.
- Det-Fly and TIB-Net experiment code/results, because the final paper does not report these experiments.
- The four compact checkpoints required by the formal two-stage protocol are retained directly because each is below 10 MB; all four have published SHA-256 values.
- Author submission metadata and cover-letter material not required to reproduce the article or experiments.
