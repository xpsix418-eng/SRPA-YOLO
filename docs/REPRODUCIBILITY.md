# Reproducibility

## Environment

Create the environment with `pip install -r requirements.txt` or `conda env create -f environment.yml`. The reported system used Ultralytics 8.3.239, Python 3.11.9, PyTorch 2.9.0, CUDA 12.8, and an NVIDIA GeForce RTX 5060 GPU.

## Data preparation

Follow `DATASET_SETUP.md`. The public repository does not include DUT Anti-UAV images or labels. The formal held-out list is `results/manifests/dut_test_dedup.txt`.

## Two-stage training

1. Stage I: `python scripts/train.py --config configs/experiments/srpa_stage1_seed42.yaml`
2. Initialization: `python scripts/build_stage2_init.py --seed 42 --source runs/srpa_stage1/seed42/weights/best.pt --output runs/srpa_stage2/init/srpa_seed42_init.pt`
3. Stage II: `python scripts/train.py --config configs/experiments/srpa_stage2_seed42.yaml`

Both stages use 200 epochs, batch 32, image size 640, two workers, SGD (`lr0=0.01`, momentum `0.937`, weight decay `0.0005`), deterministic execution, and seed 42. Stage-II seeds 40 and 41 differ only in the private-parameter initialization seed.

The teacher response loss uses temperature 2.0, classification weight 0.25, dense DFL weight 0.50, and confidence-focused DFL weight 0.125 with exponent 2.0. The best checkpoint is selected only by validation mAP@0.50:0.95.

## Evaluation

Run the command in the README. All detectors use the same letterbox preprocessing, confidence 0.001, NMS IoU 0.7, maximum 300 detections, and the bundled metric implementation. `scripts/evaluate.py` uses `model.eval()` and `torch.inference_mode()` and checks that parameters and buffers remain unchanged.

## Structural fusion

`scripts/verify_fusion.py` compares decoded outputs before and after fusion on a deterministic tensor. The test tolerance is `2e-4`, which covers floating-point operation-order differences while remaining far below prediction-scale changes.

## Latency

`scripts/benchmark_latency.py` loads SRPA-YOLO and YOLOv8n in one process, fuses both, uses FP16 on CUDA, performs 80 warm-ups per model, and records 300 alternating AB/BA forward-plus-NMS measurements at batch 1 and 640 pixels.

The paper comparison uses the included DUT Anti-UAV YOLOv8n checkpoint `weights/yolov8n_dut_seed42_best.pt`, trained with seed 42 and selected by validation mAP@0.50:0.95. Its SHA-256 is recorded in `weights/SHA256SUMS`. The upstream COCO checkpoint is not the paper's latency control.

The canonical main-table seed-42 result and the matched-seed seed-42 stability run use different checkpoints. Their run IDs, checkpoint hashes, result files, and roles are explicitly separated in `results/manifests/result_provenance.json`.

## Result locations

- Main comparison: `results/main_results/`
- Rank and mechanism ablations: `results/ablations/`
- Matched seeds: `results/multiseed/`
- Paired latency: `results/latency/`
- Test and checkpoint manifests: `results/manifests/`
