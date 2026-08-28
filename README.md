# SRPA-YOLO

Official reproducibility repository for **SRPA-YOLO: Shared-Representation and Private-Rank Allocation for Real-Time Small-UAV Detection**.

SRPA-YOLO allocates a 64-channel representation to classification and box-distribution prediction and adds a zero-output-initialized 16-channel classification-private representation. The private path does not enter box regression. At deployment, both RepConv stems and task projections are fused algebraically into one static path per detection scale.

## Installation

```bash
git clone https://github.com/xpsix418-eng/SRPA-YOLO.git
cd SRPA-YOLO
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The reported experiments used Python 3.11, PyTorch 2.3.0, CUDA 12.1, and an NVIDIA GeForce RTX 5060 Laptop GPU. See [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Dataset

DUT Anti-UAV is not redistributed. Prepare the dataset in YOLO single-class format and copy:

```bash
cp configs/datasets/dut_anti_uav.example.yaml configs/datasets/dut_anti_uav.yaml
```

Then set `path` in the copied YAML. The deduplicated test manifest is retained in `results/manifests/`. Detailed checks are described in [DATASET_SETUP.md](docs/DATASET_SETUP.md).

## Training

Stage I learns the 64-channel shared detector:

```bash
python scripts/train.py --config configs/experiments/srpa_stage1_seed42.yaml
```

Build the function-preserving Stage-II initialization:

```bash
python scripts/build_stage2_init.py --seed 42 --source runs/srpa_stage1/seed42/weights/best.pt --output runs/srpa_stage2/init/srpa_seed42_init.pt
```

Stage II trains only the classification-private representation from the zero-output state:

```bash
python scripts/train.py --config configs/experiments/srpa_stage2_seed42.yaml
```

The formal protocol uses 200 epochs, batch 32, image size 640, SGD, seed 42, and maximum validation mAP@0.50:0.95 for checkpoint selection. Matched Stage-II seeds 40, 41, and 42 are recorded under `results/multiseed/`.

## Evaluation

```bash
python scripts/evaluate.py --weights weights/srpa_yolo_dut_seed42_best.pt --data configs/datasets/dut_anti_uav.yaml --imgsz 640 --batch 32 --conf 0.001 --iou 0.7 --max-det 300
```

The evaluator reports Precision, Recall, AP@0.50, mAP@0.50:0.95, and AP@0.75 and verifies that evaluation does not update model state.

## Fusion and export

```bash
python scripts/verify_fusion.py --weights weights/srpa_yolo_dut_seed42_best.pt
python scripts/export.py --weights weights/srpa_yolo_dut_seed42_best.pt --format onnx --imgsz 640
```

See [MODEL_EXPORT.md](docs/MODEL_EXPORT.md).

## Paired latency

The formal latency protocol measures batch 1, 640 pixels, FP16, fused forward plus NMS, 80 warm-ups, and 300 timed iterations in alternating AB/BA order:

Obtain the official Ultralytics YOLOv8n checkpoint from the upstream Ultralytics release and place it at `weights/yolov8n.pt`. The file is intentionally not redistributed by this repository. The benchmark exits with a clear `FileNotFoundError` if either checkpoint is missing.

```bash
python scripts/benchmark_latency.py --reference weights/yolov8n.pt --candidate weights/srpa_yolo_dut_seed42_best.pt --candidate-name SRPA-YOLO --imgsz 640 --warmup 80 --iters 300 --out results/latency/reproduced_pair.json
```

## Reproduce the reported checkpoint evaluation

```bash
python scripts/reproduce_results.py --config configs/experiments/srpa_yolo.yaml
```

Add `--run-latency` only after placing the official YOLOv8n reference checkpoint at `weights/yolov8n.pt`.

## Main DUT Anti-UAV result

| Method | Params (M) | Precision (%) | Recall (%) | AP@0.50 (%) | mAP@0.50:0.95 (%) | AP@0.75 (%) |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8n | 3.006 | 92.762 | 83.959 | 90.128 | 60.176 | 67.579 |
| SRPA-YOLO | **2.164** | **95.850** | **92.201** | **95.185** | **67.071** | **77.329** |

Two paired low-load measurements were 9.7435 ms versus 9.8859 ms and 9.8574 ms versus 9.9893 ms for SRPA-YOLO and YOLOv8n, respectively. Hardware and software conditions must be reported with any reproduced latency.

## Repository layout

```text
configs/       model, dataset, and experiment YAML files
srpa_yolo/     public model/module/utilities API
ultralytics/   bundled compatible training and inference runtime
scripts/       train, evaluate, fuse, export, latency, reproduce
tools/         dataset, duplicate, and plotting utilities
tests/         import, forward, and fusion smoke tests
results/       formal summaries, ablations, latency, and manifests
weights/       four traceable checkpoints required by the two-stage protocol
docs/          setup, reproducibility, and export instructions
```

Manuscript sources are intentionally distributed separately and are not tracked in this reproducibility-code repository.

## Data and weights

No raw dataset is included. The repository contains only the final checkpoint and the three compact checkpoints required to reproduce the formal two-stage initialization. SHA-256 values are listed in `weights/SHA256SUMS` and `results/manifests/checkpoint_manifest.json`.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Please cite the final journal article after publication.

## License

The bundled Ultralytics-derived runtime and this repository are distributed under the [AGPL-3.0 License](LICENSE).
