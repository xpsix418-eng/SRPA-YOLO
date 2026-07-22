# DUT Anti-UAV dataset setup

The dataset is not distributed by this repository. Obtain it from its authorized source and retain the provider's terms.

Expected single-class YOLO layout:

```text
DUT-Anti-UAV/
  images/train
  images/val
  images/test
  labels/train
  labels/val
  labels/test
```

Each label row is `0 x_center y_center width height` with normalized coordinates. Copy `configs/datasets/dut_anti_uav.example.yaml` to `configs/datasets/dut_anti_uav.yaml` and set `path`. Do not commit the local YAML if it contains a machine-specific absolute path.

Audit labels before training:

```bash
python tools/dataset_check.py --data configs/datasets/dut_anti_uav.yaml --split train
python tools/dataset_check.py --data configs/datasets/dut_anti_uav.yaml --split val
python tools/dataset_check.py --data configs/datasets/dut_anti_uav.yaml --split test
```

The formal paper result uses the 2,199-image deduplicated membership list in `results/manifests/dut_test_dedup.txt`. Its entries are relative to the dataset root. Before evaluation, verify that the local `images/test` directory contains exactly this membership; do not change the list to improve a reported result.
