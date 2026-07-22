# Model fusion and export

The training head contains separate 64-channel shared and 16-channel classification-private RepConv stems. Fusion concatenates their equivalent 3x3 kernels. The box projection is padded with zero private columns, while shared and private class projections are concatenated. No private runtime branch remains.

Verify equivalence:

```bash
python scripts/verify_fusion.py --weights weights/srpa_yolo_dut_seed42_best.pt
```

The verifier reports both maximum and mean absolute output differences. The default maximum tolerance is `3e-4`, which covers the observed floating-point accumulation error of the formal checkpoint without relaxing functional equivalence.

Export:

```bash
python scripts/export.py --weights weights/srpa_yolo_dut_seed42_best.pt --format onnx --imgsz 640
```

TensorRT export requires a compatible CUDA/TensorRT installation and can be requested with `--format engine --half`.
