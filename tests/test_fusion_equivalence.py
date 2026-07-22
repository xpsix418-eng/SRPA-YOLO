import copy

import torch

from srpa_yolo import ROOT, load_model


def test_fusion_equivalence() -> None:
    model = load_model(ROOT / "configs/models/srpa_yolon.yaml").model.eval()
    fused = copy.deepcopy(model).eval().fuse(verbose=False)
    sample = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        before, after = model(sample), fused(sample)
    before = before[0] if isinstance(before, (tuple, list)) else before
    after = after[0] if isinstance(after, (tuple, list)) else after
    assert torch.allclose(before, after, atol=3e-4, rtol=1e-4)
