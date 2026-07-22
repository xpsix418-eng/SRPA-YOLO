import torch

from srpa_yolo import ROOT, load_model


def test_minimal_forward() -> None:
    model = load_model(ROOT / "configs/models/srpa_yolon.yaml").model.eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 64, 64))
    prediction = output[0] if isinstance(output, (tuple, list)) else output
    assert prediction.ndim == 3
    assert torch.isfinite(prediction).all()

