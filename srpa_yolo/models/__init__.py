"""Model loading helpers."""

from pathlib import Path

from ultralytics import YOLO


def load_model(source: str | Path) -> YOLO:
    """Load an SRPA-YOLO YAML or checkpoint with an explicit existence check."""
    path = Path(source).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Model source does not exist: {path}")
    return YOLO(str(path))


__all__ = ("load_model",)

