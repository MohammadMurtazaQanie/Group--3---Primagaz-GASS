import sys
from pathlib import Path
import torch
from ultralytics import YOLO

folder = Path(__file__).resolve().parent

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO("yolo11n-cls.pt")
    model.train(
        data=str(folder / "cls_dataset"),
        epochs=30,
        imgsz=224,
        batch=32,
        device=device,
        workers=0,
        seed=42,
        project=str(folder / "runs" / "classify"),
        name=sys.argv[1] if len(sys.argv) > 1 else "bottle_cls_v1",
        exist_ok=False,
    )
