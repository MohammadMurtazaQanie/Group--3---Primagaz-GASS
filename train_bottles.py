from pathlib import Path
import torch
from ultralytics import YOLO

folder = Path(__file__).resolve().parent

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(str(folder / "yolo11n.pt"))

    model.train(
        data=str(folder / "dataset" / "data.yaml"),
        epochs=100,
        imgsz=640,
        batch=4,
        device=device,
        workers=0,
        seed=42,
        project=str(folder / "runs" / "detect"),
        name="gas_bottles_e100",
        exist_ok=False,
        plots=True,
    )
