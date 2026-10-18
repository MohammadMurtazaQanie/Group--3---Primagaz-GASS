from pathlib import Path
import torch
from ultralytics import YOLO

folder = Path(__file__).resolve().parent

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(str(folder / "runs/detect/gas_bottles_e100/weights/best.pt"))

    model.train(
        data=str(folder / "station_dataset" / "data.yaml"),
        epochs=60,
        imgsz=640,
        batch=8,
        device=device,
        workers=0,
        seed=42,
        project=str(folder / "runs" / "detect"),
        name="station_v1",
        exist_ok=False,
        plots=True,
    )
