from pathlib import Path
from ultralytics import YOLO

folder = Path(__file__).resolve().parent

if __name__ == "__main__":
    model = YOLO(str(folder / "runs" / "detect" / "gas_bottles_e100" / "weights" / "best.pt"))


    metrics = model.val(
        data=str(folder / "dataset" / "data.yaml"),
        split="test",
        imgsz=640,
        batch=4,
        project=str(folder / "runs" / "detect"),
        name="test_eval",
        plots=True,
    )
    print("TEST precision:", round(metrics.box.mp, 3))
    print("TEST recall:   ", round(metrics.box.mr, 3))
    print("TEST mAP50:    ", round(metrics.box.map50, 3))
    print("TEST mAP50-95: ", round(metrics.box.map, 3))


    model.predict(
        source=str(folder / "dataset" / "images" / "test"),
        conf=0.25,
        save=True,
        project=str(folder / "runs" / "detect"),
        name="test_predictions",
    )
