from pathlib import Path
from ultralytics import YOLO

folder = Path(__file__).resolve().parent

if __name__ == "__main__":
    model = YOLO(str(folder / "runs" / "detect" / "gas_bottles_e100" / "weights" / "best.pt"))

    results = model.track(
        source=str(folder / "bottle_video.mp4"),
        tracker="bytetrack.yaml",   
        conf=0.25,
        save=True,                 
        stream=True,                
        project=str(folder / "runs" / "detect"),
        name="bottle_track",
    )

    all_ids = set()
    per_frame = []
    for result in results:
        ids = result.boxes.id
        if ids is not None:
            all_ids.update(int(i) for i in ids)
        per_frame.append(len(result.boxes))

    print("Frames processed:", len(per_frame))
    print("Bottles per frame: min", min(per_frame), "max", max(per_frame),
          "average", round(sum(per_frame) / len(per_frame), 1))
    print("Different track IDs used:", len(all_ids))
