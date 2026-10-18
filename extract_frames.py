from pathlib import Path
import cv2

folder = Path(__file__).resolve().parent
output = folder / "station_frames"

plan = {"14_55": 40, "14_43": 15}
cameras = ["front", "back_left", "back_right"]

for session, per_camera in plan.items():
    out_dir = output / session
    out_dir.mkdir(parents=True, exist_ok=True)
    for camera in cameras:
        video = folder / session / f"{session}_{camera}_cropped.mp4"
        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start, end = int(total * 0.02), int(total * 0.98)
        step = (end - start) // per_camera
        for i in range(per_camera):
            frame_number = start + i * step
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = cap.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            frame = cv2.resize(frame, (1280, int(height * 1280 / width)))
            name = f"{session}_{camera}_{frame_number:05d}.jpg"
            cv2.imwrite(str(out_dir / name), frame)
        cap.release()
        print(f"{session} {camera}: saved {per_camera} frames")

print("Done. Frames are in", output)
