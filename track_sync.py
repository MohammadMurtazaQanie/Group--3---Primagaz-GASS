"""Synchronised bottle tracking over the 3 Primagaz station cameras.

Every bottle gets ONE global ID that is the same in all cameras and that
increases (1, 2, 3, ...) in the order the bottles pass.

How it works:
1. Track bottles in each camera separately with YOLO + ByteTrack.
2. Throw away noise: tracks that are too short or too small
   (flickers, bottles far away in the background).
3. For every remaining track, find the moment its centre passes the middle
   of the image. That is the "passing time" of the bottle in that camera.
4. The cameras look at the same spot from different sides, so a bottle passes
   them at almost the same time. We measure the small time offset of each
   camera against the front camera and correct for it.
5. Passing times that fall close together (from any camera) belong to the
   same bottle. Bottles are numbered in time order -> global IDs.
6. Write a CSV with the bottles and a side-by-side demo video.

The videos must be synchronised (same start, same frame rate).
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

CAMERAS = ["front", "back_left", "back_right"]


def track_camera(model_path, video, conf, device):
    """Return {track_id: list of (frame, x1, y1, x2, y2, confidence)} for one video."""
    model = YOLO(str(model_path))
    tracks = {}
    for frame, result in enumerate(model.track(str(video), tracker="bytetrack.yaml",
                                               conf=conf, device=device, stream=True, verbose=False)):
        if result.boxes.id is None:
            continue
        for box, c, tid in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(),
                               result.boxes.id.tolist()):
            tracks.setdefault(int(tid), []).append((frame, *box, c))
    return tracks


def passing_frame(rows, width):
    """Frame where the box centre is closest to the middle of the image."""
    middle = width / 2
    best = min(rows, key=lambda r: abs((r[1] + r[3]) / 2 - middle))
    return best[0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--front", required=True)
    p.add_argument("--back_left", required=True)
    p.add_argument("--back_right", required=True)
    p.add_argument("--model", default="runs/detect/station_v1/weights/best.pt")
    p.add_argument("--out", default="runs/sync/demo")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--min-frames", type=int, default=15, help="ignore tracks shorter than this")
    p.add_argument("--min-width", type=float, default=0.2, help="ignore tracks narrower than this share of the image")
    p.add_argument("--window", type=float, default=1.5, help="seconds: passings closer than this are the same bottle")
    p.add_argument("--no-video", action="store_true", help="skip the demo video (faster)")
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    videos = {"front": args.front, "back_left": args.back_left, "back_right": args.back_right}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(videos["front"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap.release()

    # 1-3: track each camera, filter noise, find passing times
    passings = {}   # camera -> list of (passing_frame, track_id)
    all_tracks = {}
    for cam in CAMERAS:
        print(f"Tracking {cam} ...", flush=True)
        tracks = track_camera(args.model, videos[cam], args.conf, device)
        all_tracks[cam] = tracks
        kept = []
        for tid, rows in tracks.items():
            widest = max(r[3] - r[1] for r in rows)
            if len(rows) >= args.min_frames and widest >= args.min_width * width:
                kept.append((passing_frame(rows, width), tid))
        passings[cam] = sorted(kept)
        print(f"  {len(tracks)} raw tracks, {len(kept)} bottle passings", flush=True)

    # 4: time offset of each camera against the front camera (median of nearest differences)
    reference = np.array([f for f, _ in passings["front"]])
    offsets = {"front": 0}
    for cam in ["back_left", "back_right"]:
        diffs = [f - reference[np.argmin(np.abs(reference - f))] for f, _ in passings[cam]]
        offsets[cam] = int(np.median(diffs)) if diffs else 0
    print("Camera offsets vs front (frames):", offsets)

    # 5: merge all passings in time order; close passings = same bottle
    events = sorted((f - offsets[cam], cam, tid) for cam in CAMERAS for f, tid in passings[cam])
    window = args.window * fps
    global_of = {}           # (camera, track_id) -> global id
    bottles = []             # list of dicts
    for t, cam, tid in events:
        if not bottles or t - bottles[-1]["time"] > window:
            bottles.append({"id": len(bottles) + 1, "time": t, "cams": {}})
        bottle = bottles[-1]
        bottle["cams"].setdefault(cam, tid)
        global_of[(cam, tid)] = bottle["id"]

    with open(out / "bottles.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["global_id", "time_s", "seen_by", "front_track", "back_left_track", "back_right_track"])
        for b in bottles:
            w.writerow([b["id"], round(b["time"] / fps, 2), len(b["cams"])] +
                       [b["cams"].get(c, "") for c in CAMERAS])
    seen3 = sum(len(b["cams"]) == 3 for b in bottles)
    print(f"Bottles: {len(bottles)} (seen by all 3 cameras: {seen3})")
    print(f"Table: {out / 'bottles.csv'}")

    if args.no_video:
        return

    # 6: side-by-side demo video with the global IDs
    boxes = {cam: {} for cam in CAMERAS}   # cam -> frame -> list of (box, gid)
    for cam in CAMERAS:
        for tid, rows in all_tracks[cam].items():
            gid = global_of.get((cam, tid))
            if gid is None:
                continue
            for frame, x1, y1, x2, y2, c in rows:
                boxes[cam].setdefault(frame, []).append(((x1, y1, x2, y2), gid))

    caps = {cam: cv2.VideoCapture(videos[cam]) for cam in CAMERAS}
    tile_w, tile_h = 640, 360
    writer = cv2.VideoWriter(str(out / "sync_demo.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (tile_w * 3, tile_h + 40))
    frame = 0
    while True:
        tiles = []
        for cam in CAMERAS:
            ok, img = caps[cam].read()
            if not ok:
                break
            h, w = img.shape[:2]
            sx, sy = tile_w / w, tile_h / h
            img = cv2.resize(img, (tile_w, tile_h))
            for (x1, y1, x2, y2), gid in boxes[cam].get(frame, []):
                p1, p2 = (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy))
                cv2.rectangle(img, p1, p2, (0, 220, 0), 3)
                cv2.putText(img, f"ID {gid}", (p1[0] + 5, max(30, p1[1] + 35)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 0), 3)
            cv2.putText(img, cam, (10, tile_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            tiles.append(img)
        if len(tiles) < 3:
            break
        banner = np.zeros((40, tile_w * 3, 3), np.uint8)
        passed = sum(b["time"] <= frame for b in bottles)
        cv2.putText(banner, f"t = {frame / fps:6.1f} s    bottles passed: {passed}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(np.vstack([banner, np.hstack(tiles)]))
        frame += 1
    writer.release()
    print(f"Demo video: {out / 'sync_demo.mp4'}")


if __name__ == "__main__":
    main()
