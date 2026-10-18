"""Prototype synchronized multi-camera bottle tracking with shared global IDs.

Inputs must be pre-synchronized to the same start time and frame rate. Each
camera keeps its own ByteTrack state; local tracks are associated across views
using a simple HSV appearance histogram and box-shape score. This heuristic is
for an initial experiment and requires validation on station footage.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Hashable

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class GlobalTrack:
    global_id: int
    appearance: np.ndarray
    aspect_ratio: float
    last_seen_seconds: float
    seen_frame: int = -1
    seen_cameras: set[str] = field(default_factory=set)


def parse_video_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("Use CAMERA=VIDEO_PATH, for example cam1=video1.mp4")
    camera, raw_path = spec.split("=", 1)
    camera, raw_path = camera.strip(), raw_path.strip()
    if not camera or not raw_path:
        raise argparse.ArgumentTypeError("Camera label and video path must both be non-empty")
    return camera, Path(raw_path).expanduser()


def appearance_descriptor(frame: np.ndarray, xyxy: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(float(value))) for value in xyxy)
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    box_width, box_height = max(1, x2 - x1), max(1, y2 - y1)
    # Center crop reduces background from loose detector boxes.
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    crop_width, crop_height = max(1, int(box_width * 0.62)), max(1, int(box_height * 0.72))
    left, right = max(x1, cx - crop_width // 2), min(x2, cx + crop_width // 2)
    top, bottom = max(y1, cy - crop_height // 2), min(y2, cy + crop_height // 2)
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [24, 12], [0, 180, 30, 256])
    cv2.normalize(histogram, histogram, alpha=1.0, norm_type=cv2.NORM_L1)
    aspect_ratio = box_width / box_height
    return histogram, aspect_ratio


def appearance_similarity(track: GlobalTrack, histogram: np.ndarray, aspect_ratio: float) -> float:
    color_similarity = 1.0 - float(cv2.compareHist(track.appearance, histogram, cv2.HISTCMP_BHATTACHARYYA))
    shape_similarity = math.exp(-abs(math.log(max(1e-6, track.aspect_ratio) / max(1e-6, aspect_ratio))))
    return 0.8 * color_similarity + 0.2 * shape_similarity


def camera_filename(camera: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", camera).strip("_") or "camera"
    return f"{safe}_tracked.mp4"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", action="append", type=parse_video_spec, required=True,
                        metavar="CAMERA=PATH", help="Repeat once per synchronized camera")
    parser.add_argument("--model", type=Path,
                        default=Path(__file__).parent / "runs/detect/gas_bottles_e100/weights/best.pt")
    parser.add_argument("--output", type=Path, default=Path("runs/detect/multivideo"))
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--match-threshold", type=float, default=0.72,
                        help="Minimum appearance/shape score for reusing a global ID")
    parser.add_argument("--max-gap-seconds", type=float, default=30.0,
                        help="Forget appearance matches after this many seconds without a sighting")
    args = parser.parse_args()

    cameras = [camera for camera, _ in args.video]
    paths = [path.resolve() for _, path in args.video]
    if len(set(cameras)) != len(cameras):
        parser.error("Camera labels must be unique")
    if not args.model.is_file():
        parser.error(f"Model weights do not exist: {args.model}")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        parser.error("Video file(s) not found: " + ", ".join(missing))

    captures = [cv2.VideoCapture(str(path)) for path in paths]
    if any(not cap.isOpened() for cap in captures):
        for cap in captures:
            cap.release()
        raise RuntimeError("Could not open every video")
    fps_values = [cap.get(cv2.CAP_PROP_FPS) for cap in captures]
    reference_fps = fps_values[0]
    if reference_fps <= 0 or any(fps <= 0 or abs(fps - reference_fps) / reference_fps > 0.01 for fps in fps_values):
        for cap in captures:
            cap.release()
        raise ValueError(f"Videos must have the same valid frame rate; found {fps_values}")

    args.output.mkdir(parents=True, exist_ok=True)
    writers: dict[str, cv2.VideoWriter] = {}
    models = [YOLO(str(args.model)) for _ in cameras]
    global_tracks: dict[int, GlobalTrack] = {}
    local_to_global: dict[tuple[str, Hashable], int] = {}
    next_global_id = 1
    frame_index = 0
    csv_path = args.output / "tracks.csv"

    try:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["frame", "time_seconds", "camera", "tracker_id", "global_id",
                             "confidence", "x1", "y1", "x2", "y2"])
            while True:
                frames: dict[str, np.ndarray] = {}
                for camera, cap in zip(cameras, captures):
                    ok, frame = cap.read()
                    if ok:
                        frames[camera] = frame
                if not frames:
                    break

                timestamp = frame_index / reference_fps
                observations = []
                # Each camera owns an independent tracker; mapping is shared.
                for index, camera in enumerate(cameras):
                    frame = frames.get(camera)
                    if frame is None:
                        continue
                    result = models[index].track(
                        frame, persist=True, tracker=args.tracker,
                        conf=args.confidence, verbose=False,
                    )[0]
                    boxes = result.boxes
                    if boxes is None or len(boxes) == 0:
                        detections = []
                    else:
                        xyxy = boxes.xyxy.cpu().numpy()
                        confidences = boxes.conf.cpu().numpy()
                        tracker_ids = boxes.id
                        local_ids = tracker_ids.cpu().numpy() if tracker_ids is not None else None
                        detections = []
                        for det_index, box in enumerate(xyxy):
                            local_id: Hashable = (int(local_ids[det_index]) if local_ids is not None
                                                  else f"frame{frame_index}_det{det_index}")
                            histogram, aspect_ratio = appearance_descriptor(frame, box)
                            detections.append((local_id, box, float(confidences[det_index]), histogram, aspect_ratio))
                    observations.append((camera, frame, detections))

                # Clear per-frame camera visibility before matching this synchronized batch.
                for track in global_tracks.values():
                    if track.seen_frame != frame_index:
                        track.seen_frame = frame_index
                        track.seen_cameras.clear()

                for camera, frame, detections in observations:
                    annotated = frame.copy()
                    for local_id, box, confidence, histogram, aspect_ratio in detections:
                        mapping_key = (camera, local_id)
                        global_id = local_to_global.get(mapping_key)
                        if global_id is not None:
                            track = global_tracks[global_id]
                            # Avoid assigning one physical ID to two simultaneous detections in a camera.
                            if camera in track.seen_cameras:
                                global_id = None

                        if global_id is None:
                            candidates = []
                            for candidate_id, candidate in global_tracks.items():
                                if timestamp - candidate.last_seen_seconds > args.max_gap_seconds:
                                    continue
                                if camera in candidate.seen_cameras:
                                    continue
                                score = appearance_similarity(candidate, histogram, aspect_ratio)
                                if score >= args.match_threshold:
                                    candidates.append((score, candidate_id))
                            if candidates:
                                _, global_id = max(candidates)
                            else:
                                global_id = next_global_id
                                next_global_id += 1
                                global_tracks[global_id] = GlobalTrack(
                                    global_id, histogram.copy(), aspect_ratio, timestamp,
                                    seen_frame=frame_index,
                                )
                            local_to_global[mapping_key] = global_id

                        track = global_tracks[global_id]
                        if track.seen_frame != frame_index:
                            track.seen_frame = frame_index
                            track.seen_cameras.clear()
                        if camera not in track.seen_cameras:
                            track.appearance = 0.7 * track.appearance + 0.3 * histogram
                            cv2.normalize(track.appearance, track.appearance, alpha=1.0, norm_type=cv2.NORM_L1)
                            track.aspect_ratio = 0.7 * track.aspect_ratio + 0.3 * aspect_ratio
                            track.last_seen_seconds = timestamp
                            track.seen_cameras.add(camera)
                        else:
                            # A duplicate same-camera detection cannot safely share a global ID.
                            global_id = next_global_id
                            next_global_id += 1
                            local_to_global[mapping_key] = global_id
                            global_tracks[global_id] = GlobalTrack(
                                global_id, histogram.copy(), aspect_ratio, timestamp,
                                seen_frame=frame_index, seen_cameras={camera},
                            )

                        x1, y1, x2, y2 = (int(round(float(v))) for v in box)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (20, 180, 90), 2)
                        cv2.putText(annotated, f"ID {global_id}  {confidence:.2f}",
                                    (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (20, 180, 90), 2, cv2.LINE_AA)
                        writer.writerow([frame_index, f"{timestamp:.4f}", camera, local_id,
                                         global_id, f"{confidence:.4f}", x1, y1, x2, y2])

                    if camera not in writers:
                        height, width = annotated.shape[:2]
                        output_file = args.output / camera_filename(camera)
                        writers[camera] = cv2.VideoWriter(
                            str(output_file), cv2.VideoWriter_fourcc(*"mp4v"),
                            reference_fps, (width, height),
                        )
                        if not writers[camera].isOpened():
                            raise RuntimeError(f"Could not create output video: {output_file}")
                    writers[camera].write(annotated)
                frame_index += 1
    finally:
        for cap in captures:
            cap.release()
        for writer in writers.values():
            writer.release()

    print(f"Frames processed: {frame_index}")
    print(f"Global IDs allocated: {next_global_id - 1}")
    print(f"Track table: {csv_path}")
    print(f"Annotated videos: {args.output}")


if __name__ == "__main__":
    main()
