"""Build the OK/NOK classification dataset.

Every image is a bottle crop, exactly as track_sync.py cuts it from the video
(this is also how the classifier will see bottles in the final application).

Label = the lector's visual 'classification' column in the ground-truth Excel
(nok = big visual damage, really dirty, or a different/old bottle type).
Bottles pushed out for other reasons (tare weight, year) stay 'ok' here,
because a picture classifier cannot see those; that is the OCR's job.

Splits (always by bottle, never by single image, so the three camera views of
one bottle are never split over train and test):
  train / val : the other session (every 5th bottle -> val)
                + the plant NOK photos (80% train, 20% val), bottle cut out with our detector
  test        : --test-session (default 14_43), a different recording, used once at the end
NOK images in train are copied 3x, because NOK bottles are rare.
"""
import argparse
import csv
import random
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO

folder = Path(__file__).resolve().parent
runs = folder / "runs" / "sync"
plant_dir = folder / "images nok bottles taken at primagaz plant"
detector = folder / "runs" / "detect" / "station_v1" / "weights" / "best.pt"
out = folder / "cls_dataset"
SIZE = 320          # crops are stored with their longest side at 320 px
NOK_COPIES = 3      # oversampling of NOK in train
random.seed(42)


def save(image, split, label, name):
    target = out / split / label
    target.mkdir(parents=True, exist_ok=True)
    h, w = image.shape[:2]
    scale = SIZE / max(h, w)
    image = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))))
    copies = NOK_COPIES if (split == "train" and label == "nok") else 1
    for k in range(copies):
        cv2.imwrite(str(target / (name if k == 0 else name.replace(".jpg", f"_copy{k}.jpg"))), image)


def video_bottles(session):
    """Yield (global_id, label, [crop paths]) for every bottle linked to the Excel."""
    run = runs / f"full_{session}"
    for row in csv.DictReader(open(run / "linked_labels.csv")):
        if not row["global_id"] or not row["excel_id"]:
            continue
        gid = int(float(row["global_id"]))
        label = "nok" if row["classification"].strip().lower() == "nok" else "ok"
        yield gid, label, sorted((run / "crops").glob(f"bottle{gid:04d}_*.jpg"))


parser = argparse.ArgumentParser()
parser.add_argument("--test-session", default="14_43", choices=["14_43", "14_55"],
                    help="recording kept apart for the final test; the other one is used for train/val")
parser.add_argument("--no-plant", action="store_true", help="leave out the plant NOK photos (experiment)")
args = parser.parse_args()
train_session = "14_55" if args.test_session == "14_43" else "14_43"

if out.exists():
    shutil.rmtree(out)

# 1. video crops
for session, split_rule in [(train_session, "trainval"), (args.test_session, "test")]:
    for gid, label, crops in video_bottles(session):
        split = "test" if split_rule == "test" else ("val" if gid % 5 == 0 else "train")
        for crop in crops:
            save(cv2.imread(str(crop)), split, label, f"{session}_{crop.name}")

# 2. plant NOK photos: cut out the biggest detected bottle, like the video crops
if args.no_plant:
    plant_dir = Path("/nonexistent")
#    The same bottle was often photographed several times in a row, so photos
#    taken less than 30 s apart form one group, and whole groups go to train or val.
from datetime import datetime
model = YOLO(str(detector))
photos = sorted(plant_dir.glob("*.jpg"))
groups, last = [], None
for photo in photos:
    try:
        t = datetime.strptime(photo.stem[:15], "%Y%m%d_%H%M%S")
    except ValueError:
        t = None
    if not groups or t is None or last is None or (t - last).total_seconds() > 30:
        groups.append([])
    groups[-1].append(photo)
    last = t
random.shuffle(groups)
val_groups = set()
count = 0
for g_index, group in enumerate(groups):          # first ~20% of photos (whole groups) -> val
    if count < 0.2 * len(photos):
        val_groups.add(g_index)
        count += len(group)
used = 0
for g_index, group in enumerate(groups):
  for photo in group:
    image = cv2.imread(str(photo))
    result = model.predict(image, conf=0.3, verbose=False)[0]
    if len(result.boxes) == 0:
        continue
    areas = (result.boxes.xyxy[:, 2] - result.boxes.xyxy[:, 0]) * (result.boxes.xyxy[:, 3] - result.boxes.xyxy[:, 1])
    x1, y1, x2, y2 = (int(v) for v in result.boxes.xyxy[int(areas.argmax())].tolist())
    split = "val" if g_index in val_groups else "train"
    save(image[max(0, y1):y2, max(0, x1):x2], split, "nok", f"plant_{photo.name}")
    used += 1
print(f"Plant photos with a detected bottle: {used} of {len(photos)} ({len(groups)} photo groups)")

for split in ("train", "val", "test"):
    counts = {label: len(list((out / split / label).glob("*.jpg"))) for label in ("ok", "nok")}
    print(f"{split:5s}: ok {counts['ok']:4d}   nok {counts['nok']:4d}")
print("Dataset written to", out)
