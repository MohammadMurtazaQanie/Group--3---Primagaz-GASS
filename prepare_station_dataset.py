from pathlib import Path
import shutil

folder = Path(__file__).resolve().parent
frames = folder / "station_frames"
labels_1455 = next(folder.glob("task_2679834_*")) / "labels" / "train"
labels_1443 = next(folder.glob("task_2679840_*")) / "labels" / "train"
out = folder / "station_dataset"

if out.exists():
    shutil.rmtree(out)

images_1455 = sorted((frames / "14_55").glob("*.jpg"))
frame_numbers = sorted({int(p.stem.split("_")[-1]) for p in images_1455})
val_frames = set(frame_numbers[int(len(frame_numbers) * 0.8):])

plan = []
for image in images_1455:
    split = "val" if int(image.stem.split("_")[-1]) in val_frames else "train"
    plan.append((image, labels_1455, split))
for image in sorted((frames / "14_43").glob("*.jpg")):
    plan.append((image, labels_1443, "test"))

counts = {}
for image, label_dir, split in plan:
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, out / "images" / split / image.name)
    label = label_dir / f"{image.stem}.txt"
    target = out / "labels" / split / f"{image.stem}.txt"
    if label.exists():
        shutil.copy2(label, target)
    else:
        target.write_text("")  
    boxes = len(target.read_text().splitlines())
    n_img, n_box = counts.get(split, (0, 0))
    counts[split] = (n_img + 1, n_box + boxes)

(out / "data.yaml").write_text(
    f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\n"
    "names:\n  0: gas_bottle\n"
)
for split in ("train", "val", "test"):
    print(f"{split}: {counts[split][0]} images, {counts[split][1]} boxes")
print("Dataset written to", out)
