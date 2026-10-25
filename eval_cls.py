import argparse
import re
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO

folder = Path(__file__).resolve().parent


def bottle_key(path):
    name = re.sub(r"_copy\d+", "", path.stem)
    match = re.match(r"(\d+_\d+)_bottle(\d+)_", name)
    return f"{match.group(1)}_{match.group(2)}" if match else name


def nok_scores(model, split):
    per_image, per_bottle = [], defaultdict(lambda: [None, 0.0])
    for label in ("ok", "nok"):
        files = sorted(p for p in (folder / "cls_dataset" / split / label).glob("*.jpg") if "_copy" not in p.stem)
        for start in range(0, len(files), 64):
            batch = files[start:start + 64]
            for path, result in zip(batch, model.predict([str(p) for p in batch], verbose=False)):
                p_nok = float(result.probs.data[result.names_inv["nok"]]) if hasattr(result, "names_inv") \
                    else float(result.probs.data[[k for k, v in result.names.items() if v == "nok"][0]])
                per_image.append((label, p_nok))
                key = bottle_key(path)
                per_bottle[key][0] = label
                per_bottle[key][1] = max(per_bottle[key][1], p_nok)
    return per_image, {k: tuple(v) for k, v in per_bottle.items()}


def report(name, items, threshold):
    tp = sum(1 for label, s in items if label == "nok" and s >= threshold)
    fn = sum(1 for label, s in items if label == "nok" and s < threshold)
    fp = sum(1 for label, s in items if label == "ok" and s >= threshold)
    tn = sum(1 for label, s in items if label == "ok" and s < threshold)
    recall = tp / (tp + fn) if tp + fn else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    print(f"  {name:12s} NOK recall {recall:5.0%}  NOK precision {precision:5.0%}   "
          f"(NOK caught {tp}/{tp + fn}, good bottles pushed out {fp}/{fp + tn})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(folder / "runs/classify/bottle_cls_v1/weights/best.pt"))
    args = p.parse_args()
    model = YOLO(args.model)

    test_images, test_bottles = nok_scores(model, "test")
    session = next(iter(test_bottles))[:5]
    nok_scores_test = [s for label, s in test_bottles.values() if label == "nok"]
    full_recall = min(nok_scores_test) if nok_scores_test else 0.5
    for t, title in [(0.5, "standard threshold 0.5"),
                     (full_recall, f"threshold {full_recall:.3f} = lowest that catches every NOK bottle")]:
        print(f"Test set (session {session}), {title}:")
        report("per image", test_images, t)
        report("per bottle", list(test_bottles.values()), t)


if __name__ == "__main__":
    main()
