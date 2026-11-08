import argparse
from pathlib import Path

import pandas as pd
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="folder with crops/, linked_labels.csv and ocr_results_*.csv")
    parser.add_argument("--classifier", default="runs/classify/bottle_cls_v1/weights/best.pt")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--engine", default="rapid")
    parser.add_argument("--expired-before", type=int, default=2025, help="a year below this means rejected")
    args = parser.parse_args()

    run = Path(args.run)
    ocr = pd.read_csv(run / f"ocr_results_{args.engine}.csv")
    labels = pd.read_csv(run / "linked_labels.csv").dropna(subset=["global_id", "excel_id"])
    status = dict(zip(labels.global_id.astype(int), labels.Primagaz_status))

    classifier = YOLO(args.classifier)
    nok_index = [k for k, v in classifier.names.items() if v == "nok"][0]

    rows = []
    for _, r in ocr.iterrows():
        scores = [float(classifier.predict(str(p), imgsz=224, verbose=False)[0].probs.data[nok_index])
                  for p in sorted((run / "crops").glob(f"bottle{int(r.global_id):04d}_*.jpg"))]
        visual = bool(scores) and max(scores) >= args.threshold
        expired = pd.notna(r.year_ocr) and int(r.year_ocr) < args.expired_before
        rows.append({"global_id": int(r.global_id), "operator": status[int(r.global_id)],
                     "visual_nok": visual, "expired": expired,
                     "decision": "NOK" if visual or expired else "OK"})
    out = pd.DataFrame(rows)
    out.to_csv(run / "push_fill.csv", index=False)

    nok = out[out.operator == "NOK"]
    ok = out[out.operator == "OK"]
    caught = (nok.decision == "NOK").sum()
    pushed = (ok.decision == "NOK").sum()
    print(f"Bottles: {len(out)}   operator NOK: {len(nok)}   operator OK: {len(ok)}")
    print(f"NOK caught: {caught}/{len(nok)}  (by classifier: {nok.visual_nok.sum()}, by OCR year: {nok.expired.sum()})")
    print(f"Good bottles pushed out: {pushed}/{len(ok)}  (by classifier: {ok.visual_nok.sum()}, by OCR year: {ok.expired.sum()})")
    if len(nok):
        print(f"Recall: {100 * caught / len(nok):.0f}%")
    if caught + pushed:
        print(f"Precision: {100 * caught / (caught + pushed):.0f}%")
    print(f"Saved {run / 'push_fill.csv'}")


if __name__ == "__main__":
    main()
