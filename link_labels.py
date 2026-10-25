import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def crop_colour(path):
    image = cv2.imread(str(path))
    h, w = image.shape[:2]
    centre = image[int(h * 0.3):int(h * 0.9), int(w * 0.2):int(w * 0.8)]
    hsv = cv2.cvtColor(centre, cv2.COLOR_BGR2HSV)
    cyan = (hsv[..., 0] > 70) & (hsv[..., 0] < 105) & (hsv[..., 1] > 80) & (hsv[..., 2] > 60)
    return cyan.mean()


def excel_colour(value):
    value = str(value).lower()
    return "cyan" if "cyan" in value else ("black" if "black" in value else "other")


def align(ours, theirs, our_min, their_min):
    match, mismatch, gap, forbidden = 2, -3, -2, -1e6
    n, m = len(ours), len(theirs)

    def pair_score(i, j):
        if abs(our_min[i] - their_min[j]) > 1:
            return forbidden
        return match if ours[i] == theirs[j] else mismatch

    score = np.zeros((n + 1, m + 1))
    score[:, 0] = np.arange(n + 1) * gap
    score[0, :] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score[i, j] = max(score[i - 1, j - 1] + pair_score(i - 1, j - 1),
                              score[i - 1, j] + gap, score[i, j - 1] + gap)
    j = int(np.argmax(score[n, :]))
    pairs = [(None, k) for k in range(m - 1, j - 1, -1)]
    i = n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and score[i, j] == score[i - 1, j - 1] + pair_score(i - 1, j - 1):
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and score[i, j] == score[i - 1, j] + gap:
            pairs.append((i - 1, None)); i -= 1
        elif i == 0:
            pairs.append((None, j - 1)); j -= 1
        else:
            pairs.append((None, j - 1)); j -= 1
    return pairs[::-1]


def estimate_start(times_s, truth_minutes, clip_start):
    first = truth_minutes.min()
    excel_counts = truth_minutes.value_counts()
    best = None
    for start in range(0, 120):
        minutes = first + ((times_s + clip_start + start) // 60).astype(int)
        ours = minutes.value_counts()
        error = sum(abs(ours.get(k, 0) - excel_counts.get(k, 0)) for k in set(ours.index) | set(excel_counts.index)
                    if minutes.min() <= k <= minutes.max())
        if best is None or error < best[0]:
            best = (error, start)
    return best[1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="output folder of track_sync.py (has bottles.csv and crops/)")
    p.add_argument("--excel", required=True, help="ground-truth Excel of the same session")
    p.add_argument("--clip-start", type=float, default=0,
                   help="if the videos are a clip: seconds into the full recording where the clip starts")
    p.add_argument("--video-start", type=int, default=None,
                   help="seconds after the first Excel minute at which the FULL recording starts "
                        "(printed when linking the full video, e.g. 33 for 14_43); estimated if not given")
    args = p.parse_args()
    run = Path(args.run)

    bottles = pd.read_csv(run / "bottles.csv")
    crop_files = {gid: sorted((run / "crops").glob(f"bottle{gid:04d}_*.jpg")) for gid in bottles.global_id}
    bottles["cyan_share"] = [np.mean([crop_colour(f) for f in crop_files[g]]) if crop_files[g] else 0.0
                             for g in bottles.global_id]
    bottles["colour"] = np.where(bottles.cyan_share > 0.25, "cyan", "black")

    truth = pd.read_excel(args.excel).dropna(subset=["id"]).reset_index(drop=True)
    truth["colour_gt"] = truth["color"].map(excel_colour)

    truth_minutes = truth["Time"].dt.hour * 60 + truth["Time"].dt.minute
    start = args.video_start if args.video_start is not None else \
        estimate_start(bottles.time_s, truth_minutes, args.clip_start)
    our_minutes = (truth_minutes.min() + (bottles.time_s + args.clip_start + start) // 60).astype(int)
    print(f"Video estimated to start {start} s after {int(truth_minutes.min()) // 60}:{int(truth_minutes.min()) % 60:02d}")
    pairs = align(list(bottles.colour), list(truth.colour_gt), list(our_minutes), list(truth_minutes))
    rows = []
    for i, j in pairs:
        row = {}
        if i is not None:
            row.update(global_id=int(bottles.global_id[i]), colour=bottles.colour[i],
                       cyan_share=round(float(bottles.cyan_share[i]), 3))
            if "decision" in bottles:
                row.update(p_nok=bottles.p_nok[i], decision=bottles.decision[i])
        if j is not None:
            t = truth.iloc[j]
            row.update(excel_id=int(t["id"]), colour_gt=t.colour_gt, Primagaz_status=t["Primagaz_status"],
                       classification=t["classification"], year=t["year"], tarra=t["tarra"])
        rows.append(row)
    linked = pd.DataFrame(rows)
    linked.to_csv(run / "linked_labels.csv", index=False)

    both = linked.dropna(subset=["global_id", "excel_id"])
    print(f"Tracked bottles: {len(bottles)}   Excel bottles: {len(truth)}")
    print(f"Linked: {len(both)}   only tracked: {linked.excel_id.isna().sum()}   only in Excel: {linked.global_id.isna().sum()}")
    print(f"Colour agreement on linked bottles: {(both.colour == both.colour_gt).mean():.0%}")
    print(f"Linked bottles that are NOK (operator): {(both.Primagaz_status == 'NOK').sum()}")
    if "decision" in both:
        truth_nok = both.classification.astype(str).str.lower().eq("nok")
        said_nok = both.decision.eq("NOK")
        tp, fn = int((truth_nok & said_nok).sum()), int((truth_nok & ~said_nok).sum())
        fp, tn = int((~truth_nok & said_nok).sum()), int((~truth_nok & ~said_nok).sum())
        print("Classifier vs visual ground truth (per bottle):")
        print(f"  visible NOK caught {tp}/{tp + fn}, good bottles pushed out {fp}/{fp + tn}")
        op_nok = both.Primagaz_status.eq("NOK")
        print(f"  operator NOK bottles flagged by the classifier: {int((op_nok & said_nok).sum())}/{int(op_nok.sum())}"
              " (most operator rejections are not visible: tare/year -> OCR)")
    print(f"Table: {run / 'linked_labels.csv'}")


if __name__ == "__main__":
    main()
