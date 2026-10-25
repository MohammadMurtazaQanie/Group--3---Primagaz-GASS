import argparse
import glob
import re
import time
from collections import Counter

import cv2
import easyocr
import pandas as pd


def read_year(text):
    years = re.findall(r"(?<!\d)(20[2-4]\d)(?!\d)", text)
    return Counter(years).most_common(1)[0][0] if years else None


def read_tare(text):
    text = text.replace(",", ".").replace("O", "0")
    found = []
    for a, b in re.findall(r"(?<![\d.])(1[0-7]|9)[. ]?(\d)(?=\s*[KG6C]|\b)", text):
        value = float(f"{a}.{b}")
        if 9.5 <= value <= 17.5:
            found.append(value)
    return Counter(found).most_common(1)[0][0] if found else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="folder with crops/ and linked_labels.csv")
    parser.add_argument("--limit", type=int, default=0, help="only the first N bottles (0 = all)")
    args = parser.parse_args()

    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    labels = pd.read_csv(f"{args.run}/linked_labels.csv").dropna(subset=["global_id", "excel_id"])
    if args.limit:
        labels = labels.head(args.limit)

    rows = []
    start = time.time()
    for _, r in labels.iterrows():
        gid = int(r.global_id)
        texts = []
        for path in sorted(glob.glob(f"{args.run}/crops/bottle{gid:04d}_*.jpg")):
            image = cv2.imread(path)
            if image.shape[1] < 600:
                scale = 600 / image.shape[1]
                image = cv2.resize(image, None, fx=scale, fy=scale)
            texts += [t for _, t, conf in reader.readtext(image) if conf > 0.2]
        text = " ".join(texts).upper()
        rows.append({
            "global_id": gid,
            "year_gt": str(r.year),
            "year_ocr": read_year(text),
            "tare_gt": str(r.tarra),
            "tare_ocr": read_tare(text),
            "ocr_text": text,
        })
        print(f"bottle {gid}: year {r.year} -> {rows[-1]['year_ocr']}, tare {r.tarra} -> {rows[-1]['tare_ocr']}")

    out = pd.DataFrame(rows)
    out.to_csv(f"{args.run}/ocr_results.csv", index=False)
    seconds = (time.time() - start) / max(len(out), 1)

    has_year = out.year_gt.str.match(r"^20\d\d")
    has_tare = out.tare_gt.str.match(r"^\d")
    year_ok = (out.year_ocr == out.year_gt.str[:4]) & has_year
    tare_ok = has_tare & out.apply(
        lambda x: x.tare_ocr is not None and abs(x.tare_ocr - float(x.tare_gt)) < 0.05 if x.tare_gt[:1].isdigit() else False, axis=1)
    year_wrong = out.year_ocr.notna() & has_year & ~year_ok
    tare_wrong = out.tare_ocr.notna() & has_tare & ~tare_ok

    print(f"\n{len(out)} bottles, {seconds:.1f} s per bottle")
    print(f"Year: correct {year_ok.sum()}/{has_year.sum()}, wrong {year_wrong.sum()}")
    print(f"Tare: correct {tare_ok.sum()}/{has_tare.sum()}, wrong {tare_wrong.sum()}")
    print(f"Saved {args.run}/ocr_results.csv")


if __name__ == "__main__":
    main()
