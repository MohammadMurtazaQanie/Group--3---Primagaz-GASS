import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import cv2
import pandas as pd


def clean(text):
    text = text.upper().replace(",", ".")
    return re.sub(r"(?<=\d)O|O(?=\d)", "0", text)


def find_years(text):
    return re.findall(r"(?<!\d)(20[2-4]\d)(?!\d)", clean(text))


def find_tares(text):
    if re.search(r"\d\s*k", text):
        return []
    text = clean(text).strip(" '\"-_|")
    patterns = [r"^(\d{2})\.(\d)(?:\s*K.*)?$",
                r"(?<![\d.])(\d{2})\.?(\d)\d?\s*K[G6C]",
                r"TAR[EA]\s*(\d{2})\.?(\d)"]
    found = []
    for pattern in patterns:
        for a, b in re.findall(pattern, text):
            value = float(f"{a}.{b}")
            if 9 <= value <= 20:
                found.append(value)
    return found


def vote(values):
    return Counter(values).most_common(1)[0][0] if values else None


def make_reader(engine):
    if engine == "rapid":
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()

        def read(image):
            result, _ = ocr(image)
            return [(text, float(conf)) for _, text, conf in (result or [])]
    else:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=True, verbose=False)

        def read(image):
            return [(text, float(conf)) for _, text, conf in reader.readtext(image)]
    return read


def judge(truth, read, kind):
    truth = str(truth).strip()
    if kind == "year":
        if not re.fullmatch(r"20\d\d", truth):
            return "no ground truth"
        return "not read" if read is None else ("correct" if read == truth else "wrong")
    if not re.fullmatch(r"\d+(\.\d+)?", truth):
        return "no ground truth"
    first_decimal = int(float(truth) * 10 + 1e-6) / 10
    return "not read" if read is None else ("correct" if abs(read - first_decimal) < 0.01 else "wrong")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="folder with ocr_crops/ and linked_labels.csv")
    parser.add_argument("--engine", default="rapid", choices=["rapid", "easyocr"])
    parser.add_argument("--every", type=int, default=2, help="read every Nth saved crop per camera")
    parser.add_argument("--limit", type=int, default=0, help="only the first N bottles (0 = all)")
    args = parser.parse_args()

    run = Path(args.run)
    cache_file = run / f"ocr_texts_{args.engine}.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    reader = None
    labels = pd.read_csv(run / "linked_labels.csv").dropna(subset=["global_id", "excel_id"])
    if args.limit:
        labels = labels.head(args.limit)

    rows = []
    crops_read = 0
    start = time.time()
    for _, r in labels.iterrows():
        gid = int(r.global_id)
        by_camera = {}
        for path in sorted((run / "ocr_crops").glob(f"bottle{gid:04d}_*.jpg")):
            by_camera.setdefault(path.stem.rsplit("_", 1)[0], []).append(path)
        texts = []
        for paths in by_camera.values():
            for path in paths[::args.every]:
                if path.name not in cache:
                    if reader is None:
                        reader = make_reader(args.engine)
                    image = cv2.imread(str(path))
                    h = image.shape[0]
                    cache[path.name] = reader(image[int(0.35 * h):int(0.8 * h)])
                texts += [t for t, conf in cache[path.name] if conf > 0.2]
                crops_read += 1
        year = vote([y for t in texts for y in find_years(t)])
        tare = vote([v for t in texts for v in find_tares(t)])
        rows.append({"global_id": gid, "excel_id": int(r.excel_id),
                     "year_gt": r.year, "year_ocr": year, "year_result": judge(r.year, year, "year"),
                     "tare_gt": r.tarra, "tare_ocr": tare, "tare_result": judge(r.tarra, tare, "tare"),
                     "text": " | ".join(texts)[:300]})
        print(f"bottle {gid}: year {r.year} -> {year}, tare {r.tarra} -> {tare}", flush=True)

    cache_file.write_text(json.dumps(cache))
    out = pd.DataFrame(rows)
    out.to_csv(run / f"ocr_results_{args.engine}.csv", index=False)
    seconds = time.time() - start

    print(f"\nEngine {args.engine}: {len(out)} bottles, {crops_read} crops, "
          f"{seconds / max(len(out), 1):.1f} s per bottle")
    print(f"{'':6}{'correct':>9}{'wrong':>7}{'not read':>10}{'scored':>8}{'accuracy':>10}{'precision':>11}"
          f"{'no ground truth':>17}")
    for kind in ["year", "tare"]:
        counts = out[f"{kind}_result"].value_counts()
        c, w, n = counts.get("correct", 0), counts.get("wrong", 0), counts.get("not read", 0)
        scored = c + w + n
        accuracy = f"{100 * c / scored:.0f}%" if scored else "-"
        precision = f"{100 * c / (c + w):.0f}%" if c + w else "-"
        print(f"{kind.capitalize():6}{c:>9}{w:>7}{n:>10}{scored:>8}{accuracy:>10}{precision:>11}"
              f"{counts.get('no ground truth', 0):>17}")
    print(f"Saved {run / f'ocr_results_{args.engine}.csv'}")


if __name__ == "__main__":
    main()
