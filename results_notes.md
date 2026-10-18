# Experiment log

## Run 1 — baseline fine-tune (11 Oct 2026)
- Script: train_bottles.py (copy saved as runs/detect/gas_bottles/train_bottles_run1.py)
- Model: YOLO11n pretrained (COCO) -> 1 class `gas_bottle`
- Settings: epochs=30, imgsz=640, batch=4, seed=42, device=mps (Apple M1), optimizer auto -> AdamW lr=0.002
- Data: 25 train / 5 val images (233 / 85 boxes). Test set NOT used.
- Training time: 0.161 h (~10 min)
- Best epoch: 21 (best.pt)
- Validation (best.pt, 5 images, 85 boxes):
  - Precision 0.209 | Recall 0.259 | mAP50 0.137 | mAP50-95 0.0658
- Speed during final validation: 0.6 ms preprocess, 329.7 ms inference, 8.4 ms postprocess per image
- Observations:
  - Losses were still falling at epoch 30 -> likely undertrained.
  - Validation preview (val_batch0_pred.jpg): single cylinders are found, but with several
    duplicate/oversized boxes; crowded rack image (commons_13463479) mostly missed;
    small roadside cylinders (commons_58179965) given boxes shifted below the objects.
  - Validation metrics in results.csv repeat for consecutive epochs (e.g. epochs 2-3, 4-5),
    so validation appears not to have run every epoch; cause not confirmed.
  - MPS determinism warnings (scatter_reduce / index_put) were harmless.
- Results folder: runs/detect/gas_bottles

## Run 2 — longer training (11 Oct 2026)
- Only change from Run 1: epochs=100 (name=gas_bottles_e100). Everything else identical.
- Training time: ~15 min (MPS)
- Best epoch: 73 (best.pt). Values below are from results.csv for epoch 73; the final
  "Validating best.pt" line in the terminal should match closely.
- Validation (5 images, 85 boxes):
  - Precision 0.462 | Recall 0.247 | mAP50 0.230 | mAP50-95 0.111
- Comparison with Run 1: mAP50 0.137 -> 0.230, mAP50-95 0.066 -> 0.111, precision 0.209 -> 0.462,
  recall roughly unchanged (0.259 -> 0.247).
- Training losses kept falling to epoch 100 (cls_loss 3.83 -> 0.92), but validation peaked at
  epoch 73 and dropped slightly after -> early signs of overfitting on 25 training images.
- Validation preview (val_batch0_pred.jpg): far fewer duplicate boxes than Run 1.
  Single red cylinder found well (0.8). Grey cylinder: box covers only the top half.
  Crowded rack: only 2 of many cylinders found -> main weakness is crowded scenes (low recall).
  Building image: one likely false detection on a dark doorway/wall area next to the real cylinders.
- Decision: use Run 2 best.pt for the one-time test-set evaluation.

## Final test-set evaluation (Run 2 best.pt, used once, 11 Oct 2026)
- Script: eval_test.py. Results: runs/detect/test_eval and runs/detect/test_predictions
- Test (5 images, 30 boxes): Precision 0.704 | Recall 0.433 | mAP50 0.544 | mAP50-95 0.281
- Speed (test val, MPS): 1.2 ms preprocess, 77.5 ms inference, 2.2 ms postprocess per image
- Speed (predict, MPS): ~40 ms inference per image at 480x640
- Test is HIGHER than validation (mAP50 0.544 vs 0.230). With only 5 images per split this is
  mostly down to which images ended up where (validation has the very crowded rack photo with
  dozens of cylinders), NOT evidence that the model got better. Report both, with this caveat.
- Count per image (labelled -> predicted at conf 0.25):
  - commons_11881619 (crowded pallet): 14 -> 8. Several missed; some boxes merge two cylinders.
  - commons_180114158 (single worn cylinder): 1 -> 1. Found (0.73) but box extends far below the cylinder.
  - commons_39590510 (3 composite cylinders, white background): 3 -> 2. Yellow cylinder missed.
  - commons_6336975 (single orange cylinder): 1 -> 2. Duplicate box (0.59) on right half = false positive.
  - commons_92779676 (Butagaz store cages): 11 -> 25. Blue cylinders found well, but many extra/duplicate
    boxes on cage sections -> large over-count. Advertising picture of a bottle was correctly NOT detected.
  - Total: 30 labelled vs 38 predicted -> simple counting would be wrong in 4 of 5 images.
- Note: runs/detect/test_eval/val_batch*_labels.jpg render with corrupted colours (plotting glitch);
  the labels themselves are fine. Use runs/detect/test_predictions images in the report instead.
- Report figures copied to report_figures/

# Video tracking source
- bottle_video.mp4 = Getty Images preview clip 1214966660 (watermarked, 18.7 s, 25 fps, 768x432, 468 frames).
  Licence: NOT cleared for publishing. Use for internal experiment only; do not commit to GitHub.

## Bottle-video tracking (Run 2 best.pt + ByteTrack, 11 Oct 2026)
- Script: track_bottles.py. Output: runs/detect/bottle_track/bottle_video.mp4
- Video: delivery tricycle with ~10 red LPG cylinders (Bharatgas), camera follows then cart moves away.
- 468 frames processed. Detections per frame: min 0, max 5, average 1.9 (vs ~10 cylinders visible).
- Distinct track IDs: 21.
- Speed: 1.0 ms preprocess, 32.7 ms inference, 0.3 ms postprocess per frame at 384x640
  (eval_test.py log shows CPU; device not set in these scripts, so tracking likely ran on CPU too).
- Sampled frames (report_figures/bottle_track_frames_*.jpg):
  - SUCCESS: the cylinder lying on top kept ID 12 from ~3 s to ~15 s, despite camera motion and
    the cart shrinking into the distance. A rear cylinder kept ID 43 from ~7.5 s to 18 s.
  - MISSES: the upright cylinders (most of the load) were rarely detected, especially in the
    close-up first seconds -> detection, not tracking, is the main bottleneck. Close, partly cut-off,
    tightly packed cylinders are under-represented in our 25 training images.
  - FALSE DETECTIONS: worker's blue shirt/shoulder (ID 3, 1-3 s); at ~7.5 s one huge box over cart +
    worker (ID 39) and one on the worker's legs (ID 41).
  - 21 IDs for ~10 cylinders while only ~2 are found per frame: IDs come mostly from short-lived
    and false detections, so counting by unique IDs would be unreliable.
  - The Getty watermark lies over the cylinders and may reduce detections (not tested).

## Workshop 4 progress update (18 Oct 2026)
- Prepared `reports/Workshop_4_Progress_Report.docx` as the concise assignment submission.
- Added `track_multivideo.py`, a first synchronized multi-video prototype. It runs independent ByteTrack trackers for each camera, then attempts to associate local tracks with shared, monotonically allocated global IDs using HSV color histograms and bounding-box shape similarity. It writes annotated per-camera videos and a CSV track table.
- Prototype assumption: input files are already synchronized to the same start time and frame rate. The appearance matcher is heuristic and has not been validated; matches may be wrong, especially across different camera views.
- The local clone has no Primagaz station videos, extra 14_43 videos, or AWS per-bottle logfile. Current detector/tracker results are from general cylinder images and an unrelated stock clip, so they do not establish station performance.
- OK/NOK classification has not been trained. First define visible NOK reasons with the process expert and collect bottle-linked station images plus operator status and expert-reviewed labels.
- GitHub follow-up issues: https://github.com/MohammadMurtazaQanie/Group--3---Primagaz-GASS/issues/1, https://github.com/MohammadMurtazaQanie/Group--3---Primagaz-GASS/issues/2, https://github.com/MohammadMurtazaQanie/Group--3---Primagaz-GASS/issues/3.
- The Getty preview clip's publishing rights are uncleared; it is for internal experiments only and is not a public demo asset.
