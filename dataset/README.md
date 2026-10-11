# Dataset preparation

Source: 35 Wikimedia Commons images, manually annotated in CVAT by the student.
Class: gas_bottle. Total bounding boxes: 348.

| Split | Images | Boxes |
|---|---:|---:|
| train | 25 | 233 |
| val | 5 | 85 |
| test | 5 | 30 |

Split uses seed 42 and balances single-bottle, small-group and crowded images. The known blue-cylinder image pair remains in training. No exact duplicate file crosses a split boundary. Near-duplicate screening was visual, not exhaustive.

Training updates model weights. Validation guides model selection. Test images are reserved for final evaluation and must not be used to choose settings.

Five validation and five test images yield very uncertain performance estimates. These images do not establish Primagaz deployment readiness. All 35 labels passed numeric format, class and coordinate checks; this does not guarantee every annotation is semantically correct.

Original images and the CVAT export are preserved. The dataset contains copies. Image credit records, if retained, are in bottle_images/SOURCES.md.

If moving the project, update the path entry in data.yaml to its new absolute location.
