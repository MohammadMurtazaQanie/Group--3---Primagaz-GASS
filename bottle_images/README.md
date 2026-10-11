# Gas-bottle practice images

35 unlabelled LPG-cylinder images selected from Wikimedia Commons for an educational object-detection experiment. No bounding-box labels are included.

## Next step
Upload only the commons_*.jpg files into CVAT or Roboflow and draw a box around every identifiable gas bottle. Use one class: gas_bottle. Start with the clear single-bottle photos, then work through groups and partly hidden bottles. Include the bottle body, shoulder/guard and visible valve consistently. Do not label posters of bottles as physical bottles.

## Files
- commons_*.jpg: source images, generally served at up to 1280 pixels width (small originals may be smaller).
- SOURCES.md: authors, source pages, and licences.
- sources.json: provenance records, dimensions, and file hashes.
- Preview sheets are outside this folder so they cannot accidentally be uploaded as training images.

## Limitations for your report
This is a small, varied web-image sample, not a representative Primagaz test set. It includes product imagery as well as real-world scenes. Some crowded racks contain many small or occluded bottles and will take longer to label. The dataset is not sufficient to claim deployment readiness.

Before splitting the labelled dataset, group similar scenes and any photos of the same bottles together. In particular commons_90473360.jpg and commons_90473364.jpg belong to the same scene. Do not place these related photos in different training/validation/test splits. These grouping hints are not an exhaustive duplicate audit.

## Attribution
Licences vary by image (including CC BY, CC BY-SA, CC0, public domain and GODL-India). See SOURCES.md and each linked source page; do not describe this whole collection as CC BY 4.0. Keep attribution and applicable licence notices with shared images and derivatives.
