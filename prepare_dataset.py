"""Validate the CVAT export and prepare a reproducible YOLO dataset."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import random
import shutil

ROOT = Path(__file__).resolve().parent


def main():
    images = ROOT / 'bottle_images'
    export = ROOT / 'gas_bottle_labels'
    output = ROOT / 'dataset'
    if output.exists():
        raise SystemExit('dataset already exists. Keep it unchanged to preserve the experiment split.')

    import yaml
    config = yaml.safe_load((export / 'data.yaml').read_text())
    if config['names'] not in ({0: 'gas_bottle'}, ['gas_bottle']):
        raise ValueError('Expected one class: 0 = gas_bottle')
    labels = sorted((export / 'labels').rglob('*.txt'))
    records = {}
    for label in labels:
        image = images / f'{label.stem}.jpg'
        if not image.is_file() or label.stem in records:
            raise ValueError(f'Missing image or duplicate label: {label.name}')
        lines = [line for line in label.read_text().splitlines() if line.strip()]
        for line in lines:
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f'Invalid YOLO row in {label.name}')
            cls, x, y, w, h = map(float, fields)
            if cls != 0 or not all(math.isfinite(v) and 0 <= v <= 1 for v in (x,y,w,h)) or w <= 0 or h <= 0:
                raise ValueError(f'Invalid coordinates or class in {label.name}')
            if min(x-w/2,y-h/2) < -1e-5 or max(x+w/2,y+h/2) > 1+1e-5:
                raise ValueError(f'Box extends outside image in {label.name}')
        records[label.stem] = {'image': image, 'label': label, 'boxes': len(lines), 'sha256': hashlib.sha256(image.read_bytes()).hexdigest()}
    expected = {p.stem for p in images.glob('commons_*.jpg')}
    if set(records) != expected or len(records) != 35:
        raise ValueError('Expected matching images and labels for all 35 source images')

    # Known related images remain in training, so their shared scene cannot leak
    # into validation or testing. Grouping is based on visual source inspection.
    related = {'commons_90473360', 'commons_90473364'}
    if not related <= records.keys():
        raise ValueError('Known related image pair missing')
    strata = {'single': [], 'small_group': [], 'crowded': []}
    for name, record in sorted(records.items()):
        if name in related:
            continue
        count = record['boxes']
        key = 'single' if count <= 1 else ('small_group' if count <= 5 else 'crowded')
        strata[key].append(name)
    rng = random.Random(42)
    splits = {'train': sorted(related), 'val': [], 'test': []}
    for category, take in [('single',2),('small_group',1),('crowded',2)]:
        candidates = strata[category]
        rng.shuffle(candidates)
        if len(candidates) < 2*take:
            raise ValueError(f'Insufficient images for {category}')
        splits['val'] += candidates[:take]
        splits['test'] += candidates[take:2*take]
        splits['train'] += candidates[2*take:]
    hashes = {}
    for split, names in splits.items():
        for name in names:
            digest = records[name]['sha256']
            if digest in hashes and hashes[digest] != split:
                raise ValueError('An exact duplicate image would cross the split boundary')
            hashes[digest] = split
    if [len(splits[k]) for k in ('train','val','test')] != [25,5,5]:
        raise ValueError('Unexpected split sizes')

    manifest = {'seed':42, 'class_names':{'0':'gas_bottle'}, 'related_group_kept_in_train':sorted(related), 'splits':{}}
    for split, names in splits.items():
        (output/'images'/split).mkdir(parents=True)
        (output/'labels'/split).mkdir(parents=True)
        manifest['splits'][split] = []
        for name in sorted(names):
            r=records[name]
            shutil.copy2(r['image'],output/'images'/split/r['image'].name)
            shutil.copy2(r['label'],output/'labels'/split/r['label'].name)
            manifest['splits'][split].append({'image':r['image'].name,'boxes':r['boxes'],'sha256':r['sha256']})
    # Absolute root makes this configuration usable regardless of the shell cwd.
    (output/'data.yaml').write_text(yaml.safe_dump({'path':str(output),'train':'images/train','val':'images/val','test':'images/test','names':{0:'gas_bottle'}},sort_keys=False))
    (output/'split_manifest.json').write_text(json.dumps(manifest,indent=2))
    summary=['# Dataset preparation', '', 'Source: 35 Wikimedia Commons images, manually annotated in CVAT by the student.', 'Class: gas_bottle. Total bounding boxes: '+str(sum(r['boxes'] for r in records.values()))+'.', '', '| Split | Images | Boxes |','|---|---:|---:|']
    for split in splits:
        summary.append(f"| {split} | {len(splits[split])} | {sum(records[n]['boxes'] for n in splits[split])} |")
    summary += ['', 'Split uses seed 42 and balances single-bottle, small-group and crowded images. The known blue-cylinder image pair remains in training. No exact duplicate file crosses a split boundary. Near-duplicate screening was visual, not exhaustive.', '', 'Training updates model weights. Validation guides model selection. Test images are reserved for final evaluation and must not be used to choose settings.', '', 'Five validation and five test images yield very uncertain performance estimates. These images do not establish Primagaz deployment readiness. All 35 labels passed numeric format, class and coordinate checks; this does not guarantee every annotation is semantically correct.', '', 'Original images and the CVAT export are preserved. The dataset contains copies. Image credit records, if retained, are in bottle_images/SOURCES.md.', '', 'If moving the project, update the path entry in data.yaml to its new absolute location.']
    (output/'README.md').write_text('\n'.join(summary)+'\n')
    print('\n'.join(summary[:11]))
    print(f'\nReady: {output / "data.yaml"}')


if __name__ == '__main__':
    main()
