#!/usr/bin/env python
"""
Check a downloaded copy of CARR before training:
  * every frame of the manifest has an image, a YOLO label and a label with participant identifier,
  * the class and participant in the label files agree with the manifest,
  * every label contains exactly one box with coordinates in [0, 1],
  * the split files reference only frames of the manifest.

Usage:
  python tools/check_dataset.py --data /path/to/CARR [--skip-images]
"""
import argparse, collections, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from carr.common import Paths, ImageStore, CLASSES, SPLITS, REID_SPLITS, load_reid_split, log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True)
    ap.add_argument('--split-dir', help='folder with split files (default: <data>/splits)')
    ap.add_argument('--skip-images', action='store_true', help='do not look for the image files')
    a = ap.parse_args()
    P = Paths(a.data, work=os.path.join(os.getcwd(), 'carr_work'), splits=a.split_dir)
    M = P.manifest_rows()
    log(f'{len(M):,} frames in the manifest ({P.layout} layout)')
    problems = collections.Counter()
    examples = collections.defaultdict(list)

    def bad(kind, fn):
        problems[kind] += 1
        if len(examples[kind]) < 3:
            examples[kind].append(fn)

    std, ext = P.label_reader('labels'), P.label_reader('labels_with_person_id')
    for r in M:
        fn = r['file_name']
        try:
            s, e = std(fn).split(), ext(fn).split()
        except (FileNotFoundError, KeyError):
            bad('label file missing', fn)
            continue
        if len(s) != 5 or len(e) != 6:
            bad('not exactly one box per label', fn)
            continue
        if s[0] != r['class_id'] or e[1] != r['class_id']:
            bad('class differs from manifest', fn)
        if int(e[0]) != int(r['person_id']):
            bad('participant differs from manifest', fn)
        if s[1:] != e[2:] or not all(0 <= float(v) <= 1 for v in s[1:]):
            bad('box outside [0, 1] or labels disagree', fn)
    if not a.skip_images:
        store = ImageStore(P)
        for fn in store.missing([r['file_name'] for r in M]):
            bad('image missing', fn)
    names = {r['file_name'] for r in M}
    for p in SPLITS:
        for part in ('train', 'val', 'test'):
            try:
                files = [l.strip() for l in P.read_split(f'{p.lower()}_{part}.txt').splitlines() if l.strip()]
            except FileNotFoundError:
                bad('split file missing (run tools/make_splits.py)', f'{p.lower()}_{part}.txt')
                continue
            for f in files:
                if f not in names:
                    bad('split references unknown frame', f)
    for p in REID_SPLITS:
        try:
            for rows in load_reid_split(P, p).values():
                for r in rows:
                    if r['file'] not in names:
                        bad('re-ID split references unknown frame', r['file'])
        except FileNotFoundError:
            bad('re-ID split file missing', p)
    print(f'Classes: {", ".join(f"{i} {c}" for i, c in enumerate(CLASSES))}')
    if not problems:
        print('All checks passed.')
        return
    for k, n in problems.items():
        print(f'{k}: {n:,} (e.g. {examples[k]})')
    sys.exit(1)


if __name__ == '__main__':
    main()
