#!/usr/bin/env python
"""
Regenerate the frame lists of the CARR evaluation protocols from the manifest.

The assignment of every video to the training, validation or test set of each protocol is fixed in
splits/video_splits.csv (one row per video, one column per protocol). This script expands it into the
frame lists <protocol>_{train,val,test}.txt, writes splits_summary.csv and checks that

  * no video occurs in more than one subset of a protocol,
  * CS and CG test participants and CL test stores never occur in the training or validation set,
  * every frame listed in the re-identification splits exists in the manifest.

Usage:
  python tools/make_splits.py --manifest /path/to/CARR/carr_manifest.csv --out splits
"""
import argparse, collections, csv, os, sys

PROTOCOLS = ['CS1', 'CS2', 'CS3', 'CS4', 'CL1', 'CL2', 'CL3', 'CL4', 'CG']
CLASSES = ['No Interest', 'Viewing', 'Turning To Shelf', 'Touching', 'Picking and Returning', 'Picking and Putting']


def main():
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'splits')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--video-splits', default=os.path.join(here, 'video_splits.csv'))
    ap.add_argument('--reid', default=os.path.join(here, 'reid'))
    ap.add_argument('--out', default='splits')
    a = ap.parse_args()

    M = list(csv.DictReader(open(a.manifest, encoding='utf-8')))
    by_video = collections.defaultdict(list)
    for r in M:
        by_video[r['video_id']].append(r['file_name'])
    V = {r['video_id']: r for r in csv.DictReader(open(a.video_splits, encoding='utf-8'))}
    unknown = set(by_video) - set(V)
    if unknown:
        sys.exit(f'{len(unknown)} videos of the manifest are not in video_splits.csv, e.g. {sorted(unknown)[:3]}')

    os.makedirs(a.out, exist_ok=True)
    summary = []
    for p in PROTOCOLS:
        part = {k: sorted(v for v in by_video if V[v][p] == k) for k in ('train', 'val', 'test')}
        for k, vids in part.items():
            files = sorted(f for v in vids for f in by_video[v])
            with open(f'{a.out}/{p.lower()}_{k}.txt', 'w', encoding='utf-8') as fh:
                fh.write('\n'.join(files) + '\n')
        tr, va, te = (set(part[k]) for k in ('train', 'val', 'test'))
        assert not (tr & te or va & te or tr & va), p
        if p.startswith('CS') or p == 'CG':
            assert not ({V[v]['participant_code'] for v in tr | va} & {V[v]['participant_code'] for v in te}), p
        if p.startswith('CL'):
            assert not ({V[v]['location_name'] for v in tr | va} & {V[v]['location_name'] for v in te}), p
        nfr = lambda vs: sum(len(by_video[v]) for v in vs)
        cc = collections.Counter(V[v]['class_name'] for v in te)
        summary.append(dict(protocol=p, train_videos=len(tr), val_videos=len(va), test_videos=len(te),
                            train_frames=nfr(tr), val_frames=nfr(va), test_frames=nfr(te),
                            test_participants=' '.join(sorted({V[v]['participant_code'] for v in te})),
                            test_locations='; '.join(sorted({V[v]['location_name'] for v in te})),
                            min_test_videos_per_class=min(cc[c] for c in CLASSES)))
    with open(f'{a.out}/splits_summary.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    names = {r['file_name'] for r in M}
    for fn in sorted(os.listdir(a.reid)) if os.path.isdir(a.reid) else []:
        if fn.endswith('.csv'):
            rows = list(csv.DictReader(open(f'{a.reid}/{fn}', encoding='utf-8')))
            miss = [r['file_name'] for r in rows if r['file_name'] not in names]
            assert not miss, f'{fn}: {len(miss)} frames not in the manifest, e.g. {miss[:3]}'
    for s in summary:
        print(f"{s['protocol']:<4} videos {s['train_videos']}/{s['val_videos']}/{s['test_videos']}  "
              f"test frames {s['test_frames']:,}  test participants: {s['test_participants']}")
    print(f'Split files written to {a.out}/ - all checks passed.')


if __name__ == '__main__':
    main()
