#!/usr/bin/env python
"""
Run the CARR baselines.

Examples
--------
  # all 18 detection runs (R01-R18) on a dataset downloaded from Harvard Dataverse
  python run_benchmark.py --data /path/to/CARR --family detection

  # only CTR-GCN on the cross-subject folds
  python run_benchmark.py --data /path/to/CARR --family skeleton --models CTR-GCN --splits CS1 CS2 CS3 CS4

  # selected runs by identifier, results written to another folder
  python run_benchmark.py --data /path/to/CARR --work /path/to/output --runs R37 R38

  # a second session that works through the list from the end (sessions share the work folder)
  python run_benchmark.py --data /path/to/CARR --family detection --reverse

Every run writes <work>/runs/<family>/<run>/DONE.json when finished; finished runs are skipped,
interrupted runs resume from their last checkpoint. Results are collected in <work>/results/.
"""
import argparse, importlib, sys
from carr import common

FAMILIES = ['detection', 'skeleton', 'video', 'reid']


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='dataset root (release or packed layout)')
    ap.add_argument('--work', help='output folder for cache/, runs/ and results/')
    ap.add_argument('--split-dir', help='folder with split files (default: <data>/splits)')
    ap.add_argument('--family', choices=FAMILIES + ['all'], default='all')
    ap.add_argument('--models', nargs='*', help='restrict to these models, e.g. YOLOv8m CTR-GCN VideoMAE-B')
    ap.add_argument('--splits', nargs='*', help='restrict to these splits, e.g. CS1 CG ReID2')
    ap.add_argument('--runs', nargs='*', help='restrict to these run identifiers, e.g. R01 R37')
    ap.add_argument('--reverse', action='store_true', help='process the runs from the last to the first')
    ap.add_argument('--list', action='store_true', help='only list the selected runs and their status')
    a = ap.parse_args(argv)

    P = common.Paths(a.data, a.work, a.split_dir)
    runs = common.all_runs()
    if a.family != 'all':
        runs = [r for r in runs if r['family'] == a.family]
    if a.models:
        runs = [r for r in runs if r['model'] in a.models]
    if a.splits:
        runs = [r for r in runs if r['split'] in a.splits]
    if a.runs:
        runs = [r for r in runs if r['run_id'] in a.runs]
    if not runs:
        sys.exit('No run matches the selection.')
    if a.list:
        for r in runs:
            print(f"{r['run_id']}  {r['family']:<9}  {r['model']:<12}  {r['split']:<6}  "
                  f"{'done' if common.is_done(P, r) else 'open'}")
        return
    common.log(f'Dataset: {P.data} ({P.layout} layout); output: {P.work}; {len(runs)} runs selected')
    for fam in FAMILIES:
        sel = [r for r in runs if r['family'] == fam]
        if sel:
            mod = importlib.import_module(f'carr.{fam}')
            summary = mod.main(P, sel, a.reverse)
            if summary is not None:
                print(summary.to_string())


if __name__ == '__main__':
    main()
