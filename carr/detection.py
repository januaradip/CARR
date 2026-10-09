"""
Frame-level detection baselines (Table 11): YOLOv8m and YOLOv12m on the nine CS/CL/CG splits (runs R01-R18).
Training uses every tenth frame of the training videos; evaluation uses all test frames.
"""
import os, json, shutil, time, zipfile, concurrent.futures as cf
from .common import *

CFG = dict(
    IMGSZ=640,          # YOLO input size
    EPOCHS=50,
    PATIENCE=10,        # early stopping (epochs without improvement)
    BATCH=16,
    WORKERS=2,
    TRAIN_STRIDE=10,    # train on 1 of every 10 frames (consecutive frames are nearly identical)
    VAL_STRIDE=10,      # validation during training is subsampled as well; the test set uses ALL frames
    SHARD_VIDEOS=40,    # videos per cache shard
)
WEIGHTS = {'YOLOv8m': 'yolov8m.pt', 'YOLOv12m': 'yolo12m.pt'}
METRICS = ['mAP50_all', 'mAP50_95_all', 'mAP50_dis', 'mAP50_nondis']


def _loc():
    return f'{local_scratch()}/carr_det'


# ---------------------------------------------------------------- 1. resized image cache (640 px)
def build_cache(P, F):
    d = f'{P.cache}/det_shards'
    os.makedirs(d, exist_ok=True)
    vids = sorted(F['videos'])
    chunks = [vids[i:i + CFG['SHARD_VIDEOS']] for i in range(0, len(vids), CFG['SHARD_VIDEOS'])]

    def ok(k, c):
        j = f'{d}/shard_{k:03d}.json'
        if not (os.path.exists(f'{d}/shard_{k:03d}.zip') and os.path.exists(j)):
            return False
        with open(j) as fh:
            return json.load(fh) == c                 # shard content must match the video list exactly
    todo = [(k, c) for k, c in enumerate(chunks) if not ok(k, c)]
    if not todo:
        log(f'Detection cache complete ({len(chunks)} shards)')
        return len(chunks)
    store = ImageStore(P)
    S = CFG['IMGSZ']

    def work(stem):
        im = store.open(stem, draft=(S, S))
        im.thumbnail((S, S))
        return stem, jpeg_bytes(im, 92)

    for k, c in todo:
        files = [f for v in c for f in F['videos'][v]['files']]
        miss = store.missing(files)
        if miss:
            raise FileNotFoundError(f'{len(miss)} images not found, e.g. {miss[:3]}')
        tmp = f'{local_scratch()}/det_shard_tmp.zip'
        with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_STORED) as z, cf.ThreadPoolExecutor(4) as ex:
            for stem, b in ex.map(work, files):
                z.writestr(f'{stem}.jpg', b)
        copy_atomic(tmp, f'{d}/shard_{k:03d}.zip')
        save_json(c, f'{d}/shard_{k:03d}.json')
        os.remove(tmp)
        log(f'shard {k+1}/{len(chunks)} done ({len(files)} frames)')
    return len(chunks)


# ---------------------------------------------------------------- 2. YOLO dataset on local disk
def prepare_local(P, F, n_shards):
    LOC = _loc()
    sig = f"{len(F['frames'])}-{n_shards}-{F['sig']}"
    if os.path.exists(f'{LOC}/READY') and open(f'{LOC}/READY').read() == sig:
        return
    shutil.rmtree(LOC, ignore_errors=True)
    log('Preparing the detection dataset on local disk...')
    os.makedirs(f'{LOC}/images', exist_ok=True)
    os.makedirs(f'{LOC}/labels', exist_ok=True)
    for k in range(n_shards):
        with zipfile.ZipFile(f'{P.cache}/det_shards/shard_{k:03d}.zip') as z:
            z.extractall(f'{LOC}/images')
    read = P.label_reader('labels')
    for fn in F['frames']:
        with open(f'{LOC}/labels/{fn}.txt', 'w') as f:
            f.write(read(fn))
    log(f"Local dataset ready: {len(os.listdir(f'{LOC}/images')):,} images")
    open(f'{LOC}/READY', 'w').write(sig)


def write_yaml(F, split, parts):
    LOC = _loc()
    os.makedirs(f'{LOC}/lists', exist_ok=True)

    def files_of(vids, stride):
        return [f for v in vids for f in F['videos'][v]['files'][::stride]]

    lists = {'train': files_of(parts['train'], CFG['TRAIN_STRIDE']),
             'val': files_of(parts['val'], CFG['VAL_STRIDE']),
             'test_all': files_of(parts['test'], 1)}
    lists['test_dis'] = [f for f in lists['test_all'] if F['frames'][f]['participant'] in DISABLED]
    lists['test_nondis'] = [f for f in lists['test_all'] if F['frames'][f]['participant'] not in DISABLED]
    for k, fl in lists.items():
        with open(f'{LOC}/lists/{split}_{k}.txt', 'w') as fh:
            fh.write('\n'.join(f'{LOC}/images/{f}.jpg' for f in fl) + '\n')
    names = '\n'.join(f'  {i}: {c}' for i, c in enumerate(CLASSES))
    yamls = {}
    for tag in ('all', 'dis', 'nondis'):
        if not lists[f'test_{tag}']:
            continue
        y = f'{LOC}/lists/{split}_{tag}.yaml'
        with open(y, 'w') as fh:
            fh.write(f'path: {LOC}\ntrain: lists/{split}_train.txt\nval: lists/{split}_val.txt\n'
                     f'test: lists/{split}_test_{tag}.txt\nnames:\n{names}\n')
        yamls[tag] = y
    return yamls


# ---------------------------------------------------------------- 3. training + evaluation
def do_run(P, F, run):
    from ultralytics import YOLO
    rdir = run_dir(P, run)
    yamls = write_yaml(F, run['split'], load_split(P, run['split']))
    wdir = f'{rdir}/train/weights'
    if not os.path.exists(f'{rdir}/TRAIN_DONE'):
        last = f'{wdir}/last.pt'
        started = False
        if os.path.exists(last):
            try:
                log('Resuming training from last.pt')
                m = YOLO(last)
                m.add_callback('on_fit_epoch_end', lambda tr: heartbeat(rdir))
                m.train(resume=True)
                started = True
            except Exception as e:
                msg = str(e).lower()
                if 'nothing to resume' in msg or 'finished' in msg:
                    log('Training had already finished')
                    started = True
                else:
                    log(f'last.pt cannot be resumed ({e}); restarting this run')
                    shutil.move(f'{rdir}/train', f'{rdir}/train_broken_{int(time.time())}')
        if not started:
            m = YOLO(WEIGHTS[run['model']])
            m.add_callback('on_fit_epoch_end', lambda tr: heartbeat(rdir))
            m.train(data=yamls['all'], epochs=CFG['EPOCHS'], patience=CFG['PATIENCE'], imgsz=CFG['IMGSZ'],
                    batch=CFG['BATCH'], workers=CFG['WORKERS'], project=rdir, name='train', exist_ok=True,
                    seed=0, plots=True, verbose=False)
        open(f'{rdir}/TRAIN_DONE', 'w').write('ok')
    best = f'{wdir}/best.pt' if os.path.exists(f'{wdir}/best.pt') else f'{wdir}/last.pt'
    metrics = {}
    for tag, y in yamls.items():
        heartbeat(rdir)
        log(f'Evaluating test set ({tag})...')
        m = YOLO(best).val(data=y, split='test', imgsz=CFG['IMGSZ'], batch=CFG['BATCH'], workers=CFG['WORKERS'],
                           project=rdir, name=f'test_{tag}', exist_ok=True, plots=(tag == 'all'), verbose=False)
        metrics[f'mAP50_{tag}'] = float(m.box.map50)
        metrics[f'mAP50_95_{tag}'] = float(m.box.map)
        if tag == 'all':
            metrics['precision_all'] = float(m.box.mp)
            metrics['recall_all'] = float(m.box.mr)
            idx = getattr(m.box, 'ap_class_index', None)
            if idx is None or len(idx) == 0:
                idx = getattr(m, 'ap_class_index', [])
            for i, c in enumerate(idx):
                metrics[f'AP50_{CLASSES[int(c)]}'] = float(m.box.ap50[i])
    mark_done(P, run, metrics)
    log(f"{run['run_id']} finished: mAP@50 {metrics['mAP50_all']:.4f}")


def main(P, runs=None, reverse=False):
    P.check(need_images=False)
    F = load_frames(P)
    prepare_local(P, F, build_cache(P, F))
    execute_runs(P, runs or run_list('detection'), lambda run: do_run(P, F, run), reverse)
    return summarize(P, 'detection', METRICS)
