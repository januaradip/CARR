"""
Shared utilities for the CARR benchmark: data access, splits, caching, run bookkeeping
and a generic classifier trainer.

Two dataset layouts are supported (detected automatically):

* release layout (as downloaded from Harvard Dataverse)::

      <data>/carr_manifest.csv
      <data>/images/            *.jpg, or one or more .zip archives with the JPEG files
      <data>/labels/            YOLO labels
      <data>/labels_with_person_id/
      <data>/splits/            frame lists of the protocols and splits/reid/*.csv

* packed layout (used for Google Drive, where hundreds of thousands of small files are slow)::

      <data>/dataset/CARR_labels_clean.zip   manifest + labels + labels_with_person_id
      <data>/dataset/CARR_splits.zip         splits/*.txt and splits/reid/*.csv
      <data>/dataset/images/*.zip            JPEG frames

Every step is resumable: caches, checkpoints and finished runs are kept in the work directory
and are skipped when the same command is started again.
"""
import os, io, re, csv, json, time, glob, shutil, zipfile, pickle, random, collections, math, threading, uuid

CLASSES = ['No Interest', 'Viewing', 'Turning To Shelf', 'Touching',
           'Picking and Returning', 'Picking and Putting']
DISABLED = {'P07', 'P08', 'P09', 'P10', 'P11', 'P12', 'P13'}       # participants with mobility impairments
SPLITS = ['CS1', 'CS2', 'CS3', 'CS4', 'CL1', 'CL2', 'CL3', 'CL4', 'CG']
REID_SPLITS = ['ReID1', 'ReID2', 'ReID3', 'ReID4']

# Run identifiers R01-R71 follow this order (family, model, split).
FAMILY_MODELS = [
    ('detection', ['YOLOv8m', 'YOLOv12m'], SPLITS),
    ('skeleton',  ['LSTM', 'ST-GCN', 'CTR-GCN'], SPLITS),
    ('video',     ['R2Plus1D-18', 'VideoMAE-B'], SPLITS),
    ('reid',      ['BoT-R50', 'ViT-B16'], REID_SPLITS),
]


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def local_scratch():
    """Fast local disk for temporary copies (Colab: /content). Override with CARR_LOCAL."""
    d = os.environ.get('CARR_LOCAL') or ('/content' if os.path.isdir('/content') else
                                          os.path.join(os.path.expanduser('~'), '.cache', 'carr'))
    os.makedirs(d, exist_ok=True)
    return d


# --------------------------------------------------------------------------- dataset access
class Paths:
    """Locates the dataset and the work directory.

    data   : dataset root (release or packed layout, see module docstring)
    work   : where cache/, runs/ and results/ are written (default: <data> for the packed layout,
             ./carr_work for the release layout, which may be read-only)
    splits : optional folder with the split files (default: <data>/splits or CARR_splits.zip)
    """

    def __init__(self, data, work=None, splits=None):
        self.data = os.path.abspath(data.rstrip('/'))
        if os.path.isfile(f'{self.data}/carr_manifest.csv'):
            self.layout = 'release'
            self.images_dir = f'{self.data}/images'
            default_work = os.path.abspath('carr_work')
        elif os.path.isfile(f'{self.data}/dataset/CARR_labels_clean.zip'):
            self.layout = 'packed'
            self.images_dir = f'{self.data}/dataset/images'
            default_work = self.data
        else:
            raise FileNotFoundError(
                f'No CARR dataset found in {self.data}: expected carr_manifest.csv (release layout) '
                f'or dataset/CARR_labels_clean.zip (packed layout). See README.md.')
        self.work = os.path.abspath(work or default_work)
        self.splits_dir = os.path.abspath(splits) if splits else None
        self.cache = f'{self.work}/cache'
        self.runs = f'{self.work}/runs'
        self.results = f'{self.work}/results'
        self._zips = {}
        for d in (self.cache, self.runs, self.results):
            os.makedirs(d, exist_ok=True)

    # ---- packed-layout helpers
    def _zip(self, path):
        if path not in self._zips:
            z = zipfile.ZipFile(path)
            self._zips[path] = (z, {n.split('/', 1)[-1] if n.count('/') else n: n for n in z.namelist()},
                                z.namelist())
        return self._zips[path]

    def _zip_find(self, path, suffix):
        z, _, names = self._zip(path)
        hit = [n for n in names if n.endswith(suffix)]
        if not hit:
            raise FileNotFoundError(f'{suffix} not found in {path}')
        return z, hit[0]

    # ---- public accessors
    def signature(self):
        """Changes when the annotations change, so that cached indices are rebuilt."""
        p = f'{self.data}/carr_manifest.csv' if self.layout == 'release' else f'{self.data}/dataset/CARR_labels_clean.zip'
        return f'{self.layout}-{os.path.getsize(p)}'

    def manifest_rows(self):
        if self.layout == 'release':
            with open(f'{self.data}/carr_manifest.csv', encoding='utf-8') as f:
                return list(csv.DictReader(f))
        z, n = self._zip_find(f'{self.data}/dataset/CARR_labels_clean.zip', 'carr_manifest.csv')
        return list(csv.DictReader(io.TextIOWrapper(z.open(n), encoding='utf-8')))

    def label_reader(self, folder='labels_with_person_id'):
        """Returns a function file_name -> label text for 'labels' or 'labels_with_person_id'."""
        if self.layout == 'release':
            base = f'{self.data}/{folder}'
            return lambda fn: open(f'{base}/{fn}.txt').read()
        z, names = zipfile.ZipFile(f'{self.data}/dataset/CARR_labels_clean.zip'), None
        names = z.namelist()
        member = [n for n in names if f'{folder}/' in n and n.endswith('.txt')
                  and (folder != 'labels' or 'labels_with_person_id' not in n)][0]
        prefix = member[:member.rindex('/') + 1]
        return lambda fn: z.read(f'{prefix}{fn}.txt').decode()

    def read_split(self, name):
        """Text of a split file, e.g. 'cs1_test.txt' or 'reid/reid1_query.csv'."""
        if self.splits_dir:
            with open(f'{self.splits_dir}/{name}', encoding='utf-8') as f:
                return f.read()
        if self.layout == 'release':
            with open(f'{self.data}/splits/{name}', encoding='utf-8') as f:
                return f.read()
        z, n = self._zip_find(f'{self.data}/dataset/CARR_splits.zip', '/' + name.split('/')[-1])
        return z.read(n).decode('utf-8')

    def check(self, need_images=True):
        errs = []
        if self.layout == 'packed' and not os.path.isfile(f'{self.data}/dataset/CARR_splits.zip') and not self.splits_dir:
            errs.append(f'Missing {self.data}/dataset/CARR_splits.zip (or pass --splits)')
        if self.layout == 'release' and not self.splits_dir and not os.path.isdir(f'{self.data}/splits'):
            errs.append(f'Missing {self.data}/splits (or pass --splits)')
        if need_images and not (glob.glob(f'{self.images_dir}/**/*.zip', recursive=True)
                                or glob.glob(f'{self.images_dir}/**/*.jpg', recursive=True)):
            errs.append(f'No .jpg or .zip files in {self.images_dir}')
        if errs:
            raise FileNotFoundError('\n'.join(errs))


# --------------------------------------------------------------------------- safe IO
def atomic_write_bytes(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(data)
    os.replace(tmp, path)


def save_pickle(obj, path):
    atomic_write_bytes(path, pickle.dumps(obj, protocol=4))


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_json(obj, path):
    atomic_write_bytes(path, json.dumps(obj, indent=2).encode())


def copy_atomic(src, dst):
    """Copy a file without ever leaving a half-written destination (safe on Google Drive)."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + '.tmp'
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


# --------------------------------------------------------------------------- frame index
def _n_of(name):
    """Running number of a frame name: 'Viewing (123)' -> 123."""
    m = re.search(r'\((\d+)\)$', name) or re.search(r'_f(\d+)$', name)
    return int(m.group(1)) if m else 0


def load_frames(P):
    """Index of all frames (box, class, participant, store, video) and of all videos with their frames in
    temporal order. Built once from the manifest and the labels and cached in <work>/cache/frames.pkl."""
    cache = f'{P.cache}/frames.pkl'
    sig = P.signature()
    if os.path.exists(cache):
        try:
            obj = load_pickle(cache)
            if obj.get('sig') == sig:
                return obj
            log('Annotations changed -> rebuilding the frame index')
        except Exception:
            log('frames.pkl unreadable -> rebuilding')
    log('Building the frame index (once, about 2 minutes)...')
    read = P.label_reader('labels_with_person_id')
    frames = {}
    for r in P.manifest_rows():
        fn = r['file_name']
        t = read(fn).split()
        pid, c = int(t[0]), int(t[1])
        x, y, w, h = map(float, t[2:6])
        fi = r['frame_index']
        frames[fn] = dict(file=fn, video=r['video_id'], participant=f'P{pid:02d}',
                          location=r['location_name'], cls=c, box=(x, y, w, h),
                          n=_n_of(fn), frame_index=int(fi) if fi != '' else None)
    byv = collections.defaultdict(list)
    for fn, f in frames.items():
        byv[f['video']].append(f)
    videos = {}
    for v, lst in byv.items():
        lst.sort(key=lambda f: f['n'])
        # temporal order: original frame index; frames without an index follow their predecessor
        last, keys = -1.0, []
        for f in lst:
            last = float(f['frame_index']) if f['frame_index'] is not None else last + 1e-3
            keys.append(last)
        order = [f for _, f in sorted(zip(keys, lst), key=lambda t: t[0])]
        owner = collections.Counter(f['participant'] for f in order).most_common(1)[0][0]
        videos[v] = dict(files=[f['file'] for f in order], cls=order[0]['cls'],
                         owner=owner, location=order[0]['location'],
                         group='dis' if owner in DISABLED else 'nondis')
    obj = dict(frames=frames, videos=videos, sig=sig)
    save_pickle(obj, cache)
    log(f'Index ready: {len(frames):,} frames, {len(videos)} videos')
    return obj


def load_split(P, name):
    """{'train', 'val', 'test'} -> sorted list of video identifiers of protocol `name` (e.g. 'CS1')."""
    F = load_frames(P)
    out = {}
    for part in ('train', 'val', 'test'):
        files = [l.strip() for l in P.read_split(f'{name.lower()}_{part}.txt').splitlines() if l.strip()]
        out[part] = sorted({F['frames'][f]['video'] for f in files if f in F['frames']})
    return out


def load_reid_split(P, name):
    """{'train', 'query', 'gallery'} -> list of dict(file, pid, video, location) for re-ID fold `name`."""
    k = name[-1]
    out = {}
    for part in ('train', 'query', 'gallery'):
        rows = csv.DictReader(io.StringIO(P.read_split(f'reid/reid{k}_{part}.csv')))
        out[part] = [dict(file=r['file_name'], pid=int(r['person_id']), video=r['video_id'],
                          location=r['location_name']) for r in rows]
    return out


# --------------------------------------------------------------------------- images
class ImageStore:
    """Reads the JPEG frames from a folder or from one or more .zip archives in the images folder.
    Archives on a network drive are first copied to local disk when there is enough space."""

    def __init__(self, P):
        self.zips, self.index, self._tl = [], {}, threading.local()
        zips = sorted(glob.glob(f'{P.images_dir}/**/*.zip', recursive=True))
        if zips:
            total = sum(os.path.getsize(z) for z in zips)
            local = local_scratch()
            use_local = P.layout == 'packed' and shutil.disk_usage(local).free > total * 1.1 + 15e9
            for zp in zips:
                src = zp
                if use_local:
                    dst = f'{local}/carr_images_zip/{os.path.basename(zp)}'
                    if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(zp)):
                        log(f'Copying {os.path.basename(zp)} ({os.path.getsize(zp)/1e9:.1f} GB) to local disk...')
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        shutil.copyfile(zp, dst + '.part')
                        os.replace(dst + '.part', dst)
                    src = dst
                zi = len(self.zips)
                self.zips.append(src)
                for n in zipfile.ZipFile(src).namelist():
                    if n.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.index[os.path.splitext(os.path.basename(n))[0]] = (zi, n)
        for p in glob.glob(f'{P.images_dir}/**/*', recursive=True):
            if p.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.index.setdefault(os.path.splitext(os.path.basename(p))[0], (None, p))
        if not self.index:
            raise FileNotFoundError(f'No images found in {P.images_dir}')
        log(f'ImageStore: {len(self.index):,} images indexed')

    def read(self, stem):
        zi, n = self.index[stem]
        if zi is None:
            with open(n, 'rb') as f:
                return f.read()
        opened = getattr(self._tl, 'zf', None)
        if opened is None:
            opened = self._tl.zf = {}
        if zi not in opened:
            opened[zi] = zipfile.ZipFile(self.zips[zi])      # one handle per thread (thread-safe)
        return opened[zi].read(n)

    def open(self, stem, draft=None):
        from PIL import Image
        im = Image.open(io.BytesIO(self.read(stem)))
        if draft and im.format == 'JPEG':
            im.draft('RGB', draft)           # decode the JPEG directly at reduced resolution
        return im.convert('RGB')

    def missing(self, stems):
        return [s for s in stems if s not in self.index]


def crop_box(im, box, expand=1.0, square=False):
    """Crop around a normalised box (x_center, y_center, w, h)."""
    W, H = im.size
    x, y, w, h = box
    bw, bh = w * W * expand, h * H * expand
    if square:
        bw = bh = max(bw, bh)
    cx, cy = x * W, y * H
    return im.crop((int(round(cx - bw / 2)), int(round(cy - bh / 2)),
                    int(round(cx + bw / 2)), int(round(cy + bh / 2))))


def jpeg_bytes(im, quality=90):
    b = io.BytesIO()
    im.save(b, format='JPEG', quality=quality)
    return b.getvalue()


# --------------------------------------------------------------------------- runs & results
def all_runs():
    out, rid = [], 1
    for fam, models, splits in FAMILY_MODELS:
        for m in models:
            for s in splits:
                out.append(dict(run_id=f'R{rid:02d}', family=fam, model=m, split=s))
                rid += 1
    return out


def run_list(family):
    return [r for r in all_runs() if r['family'] == family]


def run_dir(P, run):
    return f"{P.runs}/{run['family']}/{run['run_id']}_{run['model']}_{run['split']}"


def is_done(P, run):
    return os.path.exists(f'{run_dir(P, run)}/DONE.json')


def mark_done(P, run, metrics):
    row = dict(run_id=run['run_id'], model=run['model'], split=run['split'], **metrics)
    save_json(row, f'{run_dir(P, run)}/DONE.json')
    record_result(P, run['family'], row)


def record_result(P, family, row):
    path = f'{P.results}/{family}_results.csv'
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            rows = [r for r in csv.DictReader(f) if r['run_id'] != row['run_id']]
    rows.append(dict(row))
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    rows.sort(key=lambda r: r['run_id'])
    b = io.StringIO()
    w = csv.DictWriter(b, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
    atomic_write_bytes(path, b.getvalue().encode())


def rebuild_results(P, family):
    """Rebuild <family>_results.csv from all DONE.json files."""
    for d in sorted(glob.glob(f'{P.runs}/{family}/*/DONE.json')):
        with open(d) as f:
            record_result(P, family, json.load(f))


def summarize(P, family, main_metrics):
    """Mean ± std over folds per model and protocol (CS / CL / CG / ReID)."""
    rebuild_results(P, family)
    path = f'{P.results}/{family}_results.csv'
    if not os.path.exists(path):
        log('No results yet.')
        return None
    import pandas as pd
    df = pd.read_csv(path)
    df['protocol'] = df['split'].str.replace(r'\d+$', '', regex=True)
    cols = [m for m in main_metrics if m in df.columns]
    agg = df.groupby(['model', 'protocol'])[cols].agg(['mean', 'std', 'count'])
    out = pd.DataFrame(index=agg.index)
    for m in cols:
        mean, std, cnt = agg[(m, 'mean')], agg[(m, 'std')].fillna(0), agg[(m, 'count')]
        out[m] = [f'{a:.3f} ± {s:.3f} (n={int(c)})' for a, s, c in zip(mean, std, cnt)]
    out.to_csv(f'{P.results}/{family}_summary.csv')
    return out


# --------------------------------------------------------------------------- run locks (parallel sessions)
SESSION = uuid.uuid4().hex[:8]
LOCK_STALE_SEC = 3 * 3600          # a lock not refreshed for 3 h is considered abandoned


def claim_run(rdir):
    """Claim a run for this session; False if another session is working on it."""
    os.makedirs(rdir, exist_ok=True)
    lock = f'{rdir}/LOCK'
    if os.path.exists(lock):
        try:
            with open(lock) as f:
                owner = json.load(f).get('session')
        except Exception:
            owner = None
        if owner != SESSION and time.time() - os.path.getmtime(lock) < LOCK_STALE_SEC:
            return False
    save_json({'session': SESSION, 'since': time.strftime('%Y-%m-%d %H:%M:%S')}, lock)
    time.sleep(float(os.environ.get('CARR_LOCK_WAIT', 15)))   # let a synced drive settle, then re-check
    try:
        with open(lock) as f:
            return json.load(f).get('session') == SESSION
    except Exception:
        return False


def heartbeat(rdir):
    try:
        save_json({'session': SESSION, 'beat': time.strftime('%Y-%m-%d %H:%M:%S')}, f'{rdir}/LOCK')
    except Exception:
        pass


def release_run(rdir):
    try:
        os.remove(f'{rdir}/LOCK')
    except OSError:
        pass


def execute_runs(P, runs, do_run, reverse=False):
    """Run `do_run(run)` for every unfinished run, skipping runs claimed by another session."""
    runs = runs[::-1] if reverse else runs
    busy = []
    for run in runs:
        tag = f"{run['run_id']} {run['model']} {run['split']}"
        if is_done(P, run):
            log(f'{tag}: already finished, skipped')
            continue
        rdir = run_dir(P, run)
        if not claim_run(rdir):
            log(f'{tag}: in progress in another session, skipped')
            busy.append(run['run_id'])
            continue
        log(f'===== {tag} =====')
        seed_everything(0)
        try:
            do_run(run)
        finally:
            release_run(rdir)
    if busy:
        log(f'Runs still in progress elsewhere: {busy}. Start the command again afterwards to complete the summary.')
    return busy


def seed_everything(seed=0):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- generic classifier trainer
def _torch_save_atomic(obj, path):
    import torch
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    if os.path.exists(path):
        os.replace(path, path + '.prev')
    os.replace(tmp, path)


def _torch_load_safe(path):
    import torch
    for p in (path, path + '.prev'):
        if os.path.exists(p):
            try:
                return torch.load(p, map_location='cpu', weights_only=False)
            except Exception as e:
                log(f'Checkpoint {os.path.basename(p)} unreadable ({e}); trying the backup')
    return None


def classification_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    return dict(acc=float(accuracy_score(y_true, y_pred)),
                macro_f1=float(f1_score(y_true, y_pred, average='macro', labels=list(range(6)), zero_division=0)),
                cm=confusion_matrix(y_true, y_pred, labels=list(range(6))).tolist())


def train_classifier(rdir, model, train_ds, val_ds, cfg, device='cuda'):
    """Training with a checkpoint after every epoch (automatic resume) and early stopping on validation
    macro-F1. cfg keys: epochs, lr, wd, batch, patience, workers, warmup, accum, amp."""
    import torch
    from torch.utils.data import DataLoader
    os.makedirs(rdir, exist_ok=True)
    done_flag, best_path = f'{rdir}/TRAIN_DONE', f'{rdir}/best.pt'
    if os.path.exists(done_flag) and os.path.exists(best_path):
        log('Training already finished, continuing with evaluation')
        return
    model.to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg['lr'], weight_decay=cfg['wd'])
    E = cfg['epochs']
    warm = max(1, cfg.get('warmup', 1))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: (e + 1) / warm if e < warm else 0.5 * (1 + math.cos(math.pi * (e - warm) / max(1, E - warm))))
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.get('amp', True))
    state = dict(epoch=0, best=-1.0, bad=0, history=[])
    ck = _torch_load_safe(f'{rdir}/last.pt')
    if ck is not None:
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        scaler.load_state_dict(ck['scaler'])
        state = ck['state']
        log(f"Resuming from epoch {state['epoch']} (best validation macro-F1 {state['best']:.4f})")
    tl = DataLoader(train_ds, batch_size=cfg['batch'], shuffle=True, num_workers=cfg['workers'],
                    drop_last=True, pin_memory=True, persistent_workers=cfg['workers'] > 0)
    vl = DataLoader(val_ds, batch_size=cfg['batch'], shuffle=False, num_workers=cfg['workers'], pin_memory=True)
    crit = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    accum = cfg.get('accum', 1)
    while state['epoch'] < E and state['bad'] < cfg['patience']:
        ep = state['epoch']
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, (x, y, _) in enumerate(tl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16, enabled=cfg.get('amp', True)):
                loss = crit(model(x), y) / accum
            scaler.scale(loss).backward()
            if (i + 1) % accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            tot += loss.item() * accum * len(y)
            n += len(y)
        sched.step()
        yt, yp, _, _ = predict(model, vl, device, cfg.get('amp', True))
        m = classification_metrics(yt, yp)
        state['history'].append(dict(epoch=ep + 1, loss=tot / max(1, n), val_acc=m['acc'], val_f1=m['macro_f1'],
                                     sec=round(time.time() - t0)))
        improved = m['macro_f1'] > state['best']
        if improved:
            state['best'], state['bad'] = m['macro_f1'], 0
            _torch_save_atomic({'model': model.state_dict()}, best_path)
        else:
            state['bad'] += 1
        state['epoch'] = ep + 1
        _torch_save_atomic(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                                scaler=scaler.state_dict(), state=state), f'{rdir}/last.pt')
        save_json(state['history'], f'{rdir}/history.json')
        heartbeat(rdir)
        log(f"epoch {ep+1}/{E} loss {tot/max(1,n):.4f} val_acc {m['acc']:.4f} val_F1 {m['macro_f1']:.4f}"
            f"{' *best*' if improved else ''} ({time.time()-t0:.0f}s)")
    open(done_flag, 'w').write('ok')
    cleanup_checkpoints(rdir)


def cleanup_checkpoints(rdir):
    """Remove the resume checkpoints (with optimizer state) once training is finished; best.pt is kept."""
    for f in ('last.pt', 'last.pt.prev', 'last.pt.tmp'):
        if os.path.exists(f'{rdir}/{f}'):
            os.remove(f'{rdir}/{f}')


def predict(model, loader, device='cuda', amp=True):
    import torch, numpy as np
    model.eval()
    yt, yp, probs, idx = [], [], [], []
    with torch.no_grad():
        for x, y, i in loader:
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                out = model(x.to(device, non_blocking=True)).float()
            p = out.softmax(1).cpu().numpy()
            probs.append(p)
            yp.extend(p.argmax(1).tolist())
            yt.extend(y.tolist())
            idx.extend(i.tolist())
    return np.array(yt), np.array(yp), np.concatenate(probs) if probs else np.zeros((0, 6)), np.array(idx)


def evaluate_classifier(rdir, model, test_ds, window_videos, videos_meta, cfg, device='cuda'):
    """Window- and video-level accuracy and macro-F1, overall and per participant group.
    Video-level predictions average the class probabilities of all windows of a video."""
    import torch, numpy as np
    from torch.utils.data import DataLoader
    model.load_state_dict(torch.load(f'{rdir}/best.pt', map_location='cpu', weights_only=False)['model'])
    model.to(device)
    tl = DataLoader(test_ds, batch_size=cfg['batch'], shuffle=False, num_workers=cfg['workers'], pin_memory=True)
    yt, yp, probs, idx = predict(model, tl, device, cfg.get('amp', True))
    wv = np.array([window_videos[i] for i in idx])
    res = {}
    for tag, mask in (('all', np.ones(len(yt), bool)),
                      ('dis', np.array([videos_meta[v]['group'] == 'dis' for v in wv])),
                      ('nondis', np.array([videos_meta[v]['group'] == 'nondis' for v in wv]))):
        if mask.sum() == 0:
            continue
        m = classification_metrics(yt[mask], yp[mask])
        res[f'win_acc_{tag}'], res[f'win_f1_{tag}'] = m['acc'], m['macro_f1']
        if tag == 'all':
            save_json(m['cm'], f'{rdir}/confusion_window.json')
        vt, vp = [], []
        for v in sorted(set(wv[mask])):
            sel = wv == v
            vt.append(int(yt[sel][0]))
            vp.append(int(probs[sel].mean(0).argmax()))
        mv = classification_metrics(np.array(vt), np.array(vp))
        res[f'vid_acc_{tag}'], res[f'vid_f1_{tag}'] = mv['acc'], mv['macro_f1']
        if tag == 'all':
            save_json(mv['cm'], f'{rdir}/confusion_video.json')
    return res
