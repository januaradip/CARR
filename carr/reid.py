"""
Person re-identification (Table 12): BoT-ResNet50 and ViT-B/16 on the four re-ID folds (runs R64-R71).
Step 1 (once): 256 x 128 person crops of the frames listed in the re-ID splits -> <work>/cache/reidcrops/
Step 2: training with identity (cross-entropy) and batch-hard triplet losses (BNNeck); evaluation with
Rank-1, Rank-5 and mAP. Gallery images from the same video as the query are excluded.
"""
import os, io, glob, math, time, random, collections, concurrent.futures as cf
import numpy as np
from .common import *
from .common import _torch_load_safe, _torch_save_atomic

CFG = dict(
    H=256, W=128, EXPAND=1.05, P_IDS=7, K_IMGS=8, ITERS=300, WORKERS=2,
    MODEL={
        'BoT-R50': dict(epochs=40, opt='adam', lr=3.5e-4, wd=5e-4, warmup=5),
        'ViT-B16': dict(epochs=40, opt='sgd', lr=8e-3, wd=1e-4, warmup=5),
    },
)
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


# ================================================================ 1. crop cache
def build_crops(P, F):
    d = f'{P.cache}/reidcrops'
    os.makedirs(d, exist_ok=True)
    need = collections.defaultdict(set)
    for s in REID_SPLITS:
        for part in load_reid_split(P, s).values():
            for r in part:
                need[r['video']].add(r['file'])
    todo = []
    for v in sorted(need):
        p = f'{d}/{v}.pkl'
        if not os.path.exists(p):
            todo.append(v)
        else:
            try:
                if not need[v] <= set(load_pickle(p)):
                    todo.append(v)                     # split changed -> complete the crops of this video
            except Exception:
                todo.append(v)
    if not todo:
        log('Re-ID crop cache complete')
        return
    log(f'Cropping re-ID images for {len(todo)} remaining videos...')
    store = ImageStore(P)

    def work(f):
        im = store.open(f, draft=(960, 960))
        return f, jpeg_bytes(crop_box(im, F['frames'][f]['box'], CFG['EXPAND']).resize((CFG['W'], CFG['H'])), 92)

    with cf.ThreadPoolExecutor(4) as ex:
        for i, v in enumerate(todo):
            save_pickle(dict(ex.map(work, sorted(need[v]))), f'{d}/{v}.pkl')
            if (i + 1) % 20 == 0 or i + 1 == len(todo):
                log(f're-ID crops {i+1}/{len(todo)} videos')


def load_crops(P):
    out = {}
    for p in glob.glob(f'{P.cache}/reidcrops/*.pkl'):
        out.update(load_pickle(p))
    return out


class ReIDDataset:
    def __init__(self, items, crops, train, label_map=None):
        self.items, self.crops, self.train, self.lm = items, crops, train, label_map

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import torch
        from PIL import Image
        r = self.items[i]
        im = Image.open(io.BytesIO(self.crops[r['file']])).convert('RGB').resize((CFG['W'], CFG['H']))
        x = np.asarray(im, np.float32) / 255.0
        if self.train:
            if random.random() < 0.5:
                x = x[:, ::-1]
            x = np.pad(x, ((10, 10), (10, 10), (0, 0)))
            oy, ox = random.randint(0, 20), random.randint(0, 20)
            x = x[oy:oy + CFG['H'], ox:ox + CFG['W']]
        x = (x - MEAN) / STD
        if self.train and random.random() < 0.5:                 # random erasing
            for _ in range(20):
                area = CFG['H'] * CFG['W'] * random.uniform(0.02, 0.4)
                ar = random.uniform(0.3, 3.3)
                h, w = int(round(math.sqrt(area * ar))), int(round(math.sqrt(area / ar)))
                if h < CFG['H'] and w < CFG['W']:
                    y0, x0 = random.randint(0, CFG['H'] - h), random.randint(0, CFG['W'] - w)
                    x[y0:y0 + h, x0:x0 + w] = 0
                    break
        x = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)
        y = self.lm[r['pid']] if self.lm else r['pid']
        return x, y, i


class PKSampler:
    def __init__(self, items, label_map, P_, K, iters, seed):
        self.by = collections.defaultdict(list)
        for i, r in enumerate(items):
            self.by[label_map[r['pid']]].append(i)
        self.P, self.K, self.iters, self.rng = min(P_, len(self.by)), K, iters, random.Random(seed)

    def __iter__(self):
        ids = list(self.by)
        for _ in range(self.iters):
            batch = []
            for pid in self.rng.sample(ids, self.P):
                pool = self.by[pid]
                batch += self.rng.sample(pool, self.K) if len(pool) >= self.K else self.rng.choices(pool, k=self.K)
            yield batch

    def __len__(self):
        return self.iters


# ================================================================ 2. model
def build_model(name, n_ids, pretrained=True):
    import torch, torch.nn as nn

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            if name == 'BoT-R50':
                import torchvision
                r = torchvision.models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
                r.layer4[0].conv2.stride = (1, 1)            # last stride = 1
                r.layer4[0].downsample[0].stride = (1, 1)
                s.backbone = nn.Sequential(*list(r.children())[:-2])
                s.pool = nn.AdaptiveAvgPool2d(1)
                dim = 2048
            else:
                import timm
                s.backbone = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=pretrained,
                                               img_size=(CFG['H'], CFG['W']), num_classes=0)
                s.pool = None
                dim = s.backbone.num_features
            s.bnneck = nn.BatchNorm1d(dim)
            s.bnneck.bias.requires_grad_(False)
            s.cls = nn.Linear(dim, n_ids, bias=False)
            nn.init.normal_(s.cls.weight, std=0.001)

        def forward(s, x):
            f = s.backbone(x)
            if s.pool is not None:
                f = s.pool(f).flatten(1)
            fb = s.bnneck(f)
            if s.training:
                return s.cls(fb), f
            return fb
    return Net()


def triplet_hard(feat, y, margin=0.3):
    import torch
    feat = feat.float()
    d = torch.cdist(feat, feat)
    same = y.unsqueeze(0) == y.unsqueeze(1)
    ap = (d * same.float()).max(1)[0]
    an = (d + same.float() * 1e6).min(1)[0]
    return torch.relu(ap - an + margin).mean()


def train_reid(rdir, model, items, crops, label_map, mc):
    import torch
    from torch.utils.data import DataLoader
    os.makedirs(rdir, exist_ok=True)
    if os.path.exists(f'{rdir}/TRAIN_DONE') and os.path.exists(f'{rdir}/best.pt'):
        log('Training already finished, continuing with evaluation')
        return
    model.cuda()
    params = [p for p in model.parameters() if p.requires_grad]
    if mc['opt'] == 'adam':
        opt = torch.optim.Adam(params, lr=mc['lr'], weight_decay=mc['wd'])
    else:
        opt = torch.optim.SGD(params, lr=mc['lr'], momentum=0.9, weight_decay=mc['wd'])
    E, Wu = mc['epochs'], mc['warmup']
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: 0.1 + 0.9 * e / Wu if e < Wu else 0.5 * (1 + math.cos(math.pi * (e - Wu) / max(1, E - Wu))))
    scaler = torch.amp.GradScaler('cuda')
    ce = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    state = dict(epoch=0, history=[])
    ck = _torch_load_safe(f'{rdir}/last.pt')
    if ck is not None:
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        scaler.load_state_dict(ck['scaler'])
        state = ck['state']
        log(f"Resuming from epoch {state['epoch']}")
    ds = ReIDDataset(items, crops, True, label_map)
    while state['epoch'] < E:
        ep = state['epoch']
        sampler = PKSampler(items, label_map, CFG['P_IDS'], CFG['K_IMGS'], CFG['ITERS'], seed=ep)
        dl = DataLoader(ds, batch_sampler=sampler, num_workers=CFG['WORKERS'], pin_memory=True)
        model.train()
        t0, tot = time.time(), 0.0
        for x, y, _ in dl:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16):
                logit, f = model(x)
                loss = ce(logit.float(), y) + triplet_hard(f, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
        sched.step()
        state['epoch'] = ep + 1
        state['history'].append(dict(epoch=ep + 1, loss=tot / len(sampler), sec=round(time.time() - t0)))
        _torch_save_atomic(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                                scaler=scaler.state_dict(), state=state), f'{rdir}/last.pt')
        save_json(state['history'], f'{rdir}/history.json')
        heartbeat(rdir)
        log(f"epoch {ep+1}/{E} loss {tot/len(sampler):.4f} ({time.time()-t0:.0f}s)")
    _torch_save_atomic({'model': model.state_dict()}, f'{rdir}/best.pt')
    open(f'{rdir}/TRAIN_DONE', 'w').write('ok')
    cleanup_checkpoints(rdir)


def extract(model, items, crops):
    import torch
    from torch.utils.data import DataLoader
    model.cuda().eval()
    dl = DataLoader(ReIDDataset(items, crops, False), batch_size=128, num_workers=CFG['WORKERS'])
    feats = []
    with torch.no_grad():
        for x, _, _ in dl:
            x = x.cuda()
            with torch.autocast('cuda', dtype=torch.float16):
                f = model(x).float() + model(torch.flip(x, [3])).float()
            feats.append(torch.nn.functional.normalize(f, dim=1).cpu())
    return torch.cat(feats).numpy()


def reid_metrics(qf, gf, q, g, mask_fn):
    """CMC Rank-1/5 and mAP; gallery items for which mask_fn(query, gallery) is False are excluded."""
    sim = qf @ gf.T
    r1, r5, aps = [], [], []
    for i, qi in enumerate(q):
        valid = np.array([mask_fn(qi, gj) for gj in g])
        if not valid.any():
            continue
        s = sim[i][valid]
        match = np.array([gj['pid'] == qi['pid'] for gj, ok in zip(g, valid) if ok])
        if not match.any():
            continue
        order = np.argsort(-s)
        m = match[order]
        first = np.argmax(m)
        r1.append(first < 1)
        r5.append(first < 5)
        hits = np.cumsum(m)
        prec = hits[m] / (np.nonzero(m)[0] + 1)
        aps.append(prec.mean())
    return (float(np.mean(r1)) if r1 else float('nan'), float(np.mean(r5)) if r5 else float('nan'),
            float(np.mean(aps)) if aps else float('nan'))


def evaluate(rdir, model, sp, crops, F):
    import torch
    model.load_state_dict(torch.load(f'{rdir}/best.pt', map_location='cpu', weights_only=False)['model'])
    q, g = sp['query'], sp['gallery']
    qf, gf = extract(model, q, crops), extract(model, g, crops)
    res = {}
    diff_video = lambda a, b: a['video'] != b['video']
    diff_loc = lambda a, b: a['location'] != b['location']
    for tag, sel in (('all', lambda r: True),
                     ('dis', lambda r: f"P{r['pid']:02d}" in DISABLED),
                     ('nondis', lambda r: f"P{r['pid']:02d}" not in DISABLED)):
        idx = [i for i, r in enumerate(q) if sel(r)]
        if not idx:
            continue
        r1, r5, mAP = reid_metrics(qf[idx], gf, [q[i] for i in idx], g, diff_video)
        res[f'rank1_{tag}'], res[f'rank5_{tag}'], res[f'mAP_{tag}'] = r1, r5, mAP
        if tag == 'all':
            r1x, _, mAPx = reid_metrics(qf[idx], gf, [q[i] for i in idx], g, diff_loc)
            res['rank1_crossloc'], res['mAP_crossloc'] = r1x, mAPx
    return res


# ================================================================ 3. runs
METRICS = ['rank1_all', 'mAP_all', 'rank1_crossloc', 'rank1_dis', 'rank1_nondis']


def main(P, runs=None, reverse=False):
    P.check(need_images=False)
    F = load_frames(P)
    build_crops(P, F)
    crops = load_crops(P)

    def do_run(run):
        sp = load_reid_split(P, run['split'])
        ids = sorted({r['pid'] for r in sp['train']})
        lm = {p: i for i, p in enumerate(ids)}
        log(f"train {len(sp['train'])} crops / {len(ids)} identities, query {len(sp['query'])}, gallery {len(sp['gallery'])}")
        rdir = run_dir(P, run)
        mc = CFG['MODEL'][run['model']]
        train_reid(rdir, build_model(run['model'], len(ids)), sp['train'], crops, lm, mc)
        m = evaluate(rdir, build_model(run['model'], len(ids)), sp, crops, F)
        mark_done(P, run, m)
        log(f"{run['run_id']} finished: Rank-1 {m['rank1_all']:.4f}, mAP {m['mAP_all']:.4f}")

    execute_runs(P, runs or run_list('reid'), do_run, reverse)
    return summarize(P, 'reid', METRICS)
