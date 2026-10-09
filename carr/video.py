"""
Video-based activity recognition (Table 10): R(2+1)D-18 and VideoMAE-B (Kinetics-400 pre-trained) on the nine
CS/CL/CG splits (runs R46-R63).
Step 1 (once): square person crops of every second frame, 160 x 160 px -> <work>/cache/vcrops/
Step 2: fine-tuning on 16-frame clips that cover 64 source frames.
"""
import os, io, math, random, concurrent.futures as cf
import numpy as np
from .common import *

CFG = dict(
    CROP=160, EXPAND=1.2, FRAME_STEP=2,   # keep 1 of every 2 frames
    WIN=32, TRAIN_STRIDE=32, EVAL_STRIDE=16, CLIP=16,   # in cached frames (x2 = source frames)
    WORKERS=2, PATIENCE=5, WARMUP=1,
    MODEL={
        'R2Plus1D-18': dict(size=112, batch=16, accum=1, lr=2e-4, epochs=20,
                            mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989)),
        'VideoMAE-B': dict(size=224, batch=4, accum=4, lr=5e-5, epochs=15,
                           mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    },
    WD=0.05,
)


# ================================================================ 1. crop cache
def build_crops(P, F):
    d = f'{P.cache}/vcrops'
    os.makedirs(d, exist_ok=True)
    todo = [v for v in sorted(F['videos']) if not os.path.exists(f'{d}/{v}.pkl')]
    if not todo:
        log('Video crop cache complete')
        return
    log(f'Cropping {len(todo)} remaining videos...')
    store = ImageStore(P)

    def work(f):
        im = store.open(f, draft=(960, 960))
        c = crop_box(im, F['frames'][f]['box'], CFG['EXPAND'], square=True).resize((CFG['CROP'], CFG['CROP']))
        return jpeg_bytes(c, 90)

    with cf.ThreadPoolExecutor(4) as ex:
        for i, v in enumerate(todo):
            files = F['videos'][v]['files'][::CFG['FRAME_STEP']]
            save_pickle(list(ex.map(work, files)), f'{d}/{v}.pkl')
            if (i + 1) % 20 == 0 or i + 1 == len(todo):
                log(f'crops {i+1}/{len(todo)} videos')


def load_crops(P, videos):
    d = f'{P.cache}/vcrops'
    return {v: load_pickle(f'{d}/{v}.pkl') for v in videos}


def make_windows(videos, crops, stride):
    W = CFG['WIN']
    items = []
    for v in videos:
        L = len(crops[v])
        if L >= W:
            starts = list(range(0, L - W + 1, stride))
            if starts[-1] != L - W:
                starts.append(L - W)
            items += [(v, s) for s in starts]
        else:
            items.append((v, 0))
    return items


class ClipDataset:
    def __init__(self, items, crops, F, mcfg, train):
        self.items, self.crops, self.F, self.m, self.train = items, crops, F, mcfg, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import torch
        from PIL import Image
        v, s = self.items[i]
        seqs = self.crops[v]
        L, W, K = len(seqs), CFG['WIN'], CFG['CLIP']
        pos = [(s + j * W // K) % L for j in range(K)]
        S = self.m['size']
        if self.train:
            sc = random.uniform(0.8, 1.0)
            cw = int(CFG['CROP'] * sc)
            ox, oy = random.randint(0, CFG['CROP'] - cw), random.randint(0, CFG['CROP'] - cw)
            flip = random.random() < 0.5
        else:
            cw, ox, oy, flip = CFG['CROP'], 0, 0, False
        frames = []
        for p in pos:
            im = Image.open(io.BytesIO(seqs[p])).convert('RGB').crop((ox, oy, ox + cw, oy + cw)).resize((S, S))
            if flip:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            frames.append(np.asarray(im, np.float32) / 255.0)
        x = (np.stack(frames) - np.array(self.m['mean'], np.float32)) / np.array(self.m['std'], np.float32)
        x = torch.from_numpy(x).permute(3, 0, 1, 2).contiguous()     # (C,T,H,W)
        return x, self.F['videos'][v]['cls'], i


# ================================================================ 2. model
def build_model(name, pretrained=True):
    import torch.nn as nn
    if name == 'R2Plus1D-18':
        from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
        m = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, 6)
        return m

    from transformers import VideoMAEForVideoClassification, VideoMAEConfig

    class VMAE(nn.Module):
        def __init__(s):
            super().__init__()
            if pretrained:
                s.m = VideoMAEForVideoClassification.from_pretrained(
                    'MCG-NJU/videomae-base-finetuned-kinetics', num_labels=6, ignore_mismatched_sizes=True)
            else:
                s.m = VideoMAEForVideoClassification(VideoMAEConfig(num_labels=6))

        def forward(s, x):                                      # (N,C,T,H,W) -> (N,T,C,H,W)
            return s.m(pixel_values=x.permute(0, 2, 1, 3, 4)).logits
    return VMAE()


# ================================================================ 3. runs
METRICS = ['win_acc_all', 'win_f1_all', 'vid_acc_all', 'win_f1_dis', 'win_f1_nondis']


def main(P, runs=None, reverse=False):
    P.check(need_images=False)
    F = load_frames(P)
    build_crops(P, F)
    log('Loading crops into memory...')
    crops = load_crops(P, sorted(F['videos']))

    def do_run(run):
        mc = CFG['MODEL'][run['model']]
        parts = load_split(P, run['split'])
        tr = make_windows(parts['train'], crops, CFG['TRAIN_STRIDE'])
        va = make_windows(parts['val'], crops, CFG['EVAL_STRIDE'])
        te = make_windows(parts['test'], crops, CFG['EVAL_STRIDE'])
        log(f'clips: train {len(tr)}, val {len(va)}, test {len(te)}')
        cfg = dict(epochs=mc['epochs'], lr=mc['lr'], wd=CFG['WD'], batch=mc['batch'], accum=mc['accum'],
                   patience=CFG['PATIENCE'], workers=CFG['WORKERS'], warmup=CFG['WARMUP'], amp=True)
        rdir = run_dir(P, run)
        train_classifier(rdir, build_model(run['model']), ClipDataset(tr, crops, F, mc, True),
                         ClipDataset(va, crops, F, mc, False), cfg)
        m = evaluate_classifier(rdir, build_model(run['model']), ClipDataset(te, crops, F, mc, False),
                                [v for v, _ in te], F['videos'], cfg)
        mark_done(P, run, m)
        log(f"{run['run_id']} finished: accuracy {m['win_acc_all']:.4f}, macro-F1 {m['win_f1_all']:.4f}")

    execute_runs(P, runs or run_list('video'), do_run, reverse)
    return summarize(P, 'video', METRICS)
