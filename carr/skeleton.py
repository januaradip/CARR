"""
Skeleton-based activity recognition (Table 10): LSTM, ST-GCN and CTR-GCN on the nine CS/CL/CG splits (runs R19-R45).
Step 1 (once): 17 COCO keypoints per frame with YOLOv8m-Pose, matched to the annotated box -> <work>/cache/pose/
Step 2: training on windows of 64 frames (stride 32), keypoints normalised by the annotated box.
"""
import os, io, math, random, concurrent.futures as cf
import numpy as np
from .common import *

CFG = dict(
    POSE_MODEL='yolov8m-pose.pt', POSE_IMGSZ=640, POSE_BATCH=32,
    WIN=64,            # window length (frames)
    STRIDE=32,         # window stride
    EPOCHS=60, PATIENCE=12, BATCH=64, WORKERS=2, WARMUP=3,
    LR={'LSTM': 1e-3, 'ST-GCN': 1e-3, 'CTR-GCN': 1e-3}, WD=1e-4,
)
FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]   # left-right swap of COCO-17 keypoints
EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
         (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


# ================================================================ 1. pose extraction
def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def extract_pose(P, F):
    d = f'{P.cache}/pose'
    os.makedirs(d, exist_ok=True)
    todo = [v for v in sorted(F['videos']) if not os.path.exists(f'{d}/{v}.npz')]
    if not todo:
        log('Pose cache complete')
        return
    log(f'Extracting poses for {len(todo)} remaining videos...')
    from ultralytics import YOLO
    model = YOLO(CFG['POSE_MODEL'])
    store = ImageStore(P)
    pool = cf.ThreadPoolExecutor(4)
    for vi, v in enumerate(todo):
        files = F['videos'][v]['files']
        kp = np.zeros((len(files), 17, 3), np.float32)
        matched = np.zeros(len(files), bool)
        size = None
        for s in range(0, len(files), CFG['POSE_BATCH']):
            chunk = files[s:s + CFG['POSE_BATCH']]
            imgs = list(pool.map(lambda f: store.open(f, draft=(1280, 1280)), chunk))
            res = model.predict(imgs, imgsz=CFG['POSE_IMGSZ'], conf=0.1, verbose=False, half=True)
            for j, (f, r) in enumerate(zip(chunk, res)):
                H, W = r.orig_shape
                size = (W, H)
                x, y, w, h = F['frames'][f]['box']
                gt = ((x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H)
                if r.keypoints is None or len(r.boxes) == 0:
                    continue
                bx = r.boxes.xyxy.cpu().numpy()
                ious = [_iou(gt, b) for b in bx]
                k = int(np.argmax(ious))
                if ious[k] < 0.1:
                    continue
                d_ = r.keypoints.data[k].cpu().numpy()          # (17,3) pixels + confidence
                kp[s + j, :, 0] = d_[:, 0] / W
                kp[s + j, :, 1] = d_[:, 1] / H
                kp[s + j, :, 2] = d_[:, 2]
                matched[s + j] = True
        b = io.BytesIO()
        np.savez_compressed(b, kp=kp, matched=matched, size=np.array(size or (1, 1)))
        atomic_write_bytes(f'{d}/{v}.npz', b.getvalue())
        if (vi + 1) % 10 == 0 or vi + 1 == len(todo):
            log(f'pose {vi+1}/{len(todo)} videos (match rate of last video {matched.mean():.2f})')


def load_sequences(P, F):
    """Keypoints normalised by the annotated box: box centre = 0, box height = 1."""
    seq = {}
    for v, meta in F['videos'].items():
        z = np.load(f'{P.cache}/pose/{v}.npz')
        kp, (W, H) = z['kp'].copy(), z['size']
        boxes = np.array([F['frames'][f]['box'] for f in meta['files']], np.float32)
        cx, cy, bh = boxes[:, 0] * W, boxes[:, 1] * H, np.maximum(boxes[:, 3] * H, 1e-3)
        valid = kp[:, :, 2] > 0
        xs = (kp[:, :, 0] * W - cx[:, None]) / bh[:, None]
        ys = (kp[:, :, 1] * H - cy[:, None]) / bh[:, None]
        out = np.stack([np.where(valid, xs, 0), np.where(valid, ys, 0), kp[:, :, 2]], -1).astype(np.float32)
        seq[v] = out            # (T,17,3)
    return seq


def make_windows(videos, seq, F):
    W, S = CFG['WIN'], CFG['STRIDE']
    items = []
    for v in videos:
        L = len(seq[v])
        if L >= W:
            starts = list(range(0, L - W + 1, S))
            if starts[-1] != L - W:
                starts.append(L - W)
            for s in starts:
                items.append((v, np.arange(s, s + W)))
        else:
            items.append((v, np.arange(W) % L))
    return items


class SkelDataset:
    def __init__(self, items, seq, F, train):
        self.items, self.seq, self.F, self.train = items, seq, F, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import torch
        v, idx = self.items[i]
        x = self.seq[v][idx].copy()                   # (T,17,3)
        if self.train:
            if random.random() < 0.5:
                x = x[:, FLIP]
                x[:, :, 0] *= -1
            a = math.radians(random.uniform(-10, 10))
            sc = random.uniform(0.9, 1.1)
            R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], np.float32) * sc
            x[:, :, :2] = x[:, :, :2] @ R.T
        x = torch.from_numpy(x).permute(2, 0, 1).contiguous()   # (C=3,T,V=17)
        return x, self.F['videos'][v]['cls'], i


# ================================================================ 2. model
def graph_A():
    V = 17
    adj = [[] for _ in range(V)]
    for a, b in EDGES:
        adj[a].append(b)
        adj[b].append(a)
    dist = [-1] * V
    dist[0] = 0
    q = [0]
    while q:
        u = q.pop(0)
        for w in adj[u]:
            if dist[w] < 0:
                dist[w] = dist[u] + 1
                q.append(w)
    A = np.zeros((3, V, V), np.float32)           # A[k, u, v]: v -> u
    for u in range(V):
        A[0, u, u] = 1
        for v in adj[u]:
            if dist[v] == dist[u]:
                A[0, u, v] = 1
            elif dist[v] < dist[u]:
                A[1, u, v] = 1                    # from neighbours closer to the centre joint
            else:
                A[2, u, v] = 1                    # from neighbours farther from the centre joint
    for k in range(3):
        deg = A[k].sum(1, keepdims=True)
        deg[deg == 0] = 1
        A[k] = A[k] / deg
    return A


def build_model(name):
    import torch
    import torch.nn as nn

    A = torch.tensor(graph_A())

    class DataBN(nn.Module):
        def __init__(s, C=3, V=17):
            super().__init__()
            s.bn = nn.BatchNorm1d(C * V)

        def forward(s, x):                         # (N,C,T,V)
            N, C, T, V = x.shape
            x = x.permute(0, 3, 1, 2).reshape(N, V * C, T)
            return s.bn(x).reshape(N, V, C, T).permute(0, 2, 3, 1).contiguous()

    # ---------------- LSTM
    class LSTMNet(nn.Module):
        def __init__(s):
            super().__init__()
            s.bn = DataBN()
            s.lstm = nn.LSTM(51, 128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
            s.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, 6))

        def forward(s, x):
            x = s.bn(x)
            N, C, T, V = x.shape
            x = x.permute(0, 2, 1, 3).reshape(N, T, C * V)
            h, _ = s.lstm(x)
            return s.fc(h.mean(1))

    # ---------------- ST-GCN
    class GCN(nn.Module):
        def __init__(s, cin, cout):
            super().__init__()
            s.K = A.shape[0]
            s.conv = nn.Conv2d(cin, cout * s.K, 1)
            s.register_buffer('A', A.clone())
            s.imp = nn.Parameter(torch.ones_like(A))

        def forward(s, x):
            N, _, T, V = x.shape
            x = s.conv(x).view(N, s.K, -1, T, V)
            return torch.einsum('nkctv,kuv->nctu', x, s.A * s.imp)

    class STBlock(nn.Module):
        def __init__(s, cin, cout, stride=1, residual=True):
            super().__init__()
            s.gcn = GCN(cin, cout)
            s.tcn = nn.Sequential(nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                                  nn.Conv2d(cout, cout, (9, 1), (stride, 1), (4, 0)),
                                  nn.BatchNorm2d(cout), nn.Dropout(0.1))
            if not residual:
                s.res = lambda x: 0
            elif cin == cout and stride == 1:
                s.res = lambda x: x
            else:
                s.res = nn.Sequential(nn.Conv2d(cin, cout, 1, (stride, 1)), nn.BatchNorm2d(cout))
            s.relu = nn.ReLU(inplace=True)

        def forward(s, x):
            return s.relu(s.tcn(s.gcn(x)) + s.res(x))

    class STGCN(nn.Module):
        def __init__(s):
            super().__init__()
            s.bn = DataBN()
            cfg = [(3, 64, 1, False), (64, 64, 1, True), (64, 64, 1, True), (64, 64, 1, True),
                   (64, 128, 2, True), (128, 128, 1, True), (128, 128, 1, True),
                   (128, 256, 2, True), (256, 256, 1, True), (256, 256, 1, True)]
            s.blocks = nn.ModuleList(STBlock(*c) for c in cfg)
            s.fc = nn.Linear(256, 6)

        def forward(s, x):
            x = s.bn(x)
            for b in s.blocks:
                x = b(x)
            return s.fc(x.mean((2, 3)))

    # ---------------- CTR-GCN
    class CTRGC(nn.Module):
        def __init__(s, cin, cout):
            super().__init__()
            rel = 8 if cin <= 16 else cin // 8
            s.c1, s.c2 = nn.Conv2d(cin, rel, 1), nn.Conv2d(cin, rel, 1)
            s.c3, s.c4 = nn.Conv2d(cin, cout, 1), nn.Conv2d(rel, cout, 1)

        def forward(s, x, Ak, alpha):
            x1, x2, x3 = s.c1(x).mean(-2), s.c2(x).mean(-2), s.c3(x)
            r = torch.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))          # (N,R,V,V)
            r = s.c4(r) * alpha + Ak.unsqueeze(0).unsqueeze(0)            # (N,C,V,V)
            return torch.einsum('ncuv,nctv->nctu', r, x3)

    class UnitGCN(nn.Module):
        def __init__(s, cin, cout):
            super().__init__()
            s.PA = nn.Parameter(A.clone())
            s.alpha = nn.Parameter(torch.zeros(1))
            s.convs = nn.ModuleList(CTRGC(cin, cout) for _ in range(A.shape[0]))
            s.down = (lambda x: x) if cin == cout else nn.Sequential(nn.Conv2d(cin, cout, 1), nn.BatchNorm2d(cout))
            s.bn = nn.BatchNorm2d(cout)
            nn.init.constant_(s.bn.weight, 1e-6)
            s.relu = nn.ReLU(inplace=True)

        def forward(s, x):
            y = sum(c(x, s.PA[i], s.alpha) for i, c in enumerate(s.convs))
            return s.relu(s.bn(y) + s.down(x))

    class MSTCN(nn.Module):
        def __init__(s, cin, cout, stride=1, ks=5, dil=(1, 2)):
            super().__init__()
            nb = len(dil) + 2
            bc = cout // nb
            s.br = nn.ModuleList()
            for d in dil:
                s.br.append(nn.Sequential(nn.Conv2d(cin, bc, 1), nn.BatchNorm2d(bc), nn.ReLU(inplace=True),
                                          nn.Conv2d(bc, bc, (ks, 1), (stride, 1), ((ks - 1) * d // 2, 0), (d, 1)),
                                          nn.BatchNorm2d(bc)))
            s.br.append(nn.Sequential(nn.Conv2d(cin, bc, 1), nn.BatchNorm2d(bc), nn.ReLU(inplace=True),
                                      nn.MaxPool2d((3, 1), (stride, 1), (1, 0)), nn.BatchNorm2d(bc)))
            s.br.append(nn.Sequential(nn.Conv2d(cin, cout - bc * (nb - 1), 1, (stride, 1)),
                                      nn.BatchNorm2d(cout - bc * (nb - 1))))

        def forward(s, x):
            return torch.cat([b(x) for b in s.br], 1)

    class CTRBlock(nn.Module):
        def __init__(s, cin, cout, stride=1, residual=True):
            super().__init__()
            s.gcn = UnitGCN(cin, cout)
            s.tcn = MSTCN(cout, cout, stride)
            if not residual:
                s.res = lambda x: 0
            elif cin == cout and stride == 1:
                s.res = lambda x: x
            else:
                s.res = nn.Sequential(nn.Conv2d(cin, cout, 1, (stride, 1)), nn.BatchNorm2d(cout))
            s.relu = nn.ReLU(inplace=True)

        def forward(s, x):
            return s.relu(s.tcn(s.gcn(x)) + s.res(x))

    class CTRGCN(nn.Module):
        def __init__(s):
            super().__init__()
            s.bn = DataBN()
            cfg = [(3, 64, 1, False), (64, 64, 1, True), (64, 64, 1, True), (64, 64, 1, True),
                   (64, 128, 2, True), (128, 128, 1, True), (128, 128, 1, True),
                   (128, 256, 2, True), (256, 256, 1, True), (256, 256, 1, True)]
            s.blocks = nn.ModuleList(CTRBlock(*c) for c in cfg)
            s.drop = nn.Dropout(0.1)
            s.fc = nn.Linear(256, 6)

        def forward(s, x):
            x = s.bn(x)
            for b in s.blocks:
                x = b(x)
            return s.fc(s.drop(x.mean((2, 3))))

    return {'LSTM': LSTMNet, 'ST-GCN': STGCN, 'CTR-GCN': CTRGCN}[name]()


# ================================================================ 3. runs
METRICS = ['win_acc_all', 'win_f1_all', 'vid_acc_all', 'win_f1_dis', 'win_f1_nondis']


def main(P, runs=None, reverse=False):
    P.check(need_images=False)
    F = load_frames(P)
    extract_pose(P, F)
    seq = load_sequences(P, F)

    def do_run(run):
        parts = load_split(P, run['split'])
        tr, va, te = (make_windows(parts[k], seq, F) for k in ('train', 'val', 'test'))
        log(f'windows: train {len(tr)}, val {len(va)}, test {len(te)}')
        cfg = dict(epochs=CFG['EPOCHS'], lr=CFG['LR'][run['model']], wd=CFG['WD'], batch=CFG['BATCH'],
                   patience=CFG['PATIENCE'], workers=CFG['WORKERS'], warmup=CFG['WARMUP'], amp=True)
        rdir = run_dir(P, run)
        train_classifier(rdir, build_model(run['model']), SkelDataset(tr, seq, F, True), SkelDataset(va, seq, F, False), cfg)
        m = evaluate_classifier(rdir, build_model(run['model']), SkelDataset(te, seq, F, False),
                                [v for v, _ in te], F['videos'], cfg)
        mark_done(P, run, m)
        log(f"{run['run_id']} finished: accuracy {m['win_acc_all']:.4f}, macro-F1 {m['win_f1_all']:.4f}")

    execute_runs(P, runs or run_list('skeleton'), do_run, reverse)
    return summarize(P, 'skeleton', METRICS)
