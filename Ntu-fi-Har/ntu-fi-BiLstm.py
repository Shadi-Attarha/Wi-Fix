"""
distill_ntufi.py
================
Dataset distillation for NTU-Fi HAR using the unified kmeans_herding
selection technique (same as XRF55).

WHY kmeans_herding FOR NTU-Fi (not plain k-means):
  NTU-Fi HAR activities have genuine intra-class sub-modes:
    · walk / run  — different speeds, gait cycles, body sizes
    · fall        — different directions, recovery styles
    · box/circle  — different execution extents and speeds
  Plain k-means covers cluster regions but ignores distribution shape.
  Plain herding assumes unimodal distribution — wrong for HAR activities.
  kmeans_herding: k-means guarantees cluster coverage,
                  herding within each cluster matches mean + variance,
                  cluster_order='variance' prioritises temporal diversity.

INPUT SHAPE:  (3, 114, 500)  — 3 antenna pairs × 114 subcarriers × 500 time
FEATURES:     512-d from penultimate layer of NTUFi_CNN2D_Eval
CLASSES:      6 (clean, fall, walk, box, circle, run)

Run
---
    # Step A — train feature extractor
    python distill_ntufi.py --mode train_extractor \
        --extractor_path ntufi_extractor.pt --extractor_epochs 50

    # Step B — select and evaluate
    python distill_ntufi.py --mode select \
        --ipc 10 --extractor_path ntufi_extractor.pt

    # Compare all methods
    python distill_ntufi.py --mode select \
        --ipc 10 --method all --extractor_path ntufi_extractor.pt
"""

import os
import glob
import argparse
from sklearn.preprocessing import StandardScaler
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from tqdm import tqdm
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
DATA_ROOT   = '/home/sattarha/data/WiFi-CSI-Sensing-Benchmark/NTU-Fi_HAR/'
NUM_CLASSES = 6
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

# Alphabetical order — matches glob folder sorting in NTUFi_Dataset
CLASS_NAMES = ['box', 'circle', 'clean', 'fall', 'run', 'walk']


# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────
class NTUFi_Dataset(Dataset):
    def __init__(self, root_dir, modal='CSIamp'):
        self.modal     = modal
        self.data_list = glob.glob(root_dir + '/*/*.mat')
        self.folder    = sorted(glob.glob(root_dir + '/*/'))
        self.category  = {
            self.folder[i].split('/')[-2]: i
            for i in range(len(self.folder))
        }
        print(f'  Classes found: {self.category}')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        path = self.data_list[idx]
        y    = self.category[path.split('/')[-2]]
        x    = sio.loadmat(path)[self.modal]    # (342, 2000)
        x    = (x - 42.3199) / 4.9802          # normalize
        x    = x[:, ::4]                        # (342, 500) — every 4th packet
        x    = x.reshape(3, 114, 500)           # (antenna_pairs, subcarriers, time)
        return torch.FloatTensor(x), y


# ─────────────────────────────────────────────────────────────
# MODEL

# ─────────────────────────────────────────────────────────────
class NTUFi_BiLSTM(nn.Module):
    """
    Feature extractor — NO dropout. Trained on the full dataset.
    Input: (B, 3, 114, 500) — 3 antenna pairs as channels, flattened
    with subcarriers into a 342-dim feature per timestep (500 timesteps).
    get_features() returns 512-d, L2-normalized penultimate activations.
    """
    def __init__(self, num_classes=6, input_size=342, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1,
                             batch_first=False, bidirectional=True)
        self.fc1  = nn.Linear(hidden_size * 2, 512)   # *2 for bidirectional concat
        self.fc2  = nn.Linear(512, num_classes)

    def _encode(self, x):
        B = x.size(0)
        x = x.view(B, 342, 500)
        x = x.permute(2, 0, 1)                      # (seq_len, batch, input_size)
        _, (ht, ct) = self.lstm(x)
        h = torch.cat((ht[-2], ht[-1]), dim=1)       # (B, hidden_size*2)
        return h

    def forward(self, x):
        h = self._encode(x)
        h = F.relu(self.fc1(h))
        return self.fc2(h)

    @torch.no_grad()
    def get_features(self, x):
        """
        512-d penultimate features — relu on fc1, no dropout anywhere
        in this model. Deterministic: identical output on repeated
        calls with the same input. Used for coreset selection.
        """
        h = self._encode(x)
        feat = F.relu(self.fc1(h))
        return F.normalize(feat, p=2, dim=1)


class NTUFi_BiLSTM_Eval(nn.Module):
    """
    Standard BiLSTM for NTU-Fi HAR.
    Input: (B, 3, 114, 500) — 3 antenna pairs as channels, flattened
    with subcarriers into a 342-dim feature per timestep (500 timesteps).
    get_features() returns 512-d penultimate activations (relu, no dropout).
    """
    def __init__(self, num_classes=6, input_size=342, hidden_size=64, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1,
                             batch_first=False, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc1  = nn.Linear(hidden_size * 2, 512)   # *2 for bidirectional concat
        self.fc2  = nn.Linear(512, num_classes)

    def _encode(self, x):
        B = x.size(0)
        x = x.view(B, 342, 500)
        x = x.permute(2, 0, 1)                      # (seq_len, batch, input_size)
        _, (ht, ct) = self.lstm(x)
        h = torch.cat((ht[-2], ht[-1]), dim=1)       # (B, hidden_size*2)
        return h

    def forward(self, x):
        h = self._encode(x)
        h = self.drop(F.relu(self.fc1(h)))
        return self.fc2(h)

    @torch.no_grad()
    def get_features(self, x):
        """
        512-d penultimate features — relu on fc1, NO dropout.
        Deterministic: identical output on repeated calls with same input.
        Used by kmeans_herding for coreset selection.
        """
        h = self._encode(x)
        feat = F.relu(self.fc1(h))
        return F.normalize(feat, p=2, dim=1)     # (B, 512)

# ─────────────────────────────────────────────────────────────
# TRAIN EXTRACTOR

def train_extractor(dst_train, dst_test, save_path, model_cls, model_kwargs=None,
                     epochs=50, batch_size=32):
    model_kwargs = model_kwargs or {}
    train_loader = DataLoader(dst_train, batch_size=batch_size,
                              shuffle=True, num_workers=4)
    test_loader  = DataLoader(dst_test,  batch_size=32,
                              shuffle=False, num_workers=4)

    model = model_cls(**model_kwargs).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    # ...rest unchanged...

    best_acc = 0.0
    print(f'\nTraining on {len(dst_train)} samples, {NUM_CLASSES} classes')
    print(f'Input shape per sample: (3, 114, 500)')
    print(f'{"Epoch":>6}  {"Train Loss":>12}  {"Val Acc":>10}')
    print('-' * 34)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        sched.step()

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    correct += (model(x).argmax(1) == y).sum().item()
                    total   += len(y)
            acc = 100.0 * correct / total
            print(f'{epoch+1:>6}  {total_loss/len(train_loader):>12.4f}'
                  f'  {acc:>9.2f}%')
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), save_path)

    print(f'\nBest val accuracy: {best_acc:.2f}%')
    print(f'Extractor saved → {save_path}')
    if best_acc < 60:
        print('WARNING: accuracy < 60% — try --extractor_epochs 80')
    return best_acc


# ─────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────
def extract_features(model, dataset, device, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4)
    model.eval()
    all_feats, all_y = [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc='Extracting features', ncols=70):
            all_feats.append(model.get_features(x.to(device)).cpu())
            all_y.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))

    feats = torch.cat(all_feats).numpy()    # (N, 512)
    lbls  = torch.cat(all_y).numpy()        # (N,)
    print(f'  Features: {feats.shape}  |  Labels: {lbls.shape}')

    # Quick class separation check
    print('\n  Class separation check:')
    for c in range(NUM_CLASSES):
        m = lbls == c
        print(f'    Class {c} ({CLASS_NAMES[c]:>7}):  '
              f'n={m.sum():4d}  '
              f'mean_norm={np.linalg.norm(feats[m].mean(0)):.3f}  '
              f'std={feats[m].std():.3f}')
    return feats, lbls


# ─────────────────────────────────────────────────────────────s
# SELECTION METHODS
# ─────────────────────────────────────────────────────────────

# ── K-Means (baseline — what you currently use for NTU-Fi) ───
def kmeans_select(features, labels, n_per_class):
    selected = []
    for c in range(NUM_CLASSES):
        mask    = labels == c
        cls_f   = features[mask]
        cls_idx = np.where(mask)[0]
        N_c     = len(cls_f)
        print(f'  Class {c} ({CLASS_NAMES[c]:>7}): {N_c} → {n_per_class}')

        if N_c <= n_per_class:
            selected.extend(cls_idx.tolist()); continue

        km = KMeans(n_clusters=n_per_class, random_state=42,
                    n_init=10, max_iter=300)
        km.fit(cls_f)
        chosen = []
        for k in range(n_per_class):
            m = km.labels_ == k
            if m.sum() == 0: continue
            chosen.append(np.where(m)[0][
                np.linalg.norm(cls_f[m] - km.cluster_centers_[k],
                               axis=1).argmin()])
        selected.extend(cls_idx[chosen].tolist())

    print(f'Total selected: {len(selected)}')
    return selected


# ── Herding V1 (mean matching only) ─────────────────────────
def herding_select(features, labels, n_per_class):
    feat_n = features
    selected = []

    for c in range(NUM_CLASSES):
        mask    = labels == c
        cls_f   = feat_n[mask]
        cls_idx = np.where(mask)[0]
        N_c     = len(cls_f)
        print(f'  Class {c} ({CLASS_NAMES[c]:>7}): {N_c} → {n_per_class}')

        if N_c <= n_per_class:
            selected.extend(cls_idx.tolist()); continue

        class_mean  = cls_f.mean(axis=0)
        chosen, chosen_set = [], set()
        running_sum = np.zeros_like(class_mean)

        for k in range(1, n_per_class + 1):
            new_means = (running_sum[None, :] + cls_f) / k
            dists     = np.linalg.norm(class_mean - new_means, axis=1)
            dists[list(chosen_set)] = np.inf
            best = int(np.argmin(dists))
            chosen.append(best); chosen_set.add(best)
            running_sum += cls_f[best]

        selected.extend(cls_idx[chosen].tolist())

    print(f'Total selected: {len(selected)}')
    return selected






def dpp_select(features, labels, n_per_class, num_classes,
               random_state=42):
    try:
        from dppy.finite_dpps import FiniteDPP
    except ImportError:
        raise ImportError("Run: pip install dppy")

    np.random.seed(random_state)
    scaler = StandardScaler()
    feat_n = scaler.fit_transform(features)
    selected_indices = []
    print("I am at DPP")

    for c in tqdm(range(num_classes), desc='DPP per class'):
        mask     = labels == c
        cls_feat = feat_n[mask]
        cls_idx  = np.where(mask)[0]

        if len(cls_feat) <= n_per_class:
            selected_indices.extend(cls_idx.tolist())
            continue

        pw_dist = cdist(cls_feat, cls_feat, metric='euclidean')
        sigma   = np.median(pw_dist[pw_dist > 0])
        L       = np.exp(-pw_dist**2 / (2 * sigma**2))

        try:
            dpp = FiniteDPP(kernel_type='likelihood', L=L)
            dpp.sample_exact_k_dpp(size=n_per_class)
            chosen = list(dpp.list_of_samples[0])
        except Exception as e:
            print(f"\n  Class {c}: DPP failed ({e}), using k-center")
            chosen = kcenter_greedy(cls_feat, n_per_class)

        selected_indices.extend(cls_idx[chosen].tolist())

    print(f"Total selected: {len(selected_indices)}")
    return selected_indices


def random_select(features, labels, n_per_class, num_classes, random_state=42):
    rng = np.random.default_rng(random_state)
    selected_indices = []

    for c in range(num_classes):
        mask = labels == c
        cls_idx = np.where(mask)[0]

        print(f"  Class {c} ({CLASS_NAMES[c]}): {len(cls_idx)} -> {n_per_class}")

        if len(cls_idx) <= n_per_class:
            selected_indices.extend(cls_idx.tolist())
        else:
            chosen = rng.choice(cls_idx, size=n_per_class, replace=False)
            selected_indices.extend(chosen.tolist())

    return selected_indices

# ─────────────────────────────────────────────────────────────
# WARMUP SCHEDULER
# ─────────────────────────────────────────────────────────────
def make_scheduler_with_warmup(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = ((epoch - warmup_epochs) /
                    max(1, total_epochs - warmup_epochs))
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────
# EVALUATE DISTILLED SUBSET
# ─────────────────────────────────────────────────────────────
def evaluate_distilled(selected_idx, dst_train, dst_test,
                        epochs=100, n_runs=3,
                        label_smoothing=0.1,
                        warmup_epochs=10,
                        dropout=0.3):
    print(f'\n  Building distilled dataset ({len(selected_idx)} samples)...')
    dist_x = torch.stack([dst_train[i][0] for i in selected_idx])
    dist_y = torch.tensor([dst_train[i][1] for i in selected_idx],
                           dtype=torch.long)
    dist_loader = DataLoader(
        torch.utils.data.TensorDataset(dist_x, dist_y),
        batch_size=min(32, len(dist_x)), shuffle=True)

    print('  Building test loader...')
    test_x = torch.stack([dst_test[i][0] for i in range(len(dst_test))])
    test_y = torch.tensor([dst_test[i][1] for i in range(len(dst_test))],
                           dtype=torch.long)
    test_loader = DataLoader(
        torch.utils.data.TensorDataset(test_x, test_y),
        batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    accs = []

    for run in range(n_runs):
        torch.manual_seed(run * 42)
        model = NTUFi_BiLSTM_Eval(NUM_CLASSES, dropout=dropout).to(DEVICE)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3,
                                  weight_decay=1e-4)
        sched = make_scheduler_with_warmup(opt, warmup_epochs, epochs)

        best_run_acc = 0.0
        for epoch in range(epochs):
            model.train()
            for xb, yb in dist_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                loss = criterion(model(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()

            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for xb, yb in test_loader:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        correct += (model(xb).argmax(1) == yb).sum().item()
                        total   += len(yb)
                acc = 100.0 * correct / total
                best_run_acc = max(best_run_acc, acc)
                print( f'    Run {run+1}/{n_runs} | ' f'Epoch {epoch+1:>4}/{epochs} | ' f'Acc: {acc:>6.2f}% | ' f'Best: {best_run_acc:>6.2f}%')

        accs.append(best_run_acc)
        print(f'    Run {run+1}/{n_runs}: {best_run_acc:.2f}%')

    return float(np.mean(accs)), float(np.std(accs))


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main(args):
    print(f'\n{"="*58}')
    print('  NTU-Fi HAR — Dataset Distillation')
    print(f'  Device : {DEVICE}')
    print(f'  Data   : {args.data_root}')
    print(f'{"="*58}\n')

    train_root = os.path.join(args.data_root, 'train_amp')
    test_root  = os.path.join(args.data_root, 'test_amp')

    dst_train = NTUFi_Dataset(train_root)
    dst_test  = NTUFi_Dataset(test_root)
    print(f'Train: {len(dst_train)}  |  Test: {len(dst_test)}\n')

    if args.mode == 'train_extractor':
      print('=== TRAINING FEATURE EXTRACTOR ===')
      best = train_extractor(
        dst_train, dst_test,
        save_path=args.extractor_path,
        model_cls=NTUFi_BiLSTM,
        model_kwargs={'num_classes': NUM_CLASSES},
        epochs=args.extractor_epochs)
      print(f'\nDone. Best accuracy: {best:.2f}%')
      print('Now run with --mode select.')
      return

    # ── Load extractor ────────────────────────────────────────
    if not os.path.exists(args.extractor_path):
        raise FileNotFoundError(
            f'Extractor not found at {args.extractor_path}\n'
            f'Run: python {__file__} --mode train_extractor')

    print(f'Loading extractor from {args.extractor_path}...')
    extractor = NTUFi_BiLSTM_Eval(NUM_CLASSES).to(DEVICE)
    extractor.load_state_dict(
        torch.load(args.extractor_path, map_location=DEVICE,
                   weights_only=False))
    extractor.eval()
    print('Extractor loaded ✓\n')

    features, labels = extract_features(
        extractor, dst_train, DEVICE, batch_size=32)

    if args.feat_dim < features.shape[1]:
       print("featured reduced")

    
    print(f'\nIPC = {args.ipc}  '
          f'({args.ipc * NUM_CLASSES} / {len(dst_train)} samples)\n')

    results = {}

    def run_eval(name, idx):
        torch.save({'indices': idx, 'method': name, 'ipc': args.ipc},
                   f'ntufi_{name}_ipc{args.ipc}.pt')
        acc, std = evaluate_distilled(
            idx, dst_train, dst_test,
            epochs=args.eval_epochs,
            label_smoothing=args.label_smoothing,
            warmup_epochs=args.warmup_epochs,
            dropout=args.dropout)
        results[name] = (acc, std)
        print(f'  {name:<18} → {acc:.2f}% ± {std:.2f}%\n')
 
    
    if args.method in ('DPP', 'all'):
        print("\n" + "="*55)
        print("DPP SELECTION")
        print("="*55)
        run_eval('DPP', dpp_select(features, labels,
                                            args.ipc, NUM_CLASSES))
     
    if args.method in ('random', 'all'):
        print('=' * 55)
        print('Random  (current NTU-Fi baseline)')
        print('=' * 55)
        run_eval('random', random_select(features, labels,
                                            args.ipc, NUM_CLASSES))
    
    if args.method in ('kmeans', 'all'):
        print('=' * 55)
        print('K-MEANS  (current NTU-Fi baseline)')
        print('=' * 55)
        run_eval('kmeans', kmeans_select(features, labels, args.ipc))

    if args.method in ('herding', 'all'):
        print('=' * 55)
        print('HERDING V1  (mean matching only)')
        print('=' * 55)
        run_eval('herding', herding_select(features, labels, args.ipc))


    
    
    


            
 
    # ── Summary ───────────────────────────────────────────────
    method_desc = {
        'kmeans':         'cluster prototypes (baseline)',
        'herding':        'mean matching only',
        'kmeans_herding': 'prototype + mean + variance (unified)',
    }
    print('\n' + '=' * 55)
    print(f'SUMMARY  IPC={args.ipc}  '
          f'({args.ipc * NUM_CLASSES} / {len(dst_train)} samples)')
    print('=' * 55)
    print(f'{"Method":<18}  {"Description":<35}  {"Acc":>8}  {"Std":>6}')
    print('-' * 72)
    for method, (acc, std) in sorted(results.items(),
                                      key=lambda x: -x[1][0]):
        desc = method_desc.get(method, '')
        print(f'{method:<18}  {desc:<35}  {acc:>7.2f}%  {std:>5.2f}%')
    print('=' * 55)
    print('Indices saved to ntufi_<method>_ipc<N>.pt')


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', default='select',
                   choices=['train_extractor', 'select'])
    p.add_argument('--data_root',        default=DATA_ROOT)
    p.add_argument('--method', default='kmeans',
               choices=['kmeans', 'herding', 'all',
                        'random', 'DPP'])   # add 'entropy'
    p.add_argument('--extractor_path',   default='BiLstm_extractor_ntu_normlized.pt')
    p.add_argument('--feat_dim', type=int, default=512,
                    help='PCA reduced features')
    p.add_argument('--extractor_epochs', type=int,   default=70)
    p.add_argument('--ipc',              type=int,   default=10)
    p.add_argument('--eval_epochs',      type=int,   default=500)
    p.add_argument('--label_smoothing',  type=float, default=0.1)
    p.add_argument('--warmup_epochs',    type=int,   default=10)
    p.add_argument('--dropout',          type=float, default=0.3)
    main(p.parse_args())
