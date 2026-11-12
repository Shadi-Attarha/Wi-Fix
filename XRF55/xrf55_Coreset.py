"""
DPP-based Coreset Selection for XRF55 Dataset
===============================================
Corrected architecture: treats (270,1000) as (9,30,1000) — 
9 WiFi links as channels, 30 subcarriers × 1000 time as 2D plane.

WHY this is the right fix (parallel to Widar3.0):

  Widar mistake:   Conv2d(22,...) treated 22 TIMESTEPS as channels
  Widar fix:       Conv3d(1,...) with (1,22,20,20) — time as real axis

  XRF55 mistake:   Conv1d(270,...) treats 270=3×3×30 as flat channels
                   — loses subcarrier frequency ordering
  XRF55 fix:       reshape (270,1000) → (9,30,1000)
                   Conv2d(9,...) — 9 links as channels,
                   (30 subcarriers × 1000 time) as 2D frequency-time plane

  Physical justification:
  - 9 links (3 receivers × 3 antennas) are SIMULTANEOUS measurements
    with no ordering between them → correct as channels (like RGB)
  - 30 OFDM subcarriers have FREQUENCY ordering — adjacent subcarriers
    are more correlated (adjacent frequencies in OFDM spectrum)
    → frequency axis must be preserved, not flattened
  - 1000 packets have TEMPORAL ordering → time axis preserved
  - Conv2d kernel slides over (frequency × time) jointly, capturing
    both frequency-local and time-local CSI patterns

Run:
    # Edit DATA_PATH, TEXT_PATH at top, then:

    # Step A — train feature extractor (~15 min)
    python dpp_xrf55.py --mode train_extractor \
        --extractor_path xrf55_extractor.pt --extractor_epochs 30

    # Step B — select and evaluate
    python dpp_xrf55.py --mode select \
        --ipc 10 --method all --extractor_path xrf55_extractor.pt

    # Herding v2 only (best method from Widar experiments)
    python dpp_xrf55.py --mode select --ipc 10 --method herding_v2 \
        --herding_var_weight 0.2 --herding_diversity 0.0

Install: pip install dppy
"""

import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from tqdm import tqdm
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


# ─────────────────────────────────────────────────────────────
# CONFIG — edit these two paths
# ─────────────────────────────────────────────────────────────
DATA_PATH    = '/home/sattarha/data/XRF55-repo/dataset/XRF_dataset/'
TEXT_PATH    = '/home/sattarha/data/XRF55-repo/dataset/XRF_dataset/'
SCENE        = 'dml'
NUM_CLASSES  = 55
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASS_NAMES  = [str(i) for i in range(NUM_CLASSES)]

# Physical constants from paper
N_LINKS       = 9     # 3 receivers × 3 antennas
N_SUBCARRIERS = 30    # OFDM subcarriers
N_TIME        = 1000  # packets (200 Hz × ~5 s)


# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────
class XRF55_Dataset(Dataset):
    def __init__(self, data_path, text_path, is_train=True, scene='dml'):
        self.data_path = data_path
        self.is_train  = is_train
        self.mean      = 9.6302
        self.std       = 3.8425

        split         = 'train_data' if is_train else 'test_data'
        self.wifi_dir = os.path.join(data_path, split, 'WiFi')
        text_file     = os.path.join(text_path,
                                      f'{scene}_train.txt' if is_train
                                      else f'{scene}_val.txt')

        self.file_names = []
        self.labels     = []
        with open(text_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                self.file_names.append(parts[0])
                self.labels.append(int(parts[2]) - 1)  # 1-indexed → 0-indexed

        print(f"  {'Train' if is_train else 'Test'}: "
              f"{len(self.labels)} samples, "
              f"{len(set(self.labels))} classes")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        fname = self.file_names[idx]
        label = self.labels[idx]
        fpath = os.path.join(self.wifi_dir, fname + '.npy')

        x = np.load(fpath)              # (270, 1000) as stored
        x = (x - self.mean) / self.std  # normalize

        # KEY RESHAPE: (270, 1000) → (9, 30, 1000)
        x = x.reshape(N_LINKS, N_SUBCARRIERS, N_TIME)

        return torch.FloatTensor(x), label


# ─────────────────────────────────────────────────────────────
# BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────

class SEBlock2D(nn.Module):
    """
    Squeeze-and-Excitation: channel attention over the 9 antenna links.
    Learns which antenna combinations are most informative per gesture.
    reduction=8 keeps the bottleneck light; clamp to ≥8 avoids collapse
    in early stages where channel count is still 64.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.shape[:2]
        s = self.pool(x).view(b, c)
        s = self.fc(s).view(b, c, 1, 1)
        return x * s


class BasicBlock2D(nn.Module):

    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.se    = SEBlock2D(out_ch)

        # Projection shortcut only when shape changes
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + self.skip(x), inplace=True)





class XRF55_ResNet2D(nn.Module):
    def __init__(self, num_classes=55):
        super().__init__()

        # Stem: asymmetric kernel large along time, small along freq
        # (9, 30, 1000) → (64, 30, 500) → (64, 30, 250)
        self.stem = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=(3, 7), stride=(1, 2),
                      padding=(1, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        )

        # 4 residual stages — same depth as authors' xrf_ResNet
        self.stage1 = self._make_stage(64,  64,  n=2, stride=1)
        # → (B, 64,  30, 250)
        self.stage2 = self._make_stage(64,  128, n=2, stride=2)
        # → (B, 128, 15, 125)
        self.stage3 = self._make_stage(128, 256, n=2, stride=(1, 2))
        # → (B, 256, 15, 63)
        self.stage4 = self._make_stage(256, 512, n=2, stride=(1, 2))
        # → (B, 512, 15, 32)

        self.gap = nn.AdaptiveAvgPool2d(1)   # → (B, 512)
        self.fc  = nn.Linear(512, num_classes)

        self._init_weights()

    def _make_stage(self, in_ch, out_ch, n, stride):
        layers = [BasicBlock2D(in_ch, out_ch, stride=stride)]
        for _ in range(1, n):
            layers.append(BasicBlock2D(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x).squeeze(-1).squeeze(-1)
        return self.fc(x)

    def get_features(self, x):
        """
        512-dim L2-normalized features for coreset selection.
        Unit-sphere projection makes herding and kmeans_herding geometry
        well-calibrated across all 55 classes.
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x).squeeze(-1).squeeze(-1)       # (B, 512)
        return F.normalize(x, p=2, dim=1)             # unit hypersphere


# ─────────────────────────────────────────────────────────────
# EVAL MODEL  (same backbone + dropout before classifier)
# ─────────────────────────────────────────────────────────────
class XRF55_ResNet2D_Eval(nn.Module):
    def __init__(self, num_classes=55, dropout=0.3):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=(3, 7), stride=(1, 2),
                      padding=(1, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        )
        self.stage1 = self._make_stage(64,  64,  n=2, stride=1)
        self.stage2 = self._make_stage(64,  128, n=2, stride=2)
        self.stage3 = self._make_stage(128, 256, n=2, stride=(1, 2))
        self.stage4 = self._make_stage(256, 512, n=2, stride=(1, 2))
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.drop   = nn.Dropout(dropout)
        self.fc     = nn.Linear(512, num_classes)
        self._init_weights()

    def _make_stage(self, in_ch, out_ch, n, stride):
        layers = [BasicBlock2D(in_ch, out_ch, stride=stride)]
        for _ in range(1, n):
            layers.append(BasicBlock2D(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x).squeeze(-1).squeeze(-1)
        x = self.drop(x)
        return self.fc(x)


# ─────────────────────────────────────────────────────────────
# TRAIN EXTRACTOR
# ─────────────────────────────────────────────────────────────
def train_extractor(dst_train, dst_test, save_path,
                    epochs=30, batch_size=32):
    train_loader = DataLoader(dst_train, batch_size=batch_size,
                              shuffle=True, num_workers=4)
    test_loader  = DataLoader(dst_test,  batch_size=32,
                              shuffle=False, num_workers=4)

    model = XRF55_ResNet2D(NUM_CLASSES).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    best_acc = 0.0
    print(f"\nTraining on {len(dst_train)} samples ({NUM_CLASSES} classes)...")
    print(f"Input shape per sample: ({N_LINKS}, {N_SUBCARRIERS}, {N_TIME})")
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Acc':>10}")
    print("-" * 34)

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
            print(f"{epoch+1:>6}  {total_loss/len(train_loader):>12.4f}"
                  f"  {acc:>9.2f}%")
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), save_path)

    print(f"\nBest val accuracy: {best_acc:.2f}%")
    print(f"Extractor saved to: {save_path}")
    if best_acc < 50:
        print("WARNING: accuracy < 50% — try --extractor_epochs 60")
    return best_acc


# ─────────────────────────────────────────────────────────────
# EXTRACT FEATURES
# ─────────────────────────────────────────────────────────────
def extract_cnn_features(model, dataset, device, batch_size=16):
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4)
    model.eval()
    all_feats, all_y = [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc='Extracting CNN2D features'):
            x = x.to(device)
            feats = model.get_features(x)
            all_feats.append(feats.cpu())
            all_y.append(torch.tensor(y) if not isinstance(y, torch.Tensor)
                         else y)

    all_feats = torch.cat(all_feats).numpy()   # (N, 1024)
    all_y     = torch.cat(all_y).numpy()       # (N,)
    print(f"Extracted features: {all_feats.shape}")

    print("\nClass separation check (10 sample classes):")
    for c in range(min(10, NUM_CLASSES)):
        mask = all_y == c
        if mask.sum() == 0:
            continue
        print(f"  Class {c}: n={mask.sum()}  "
              f"mean_norm={np.linalg.norm(all_feats[mask].mean(0)):.3f}  "
              f"std={all_feats[mask].std():.3f}")

    return all_feats, all_y


# ═════════════════════════════════════════════════════════════
# SELECTION METHODS
# ═════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────
# DPP
# ─────────────────────────────────────────────────────────────
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

    for c in tqdm(range(num_classes), desc='DPP per class'):
        mask     = labels == c
        cls_feat = feat_n[mask]
        cls_idx  = np.where(mask)[0]

        if len(cls_feat) <= n_per_class:
            selected_indices.extend(cls_idx.tolist()); continue

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


# ─────────────────────────────────────────────────────────────
# K-CENTER
# ─────────────────────────────────────────────────────────────
def kcenter_greedy(features, n_select):
    selected = [np.random.randint(len(features))]
    dists    = np.full(len(features), np.inf)
    for _ in range(n_select - 1):
        new_d = np.linalg.norm(features - features[selected[-1]], axis=1)
        dists = np.minimum(dists, new_d)
        dists[selected] = -np.inf
        selected.append(int(np.argmax(dists)))
    return selected


def kcenter_select(features, labels, n_per_class, num_classes):
    selected_indices = []
    for c in tqdm(range(num_classes), desc='K-center per class'):
        mask     = labels == c
        cls_feat = features[mask]
        cls_idx  = np.where(mask)[0]
        chosen = (cls_idx.tolist() if len(cls_feat) <= n_per_class
                  else cls_idx[kcenter_greedy(cls_feat, n_per_class)].tolist())
        selected_indices.extend(chosen)
    print(f"Total selected: {len(selected_indices)}")
    return selected_indices


# ─────────────────────────────────────────────────────────────
# K-MEANS
# ─────────────────────────────────────────────────────────────
def kmeans_select(features, labels, n_per_class, num_classes,
                  random_state=42):
    from sklearn.cluster import KMeans
    selected_indices = []
    for c in tqdm(range(num_classes), desc='K-means per class'):
        mask     = labels == c
        cls_feat = features[mask]
        cls_idx  = np.where(mask)[0]
        if len(cls_feat) <= n_per_class:
            selected_indices.extend(cls_idx.tolist()); continue
        km = KMeans(n_clusters=n_per_class, random_state=random_state,
                    n_init=10)
        km.fit(cls_feat)
        chosen = []
        for k in range(n_per_class):
            mk = km.labels_ == k
            if mk.sum() == 0:
                continue
            fk = cls_feat[mk]
            ik = np.where(mk)[0]
            chosen.append(ik[np.linalg.norm(
                fk - km.cluster_centers_[k], axis=1).argmin()])
        selected_indices.extend(cls_idx[chosen].tolist())
    print(f"Total selected: {len(selected_indices)}")
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
# EVALUATE DISTILLED DATASET
# ─────────────────────────────────────────────────────────────
def evaluate_distilled(selected_idx, dst_train, dst_test,
                        num_classes, device,
                        epochs=500, n_runs=3,
                        label_smoothing=0.1,
                        warmup_epochs=10,
                        dropout=0.3):
    print(f"  Building distilled dataset ({len(selected_idx)} samples)...")
    dist_x = torch.stack([dst_train[i][0] for i in selected_idx])
    dist_y = torch.tensor([dst_train[i][1] for i in selected_idx],
                           dtype=torch.long)
    dist_loader = DataLoader(
        torch.utils.data.TensorDataset(dist_x, dist_y),
        batch_size=min(32, len(dist_x)), shuffle=True
    )

    print("  Building test loader...")
    test_x = torch.stack([dst_test[i][0] for i in range(len(dst_test))])
    test_y = torch.tensor([dst_test[i][1] for i in range(len(dst_test))],
                           dtype=torch.long)
    test_loader = DataLoader(
        torch.utils.data.TensorDataset(test_x, test_y),
        batch_size=32, shuffle=False
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    accs = []
    for run in range(n_runs):
        torch.manual_seed(run * 42)
        model = XRF55_ResNet2D_Eval(num_classes, dropout=dropout).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3,
                                  weight_decay=1e-4)
        sched = make_scheduler_with_warmup(opt, warmup_epochs, epochs)

        best_run_acc = 0.0
        for epoch in range(epochs):
            model.train()
            for xb, yb in dist_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = criterion(model(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()

            if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
                model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for xb, yb in test_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        correct += (model(xb).argmax(1) == yb).sum().item()
                        total   += len(yb)
                acc = 100.0 * correct / total
                best_run_acc = max(best_run_acc, acc)

        accs.append(best_run_acc)
        print(f"    Run {run+1}/{n_runs}: {best_run_acc:.2f}%")

    return float(np.mean(accs)), float(np.std(accs))

class XRF55_CNN2D_Eval(nn.Module):
    def __init__(self, num_classes=55, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(9,   64,  kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,  128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(256)
        self.pool  = nn.MaxPool2d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.drop  = nn.Dropout(dropout)
        self.fc1   = nn.Linear(256 * 4 * 4, 1024)
        self.fc2   = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)
# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main(args):
    print(f"Device:       {DEVICE}")
    print(f"Architecture: Conv2d(9,...) over ({N_SUBCARRIERS},{N_TIME}) "
          f"— {N_LINKS} links as channels")

    dst_train = XRF55_Dataset(DATA_PATH, TEXT_PATH,
                               is_train=True,  scene=SCENE)
    dst_test  = XRF55_Dataset(DATA_PATH, TEXT_PATH,
                               is_train=False, scene=SCENE)
    print(f"Train: {len(dst_train)}  |  Test: {len(dst_test)}")

    if args.mode == 'train_extractor':
        print("\n=== TRAINING FEATURE EXTRACTOR ON FULL XRF55 DATA ===")
        best = train_extractor(dst_train, dst_test,
                               save_path=args.extractor_path,
                               epochs=args.extractor_epochs)
        print(f"\nDone. Best accuracy: {best:.2f}%")
        print("Now run with --mode select.")
        return

    print(f"\nIPC: {args.ipc} per class ({args.ipc * NUM_CLASSES} total)")
    print(f"Method: {args.method}")

    if not os.path.exists(args.extractor_path):
        raise FileNotFoundError(
            f"Extractor not found at {args.extractor_path}.\n"
            f"Run: python {__file__} --mode train_extractor"
        )

    print(f"\nLoading extractor from {args.extractor_path}...")
    extractor = XRF55_ResNet2D(NUM_CLASSES).to(DEVICE)
    extractor.load_state_dict(
        torch.load(args.extractor_path, map_location=DEVICE,
                   weights_only=False)
    )
    extractor.eval()
    print("Extractor loaded ✓")

    print("\nExtracting discriminative CNN2D features...")
    features, labels = extract_cnn_features(
        extractor, dst_train, DEVICE, batch_size=args.batch_size)

    results = {}

    def run_eval(name, idx):
        torch.save({'indices': idx, 'method': name, 'ipc': args.ipc},
                   f'xrf55_{name}_ipc{args.ipc}.pt')
        acc, std = evaluate_distilled(
            idx, dst_train, dst_test, NUM_CLASSES, DEVICE,
            epochs=args.eval_epochs,
            label_smoothing=args.label_smoothing,
            warmup_epochs=args.warmup_epochs,
            dropout=args.dropout)
        results[name] = (acc, std)
        print(f"{name} → {acc:.2f}% ± {std:.2f}%")

    if args.method in ('herding', 'all'):
        print("\n" + "="*55)
        print("HERDING V1 — greedy class-mean matching")
        print("="*55)
        run_eval('herding', herding_select(
            features, labels, args.ipc, NUM_CLASSES))

    if args.method in ('herding_v2', 'all'):
        print("\n" + "="*55)
        print(f"HERDING V2 — mean+variance matching  "
              f"(var_weight={args.herding_var_weight}, "
              f"diversity={args.herding_diversity})")
        print("="*55)
        run_eval('herding_v2', herding_select_v2(
            features, labels, args.ipc, NUM_CLASSES,
            var_weight=args.herding_var_weight,
            diversity_weight=args.herding_diversity))

    if args.method in ('herding_v3', 'all'):
        print("\n" + "="*55)
        print(f"HERDING V3 — cluster-aware herding  "
              f"(var_weight={args.herding_var_weight}, "
              f"mult={args.herding_subclusters_mult})")
        print("  Stage 1: k-means sub-clustering (discovers modes)")
        print("  Stage 2: herding-v2 over centroids (mean+var matching)")
        print("  Stage 3: nearest real sample per chosen centroid")
        print("="*55)
        run_eval('herding_v3', herding_select_v3(
            features, labels, args.ipc, NUM_CLASSES,
            var_weight=args.herding_var_weight,
            n_subclusters_mult=args.herding_subclusters_mult))

    if args.method in ('DPP', 'all'):
        print("\n" + "="*55)
        print("DPP SELECTION")
        print("="*55)
        run_eval('DPP', dpp_select(
            features, labels, args.ipc, NUM_CLASSES))

    if args.method in ('kmeans', 'all'):
        print("\n" + "="*55)
        print("K-MEANS SELECTION")
        print("="*55)
        run_eval('kmeans', kmeans_select(
            features, labels, args.ipc, NUM_CLASSES))

    if args.method in ('kmeans_herding', 'all'):
        print("\n" + "="*55)
        print(f"KMEANS-HERDING — prototype-constrained herding  "
              f"(order={args.kh_cluster_order}, "
              f"mean_w={args.kh_mean_weight}, "
              f"var_w={args.kh_var_weight}, "
              f"div_w={args.kh_inter_div_weight})")
        print("="*55)
        run_eval('kmeans_herding', kmeans_herding_select(
            features, labels, args.ipc, NUM_CLASSES,
            mean_weight=args.kh_mean_weight,
            var_weight=args.kh_var_weight,
            inter_div_weight=args.kh_inter_div_weight,
            cluster_order_by=args.kh_cluster_order))

    if args.method in ('kcenter', 'all'):
        print("\n" + "="*55)
        print("K-CENTER SELECTION")
        print("="*55)
        run_eval('kcenter', kcenter_select(
            features, labels, args.ipc, NUM_CLASSES))

    method_type = {
        'herding':        'mean matching',
        'herding_v2':     'mean+var matching',
        'herding_v3':     'cluster-aware herding',
        'DPP':            'diverse set',
        'kmeans':         'prototypes',
        'kmeans_herding': 'prototype-constrained herding',
        'kcenter':        'max coverage',
    }

    print("\n" + "="*55)
    print(f"SUMMARY  IPC={args.ipc}  "
          f"({args.ipc * NUM_CLASSES} / {len(dst_train)} samples)")
    print("="*55)
    print(f"{'Method':<14}  {'Type':<20}  {'Accuracy':>10}  {'Std':>6}")
    print("-" * 56)
    for method, (acc, std) in sorted(results.items(),
                                      key=lambda x: -x[1][0]):
        mtype = method_type.get(method, '')
        print(f"{method:<14}  {mtype:<20}  {acc:>9.2f}%  {std:>5.2f}%")
    print("="*55)
    print("Indices saved to xrf55_<method>_ipc<N>.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='select',
                        choices=['train_extractor', 'select'])
    parser.add_argument('--extractor_path',   type=str,
                        default='xrf55_extractorResNet.pt')
    parser.add_argument('--extractor_epochs', type=int, default=30)
    parser.add_argument('--ipc',    type=int, default=10)
    parser.add_argument('--method', type=str, default='all',
                        choices=['herding',
                                 'DPP', 'kmeans', 'kcenter',
                                 'kmeans_herding', 'all'])
    parser.add_argument('--eval_epochs',     type=int,   default=500)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--warmup_epochs',   type=int,   default=10)
    parser.add_argument('--dropout',         type=float, default=0.3)
    parser.add_argument('--batch_size',      type=int,   default=16,
                        help='Reduce if OOM — XRF55 samples are large')
    # herding_v2 / herding_v3 shared parameters
    parser.add_argument('--herding_var_weight', type=float, default=0.2,
                        help='Variance-matching weight [0,1). '
                             '0.0 = mean only. 0.1–0.3 recommended.')
    parser.add_argument('--herding_diversity',  type=float, default=0.0,
                        help='Diversity repulsion weight for herding_v2 [0,1).')
    # herding_v3 only
    parser.add_argument('--herding_subclusters_mult', type=int, default=3,
                        help='Over-clustering factor M for herding_v3.')
    # kmeans_herding parameters
    parser.add_argument('--kh_mean_weight',    type=float, default=0.2,
                        help='Herding mean-matching weight for kmeans_herding.')
    parser.add_argument('--kh_var_weight',     type=float, default=0.1,
                        help='Herding variance-matching weight for kmeans_herding. '
                             '0.0–0.15 recommended.')
    parser.add_argument('--kh_inter_div_weight', type=float, default=0.0,
                        help='Inter-cluster diversity weight for kmeans_herding. '
                             '0.0=off (default). 0.1–0.2 if adjacent clusters overlap.')
    parser.add_argument('--kh_cluster_order',  type=str,   default='variance',
                        choices=['variance', 'size', 'distance'],
                        help='Cluster processing order for kmeans_herding. '
                             'variance=high-variance first (recommended for XRF55), '
                             'size=dense first, distance=farthest from mean first.')
    args = parser.parse_args()
    main(args)
