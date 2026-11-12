"""
DPP-based Coreset Selection for XRF55 Dataset
===============================================
Methods available:
  --method random   : random subset selection (new baseline)
  --method DPP      : diversity-aware DPP selection
  --method kmeans   : k-means centroid selection
  --method kcenter  : greedy k-center selection
  --method all      : run all four methods and compare

Run:
    # Step A — train feature extractor (~15 min, one-time)
    python xrf55_CNN2D.py --mode train_extractor \
        --extractor_path xrf55_extractor.pt --extractor_epochs 30

    # Step B — random selection only
    python xrf55_CNN2D.py --mode select \
        --ipc 10 --method random \
        --extractor_path xrf55_extractor.pt

    # Step B — all methods for comparison
    python xrf55_CNN2D.py --mode select \
        --ipc 10 --method all \
        --extractor_path xrf55_extractor.pt

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
import numpy as np
import warnings
from sklearn.mixture import GaussianMixture
from sklearn.exceptions import ConvergenceWarning


# ─────────────────────────────────────────────────────────────
# CONFIG — edit these two paths
# ─────────────────────────────────────────────────────────────
DATA_PATH    = '/dataset/XRF_dataset/'
TEXT_PATH    = '/dataset/XRF_dataset/'
SCENE        = 'dml'
NUM_CLASSES  = 55
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASS_NAMES  = [str(i) for i in range(NUM_CLASSES)]

N_LINKS       = 9     # 3 receivers × 3 antennas
N_SUBCARRIERS = 30    # OFDM subcarriers
N_TIME        = 1000  # packets (200 Hz × ~5s)


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
                self.labels.append(int(parts[2]) - 1)

        print(f"  {'Train' if is_train else 'Test'}: "
              f"{len(self.labels)} samples, "
              f"{len(set(self.labels))} classes")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        fname = self.file_names[idx]
        label = self.labels[idx]
        fpath = os.path.join(self.wifi_dir, fname + '.npy')

        x = np.load(fpath, allow_pickle=False)
        x = np.array(x)
        x = (x - self.mean) / self.std
        x = x.reshape(N_LINKS, N_SUBCARRIERS, N_TIME)  # (9, 30, 1000)

        return torch.FloatTensor(x), label


# ─────────────────────────────────────────────────────────────
# MODEL — XRF55_CNN2D  (feature extractor)
# ─────────────────────────────────────────────────────────────
class XRF55_CNN2D(nn.Module):
    def __init__(self, num_classes=55):
        super().__init__()
        self.conv1 = nn.Conv2d(9,   64,  kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,  128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(256)
        self.pool  = nn.MaxPool2d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def get_features(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return F.relu(self.fc1(x))                         # (B, 1024)


# ─────────────────────────────────────────────────────────────
# EVAL MODEL — same + dropout for small subsets
# ─────────────────────────────────────────────────────────────
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
# TRAIN EXTRACTOR ON FULL DATA
# ─────────────────────────────────────────────────────────────
def train_extractor(dst_train, dst_test, save_path,
                    epochs=30, batch_size=32):
    train_loader = DataLoader(dst_train, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    test_loader  = DataLoader(dst_test,  batch_size=32,
                              shuffle=False, num_workers=2)

    model = XRF55_CNN2D(NUM_CLASSES).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    best_acc = 0.0
    print(f"\nTraining on {len(dst_train)} samples ({NUM_CLASSES} classes)...")
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Acc':>10}")
    print("-" * 34)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
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
    return best_acc


# ─────────────────────────────────────────────────────────────
# EXTRACT FEATURES
# ─────────────────────────────────────────────────────────────
def extract_cnn_features(model, dataset, device, batch_size=16):
    """
    NOTE: random selection does NOT need features — it selects
    directly from raw data indices. This function is only called
    for DPP, kmeans, and kcenter methods.
    """
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0)
    model.eval()
    all_feats, all_y = [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc='Extracting CNN2D features'):
            x = x.to(device)
            feats = model.get_features(x)
            all_feats.append(feats.cpu())
            all_y.append(torch.tensor(y) if not isinstance(y, torch.Tensor)
                         else y)

    all_feats = torch.cat(all_feats).numpy()
    all_y     = torch.cat(all_y).numpy()
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


# ─────────────────────────────────────────────────────────────
# SELECTION METHODS
# ─────────────────────────────────────────────────────────────

def random_select(labels, n_per_class, num_classes, random_state=42):
    """
    RANDOM SELECTION — the simplest possible baseline.

    WHY this is important:
    Random selection sets the lower bound for any selection method.
    If DPP, k-means, or k-center do not outperform random selection,
    it means the feature space is not discriminative enough to guide
    meaningful diversity-aware selection.

    HOW it works:
    For each class independently, randomly sample n_per_class indices
    without replacement. No feature extraction needed — operates
    directly on class label arrays.

    WHY per-class (not global):
    Global random sampling can produce class-imbalanced subsets
    when class sizes differ. Per-class sampling guarantees exactly
    n_per_class samples from every class, making it a fair comparison
    against DPP and k-means which also select per-class.

    random_state: set for reproducibility — same seed = same subset.
    Change seed (--random_seed) to measure variance of random selection.
    """
    rng = np.random.RandomState(random_state)
    selected_indices = []

    for c in range(num_classes):
        class_mask = np.array(labels) == c
        class_idx  = np.where(class_mask)[0]

        print(f"  Class {c}: {len(class_idx)} samples → "
              f"selecting {n_per_class} randomly")

        if len(class_idx) <= n_per_class:
            # Not enough samples — take all of them
            selected_indices.extend(class_idx.tolist())
        else:
            # Random sample WITHOUT replacement
            chosen = rng.choice(class_idx, size=n_per_class, replace=False)
            selected_indices.extend(chosen.tolist())

    print(f"Total selected: {len(selected_indices)}")
    return selected_indices


def herding_select(features, labels, n_per_class, num_classes):
    """
    Original herding: greedy class-mean matching (1st moment only).
    Included as a baseline to compare against herding_v2.
    """
    sc     = StandardScaler()
    feat_n = sc.fit_transform(features)
    selected = []

    for c in range(num_classes):
        mask    = labels == c
        cls_f   = feat_n[mask]
        cls_idx = np.where(mask)[0]
        N_c     = len(cls_f)
        print(f"  Class {c} ({CLASS_NAMES[c]}): {N_c} → {n_per_class}")

        if N_c <= n_per_class:
            selected.extend(cls_idx.tolist()); continue

        class_mean  = cls_f.mean(axis=0)
        chosen      = []
        chosen_set  = set()
        running_sum = np.zeros_like(class_mean)

        for k in range(1, n_per_class + 1):
            new_means = (running_sum[None, :] + cls_f) / k
            dists     = np.linalg.norm(class_mean[None, :] - new_means, axis=1)
            dists[list(chosen_set)] = np.inf
            best = int(np.argmin(dists))
            chosen.append(best); chosen_set.add(best)
            running_sum += cls_f[best]

        err = np.linalg.norm(class_mean - cls_f[chosen].mean(axis=0))
        print(f"    Mean approx error: {err:.4f}  (0=perfect)")
        selected.extend(cls_idx[chosen].tolist())

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


def kmeans_select(features, labels, n_per_class, num_classes,
                  random_state=42):
    from sklearn.cluster import KMeans
    selected_indices = []
    for c in tqdm(range(num_classes), desc='K-means per class'):
        mask     = labels == c
        cls_feat = features[mask]
        cls_idx  = np.where(mask)[0]
        if len(cls_feat) <= n_per_class:
            selected_indices.extend(cls_idx.tolist())
            continue
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

import numpy as np
import warnings
from sklearn.mixture import GaussianMixture
from sklearn.exceptions import ConvergenceWarning
###########
from sklearn.decomposition import PCA

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
        model = XRF55_CNN2D_Eval(num_classes, dropout=dropout).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3,
                                  weight_decay=1e-4)
        sched = make_scheduler_with_warmup(opt, warmup_epochs, epochs)

        best_run_acc = 0.0
        for epoch in range(epochs):
            model.train()
            for xb, yb in dist_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = criterion(model(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
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


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main(args):
    print(f"Device: {DEVICE}")

    dst_train = XRF55_Dataset(DATA_PATH, TEXT_PATH,
                               is_train=True,  scene=SCENE)
    dst_test  = XRF55_Dataset(DATA_PATH, TEXT_PATH,
                               is_train=False, scene=SCENE)
    print(f"Train: {len(dst_train)}  |  Test: {len(dst_test)}")

    # ── Mode: train extractor ─────────────────────────────────
    if args.mode == 'train_extractor':
        print("\n=== TRAINING FEATURE EXTRACTOR ON FULL XRF55 DATA ===")
        best = train_extractor(dst_train, dst_test,
                               save_path=args.extractor_path,
                               epochs=args.extractor_epochs)
        print(f"\nDone. Best accuracy: {best:.2f}%")
        print("Now run with --mode select.")
        return

    # ── Mode: select ──────────────────────────────────────────
    print(f"\nIPC: {args.ipc} per class ({args.ipc * NUM_CLASSES} total)")
    print(f"Method: {args.method}")

    # Determine which methods need feature extraction
    needs_features = any(m in args.method
                         for m in ('DPP', 'kmeans', 'kcenter', 'all', 'herding'))

    # Random selection does NOT need the extractor or feature extraction
    # Only load them if a feature-based method is requested
    features, labels_feat = None, None

    if needs_features:
        if not os.path.exists(args.extractor_path):
            raise FileNotFoundError(
                f"Extractor not found at {args.extractor_path}.\n"
                f"Run first:\n"
                f"  python {__file__} --mode train_extractor\n\n"
                f"NOTE: --method random does NOT need the extractor.\n"
                f"You can run random selection without training it:\n"
                f"  python {__file__} --mode select "
                f"--method random --ipc {args.ipc}"
            )

        print(f"\nLoading extractor from {args.extractor_path}...")
        extractor = XRF55_CNN2D(NUM_CLASSES).to(DEVICE)
        extractor.load_state_dict(
            torch.load(args.extractor_path, map_location=DEVICE,
                       weights_only=False)
        )
        extractor.eval()
        print("Extractor loaded ✓")

        print("\nExtracting discriminative CNN2D features...")
        features, labels_feat = extract_cnn_features(
            extractor, dst_train, DEVICE, batch_size=args.batch_size)

    # Get raw labels for random selection (no features needed)
    raw_labels = np.array(dst_train.labels)

    results = {}

    # ── RANDOM SELECTION ──────────────────────────────────────
    if args.method in ('random', 'all'):
        print("\n" + "="*55)
        print("RANDOM SELECTION  (lower-bound baseline)")
        print("="*55)
        print("WHY: establishes the lower bound — any method that does")
        print("not outperform random selection provides no benefit over")
        print("simply picking samples at random.\n")

        idx = random_select(raw_labels, args.ipc, NUM_CLASSES,
                            random_state=args.random_seed)
        torch.save({'indices': idx, 'method': 'random',
                    'ipc': args.ipc, 'seed': args.random_seed},
                   f'xrf55_random_ipc{args.ipc}.pt')
        acc, std = evaluate_distilled(
            idx, dst_train, dst_test, NUM_CLASSES, DEVICE,
            epochs=args.eval_epochs,
            label_smoothing=args.label_smoothing,
            warmup_epochs=args.warmup_epochs,
            dropout=args.dropout)
        results['random'] = (acc, std)
        print(f"Random → {acc:.2f}% ± {std:.2f}%")

    
    if args.method in ('herding', 'all'):
        print("\n" + "="*55)
        print("HERDING V1 — greedy class-mean matching")
        print("="*55)
        idx = herding_select(
            features, labels_feat, args.ipc, NUM_CLASSES)
        torch.save({'indices': idx, 'method': 'herding',
                    'ipc': args.ipc, 'seed': args.random_seed},
                   f'xrf55_herding_ipc{args.ipc}.pt')
        acc, std = evaluate_distilled(
            idx, dst_train, dst_test, NUM_CLASSES, DEVICE,
            epochs=args.eval_epochs,
            label_smoothing=args.label_smoothing,
            warmup_epochs=args.warmup_epochs,
            dropout=args.dropout)
        results['herding'] = (acc, std)
        print(f"herding → {acc:.2f}% ± {std:.2f}%")

    
    if args.method in ('DPP', 'all'):
        print("\n" + "="*55)
        print("DPP SELECTION")
        print("="*55)
        idx = dpp_select(features, labels_feat, args.ipc, NUM_CLASSES)
        torch.save({'indices': idx, 'method': 'DPP', 'ipc': args.ipc},
                   f'xrf55_DPP_ipc{args.ipc}.pt')
        acc, std = evaluate_distilled(
            idx, dst_train, dst_test, NUM_CLASSES, DEVICE,
            epochs=args.eval_epochs,
            label_smoothing=args.label_smoothing,
            warmup_epochs=args.warmup_epochs,
            dropout=args.dropout)
        results['DPP'] = (acc, std)
        print(f"DPP → {acc:.2f}% ± {std:.2f}%")

    # ── K-MEANS SELECTION ─────────────────────────────────────
    if args.method in ('kmeans', 'all'):
        print("\n" + "="*55)
        print("K-MEANS SELECTION")
        print("="*55)
        idx = kmeans_select(features, labels_feat, args.ipc, NUM_CLASSES)
        torch.save({'indices': idx, 'method': 'kmeans', 'ipc': args.ipc},
                   f'xrf55_kmeans_ipc{args.ipc}.pt')
        acc, std = evaluate_distilled(
            idx, dst_train, dst_test, NUM_CLASSES, DEVICE,
            epochs=args.eval_epochs,
            label_smoothing=args.label_smoothing,
            warmup_epochs=args.warmup_epochs,
            dropout=args.dropout)
        results['kmeans'] = (acc, std)
        print(f"K-means → {acc:.2f}% ± {std:.2f}%")

    # ── K-CENTER SELECTION ────────────────────────────────────
    if args.method in ('kcenter', 'all'):
        print("\n" + "="*55)
        print("K-CENTER SELECTION")
        print("="*55)
        idx = kcenter_select(features, labels_feat, args.ipc, NUM_CLASSES)
        torch.save({'indices': idx, 'method': 'kcenter', 'ipc': args.ipc},
                   f'xrf55_kcenter_ipc{args.ipc}.pt')
        acc, std = evaluate_distilled(
            idx, dst_train, dst_test, NUM_CLASSES, DEVICE,
            epochs=args.eval_epochs,
            label_smoothing=args.label_smoothing,
            warmup_epochs=args.warmup_epochs,
            dropout=args.dropout)
        results['kcenter'] = (acc, std)
        print(f"K-center → {acc:.2f}% ± {std:.2f}%")

    # ── SUMMARY ───────────────────────────────────────────────
    print("\n" + "="*55)
    print(f"SUMMARY  IPC={args.ipc}  "
          f"({args.ipc * NUM_CLASSES} / {len(dst_train)} samples)")
    print("="*55)
    print(f"{'Method':<12}  {'Accuracy':>10}  {'Std':>6}")
    print("-" * 34)

    # Sort by accuracy descending, always show random last for reference
    non_random = {k: v for k, v in results.items() if k != 'random'}
    for method, (acc, std) in sorted(non_random.items(),
                                      key=lambda x: -x[1][0]):
        print(f"{method:<12}  {acc:>9.2f}%  {std:>5.2f}%")
    if 'random' in results:
        acc, std = results['random']
        print("-" * 34)
        print(f"{'random':<12}  {acc:>9.2f}%  {std:>5.2f}%  ← lower bound")

    print("="*55)

    if 'random' in results and len(non_random) > 0:
        best_method = max(non_random.items(), key=lambda x: x[1][0])
        gap = best_method[1][0] - results['random'][0]
        print(f"\nBest method ({best_method[0]}) outperforms random "
              f"by {gap:.2f}% — "
              f"{'meaningful gain' if gap > 2 else 'marginal gain'}.")

    print("\nIndices saved to xrf55_<method>_ipc<N>.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='select',
                        choices=['train_extractor', 'select'])
    parser.add_argument('--extractor_path',   type=str,
                        default='xrf55_extractor.pt')
    parser.add_argument('--extractor_epochs', type=int, default=30)
    parser.add_argument('--ipc',    type=int, default=10)

    # ── method choices now include 'random' ───────────────────
    parser.add_argument('--method', type=str, default='all',
                        choices=['random', 'DPP', 'kmeans', 'kcenter', 'all','herding'],
                        help=(
                            'random  : random per-class sampling (no extractor needed)\n'
                            'DPP     : diversity-aware DPP selection\n'
                            'kmeans  : k-means centroid selection\n'
                            'kcenter : greedy k-center selection\n'
                            'all     : run all four methods'
                        ))

    parser.add_argument('--eval_epochs',     type=int,   default=500)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--warmup_epochs',   type=int,   default=10)
    parser.add_argument('--dropout',         type=float, default=0.3)
    parser.add_argument('--batch_size',      type=int,   default=64)


    # ── random seed for random selection reproducibility ──────
    parser.add_argument('--random_seed',     type=int,   default=42,
                        help='seed for random selection — change to '
                             'measure variance of random baseline')

    args = parser.parse_args()
    main(args)
