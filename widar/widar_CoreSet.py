"""
Coreset Selection for Widar3.0 BVP 
======================================================

SELECTION SPACES
----------------
  --selection_space feature
      Select using 512-d CNN3D extractor features (default).

  --selection_space raw
      Select using the normalised, reshaped model-input tensors:
      (22, 20, 20) → 8 800 dims, then optional StandardScaler + PCA.

  --selection_space original_csv
      Select using the raw CSV values BEFORE normalisation/reshape:
      342 columns → flattened, then optional StandardScaler + PCA.

  The selected indices always refer to dst_train, so evaluation is
  identical across spaces: train a new CNN on the selected raw samples.

INDEX-ALIGNMENT GUARANTEE
--------------------------
  All three extractors iterate the dataset by sequential integer index
  (range(len(dataset))), which is the only portable way to guarantee that
  row i of the selection matrix == dst_train[i].

  DataLoader with num_workers > 0 does NOT guarantee this, even with
  shuffle=False, because worker prefetch races can re-order batches across
  PyTorch versions and platforms.  num_workers=0 is used throughout.

  Widar_Dataset.data_list is now sorted (the original glob.glob order is
  non-deterministic across filesystems and Python versions).

METHODS:
   random, kmeans, kcenter, DPP,  herding (Greedy mean matching)






Run:
    python widar_CoreSet.py --mode train_extractor
    python widar_CoreSet.py --mode select --ipc 100 --method all
    python widar_CoreSet.py --mode select --ipc 50  --method all --selection_space raw
    python widar_CoreSet.py --mode select --ipc 10  --method all --selection_space original_csv
"""

import os
import glob
import argparse
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from tqdm import tqdm




# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
DATA_ROOT   = 'the address to your dataset'
NUM_CLASSES = 6
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASS_NAMES = ['Push&Pull', 'Sweep', 'Clap', 'Slide',
               'Draw-O(H)', 'Draw-Zigzag(H)']


# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────
class Widar_Dataset(Dataset):
    def __init__(self, root_dir):
        # Sort so that data_list order is deterministic across filesystems
        # and Python versions.  __getitem__(i) == row i of any selection
        # matrix is only guaranteed when the list order is fixed.
        self.data_list = sorted(glob.glob(root_dir + '/*/*.csv'))
        self.folder    = sorted(glob.glob(root_dir + '/*/'))
        self.category  = {self.folder[i].split('/')[-2]: i
                          for i in range(len(self.folder))}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        path = self.data_list[idx]
        y    = self.category[path.split('/')[-2]]
        x    = np.genfromtxt(path, delimiter=',')
        x    = (x - 0.0025) / 0.0119
        x    = x.reshape(22, 20, 20)
        x    = np.clip(x, -3, 3) / 3.0
        return torch.FloatTensor(x), y


# ─────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────
class Widar_CNN3D(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.conv1 = nn.Conv3d(1,   64,  kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(64,  128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool3d(2)
        self.fc1   = nn.Linear(256 * 2 * 2 * 2, 512)
        self.fc2   = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def get_features(self, x):
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return F.relu(self.fc1(x))


class Widar_CNN3D_Eval(nn.Module):
    def __init__(self, num_classes=6, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv3d(1,   64,  kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(64,  128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool3d(2)
        self.drop  = nn.Dropout(dropout)
        self.fc1   = nn.Linear(256 * 2 * 2 * 2, 512)
        self.fc2   = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)



# ─────────────────────────────────────────────────────────────
# TRAIN EXTRACTOR
# ─────────────────────────────────────────────────────────────
def train_extractor(dst_train, dst_test, save_path,
                    epochs=100, batch_size=128):
    train_loader = DataLoader(dst_train, batch_size=batch_size,
                              shuffle=True, num_workers=4)
    test_loader  = DataLoader(dst_test,  batch_size=256,
                              shuffle=False, num_workers=4)
    model = Widar_CNN3D(NUM_CLASSES).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_acc = 0.0
    print(f"\nTraining extractor on {len(dst_train)} samples...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        sched.step()
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    correct += (model(x).argmax(1) == y).sum().item()
                    total   += len(y)
            acc = 100.0 * correct / total
            print(f"  Epoch {epoch+1:3d}: {total_loss/len(train_loader):.4f}"
                  f"  val={acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), save_path)
    print(f"Best val: {best_acc:.2f}%  Saved: {save_path}")
    return best_acc


# ─────────────────────────────────────────────────────────────
# SELECTION MATRIX BUILDERS
# ─────────────────────────────────────────────────────────────

def standardize_and_pca(x, standardize=True, pca_dim=128,
                                random_state=42):
    """StandardScaler + PCA helper shared by all raw extractors."""
    if standardize:
        scaler = StandardScaler()
        x = scaler.fit_transform(x).astype(np.float32)

    if pca_dim is None or pca_dim <= 0:
        return x.astype(np.float32)

    effective_dim = min(int(pca_dim), x.shape[0] - 1, x.shape[1])
    if effective_dim < 1:
        return x.astype(np.float32)

    print(f"  Applying PCA: {x.shape[1]} -> {effective_dim} dims")
    pca = PCA(n_components=effective_dim, random_state=random_state)
    return pca.fit_transform(x).astype(np.float32)


def extract_cnn_features(model, dataset, device, batch_size=256):
    """
    Extract 512-d CNN3D features.

    num_workers=0 guarantees that the row order of the returned matrix
    matches dataset[0], dataset[1], ... exactly, which is required for
    selector indices to align with dst_train.
    """
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0)
    model.eval()
    all_feats, all_y = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc='Extracting CNN3D features', ncols=70):
            feats = model.get_features(x.to(device))
            all_feats.append(feats.cpu())
            all_y.append(torch.tensor(y) if not isinstance(y, torch.Tensor)
                         else y)
    all_feats = torch.cat(all_feats).numpy().astype(np.float32)
    all_y     = torch.cat(all_y).numpy()
    print(f"  Feature matrix: {all_feats.shape} | Labels: {all_y.shape}")
    _print_class_stats(all_feats, all_y, name="feature")
    return all_feats, all_y


def extract_raw_samples(dataset, standardize=True, pca_dim=128,
                        random_state=42):
    """
    Raw model-input sample space.

    Iterates dataset by sequential integer index so that row i of the
    returned matrix == dst_train[i].  Each sample goes through the same
    normalisation/reshape/clip used during training:
        CSV (N, 342) -> normalised -> reshaped (22, 20, 20) -> clipped
        -> flattened to 8 800 dims, then optional StandardScaler + PCA.

    num_workers=0 / direct index iteration is intentional — see module
    docstring for the DataLoader ordering caveat.
    """
    all_x, all_y = [], []

    for i in tqdm(range(len(dataset)),
                  desc='Extracting raw model-input samples', ncols=70):
        x, y = dataset[i]                              # (22, 20, 20), int
        all_x.append(x.reshape(-1).float())
        all_y.append(torch.tensor(y))

    raw_x  = torch.stack(all_x).numpy().astype(np.float32)  # (N, 8800)
    labels = torch.stack(all_y).numpy()

    assert len(labels) == len(dataset), (
        f"Row count mismatch: matrix has {len(labels)} rows "
        f"but dataset has {len(dataset)} items."
    )

    raw_x = standardize_and_pca(
        raw_x, standardize=standardize, pca_dim=pca_dim,
        random_state=random_state
    )
    print(f"  Raw matrix: {raw_x.shape} | Labels: {labels.shape}")
    _print_class_stats(raw_x, labels, name="raw")
    return raw_x, labels


def extract_original_csv_samples(dataset, standardize=True, pca_dim=128,
                                  random_state=42):
    """
    Original CSV value space — BEFORE normalisation, reshape or clip.

    Reads dataset.data_list[i] directly (sorted in __init__), so row i
    maps to dst_train[i] without any DataLoader ordering ambiguity.
    The raw CSV has shape (N_rows, 342); it is flattened to one vector
    per sample before optional StandardScaler + PCA.

    Index alignment: Widar_Dataset stores
        self.data_list = sorted(glob.glob(...))
    and __getitem__(i) loads self.data_list[i].  This function iterates
    enumerate(dataset.data_list) in the same sorted order, so the
    guarantee holds.
    """
    if not hasattr(dataset, 'data_list') or not hasattr(dataset, 'category'):
        raise ValueError(
            "extract_original_csv_samples expects a Widar_Dataset-like object."
        )

    all_x, all_y = [], []

    for path in tqdm(dataset.data_list,
                     desc='Extracting original CSV samples', ncols=70):
        y = dataset.category[path.split('/')[-2]]
        x = np.genfromtxt(path, delimiter=',').astype(np.float32)  # no norm
        all_x.append(x.reshape(1, -1))
        all_y.append(y)

    raw_x  = np.concatenate(all_x, axis=0)   # (N, 342*n_rows)
    labels = np.asarray(all_y)

    assert len(labels) == len(dataset), (
        f"Row count mismatch: matrix has {len(labels)} rows "
        f"but dataset has {len(dataset)} items."
    )

    raw_x = standardize_and_pca(
        raw_x, standardize=standardize, pca_dim=pca_dim,
        random_state=random_state
    )
    print(f"  Original CSV matrix: {raw_x.shape} | Labels: {labels.shape}")
    _print_class_stats(raw_x, labels, name="original_csv")
    return raw_x, labels


def _print_class_stats(x, labels, name="matrix"):
    print(f"\n  Class stats in {name} space:")
    for c in range(NUM_CLASSES):
        m = labels == c
        if not np.any(m):
            print(f"    Class {c} ({CLASS_NAMES[c]:>14}): n=0")
            continue
        mn  = np.linalg.norm(x[m].mean(0))
        std = x[m].std()
        print(f"    Class {c} ({CLASS_NAMES[c]:>14}): "
              f"n={m.sum():5d}  mean_norm={mn:.3f}  std={std:.3f}")
    print("")


def build_selection_matrix(args, dst_train, extractor=None):
    """
    Dispatcher: returns (features, labels) in whichever space was requested.
    Row i of the returned matrix always corresponds to dst_train[i].
    """
    pca_dim = None if args.raw_pca_dim <= 0 else args.raw_pca_dim

    if args.selection_space == 'feature':
        if extractor is None:
            raise ValueError(
                "selection_space='feature' requires a loaded extractor model."
            )
        return extract_cnn_features(extractor, dst_train, DEVICE)

    if args.selection_space == 'raw':
        return extract_raw_samples(
            dst_train,
            standardize=not args.no_standardize_raw,
            pca_dim=pca_dim,
            random_state=args.seed,
        )

    if args.selection_space == 'original_csv':
        return extract_original_csv_samples(
            dst_train,
            standardize=not args.no_standardize_raw,
            pca_dim=pca_dim,
            random_state=args.seed,
        )

    raise ValueError(f"Unknown selection_space: {args.selection_space!r}")


# ═════════════════════════════════════════════════════════════
# EXISTING SELECTION METHODS (unchanged from original)
# ═════════════════════════════════════════════════════════════

def dpp_select(features, labels, n_per_class, num_classes,
               random_state=42):
    try:
        from dppy.finite_dpps import FiniteDPP
    except ImportError:
        raise ImportError("pip install dppy --break-system-packages")
    np.random.seed(random_state)

    feat_n = features
    selected = []
    for c in range(num_classes):
        mask    = labels == c
        cls_f   = feat_n[mask]
        cls_idx = np.where(mask)[0]
        print(f"  Class {c} ({CLASS_NAMES[c]}): {len(cls_f)} → {n_per_class}")
        if len(cls_f) <= n_per_class:
            selected.extend(cls_idx.tolist()); continue
        pw    = cdist(cls_f, cls_f, 'euclidean')
        sigma = np.median(pw[pw > 0])
        L     = np.exp(-pw**2 / (2 * sigma**2))
        try:
            dpp = FiniteDPP(kernel_type='likelihood', L=L)
            dpp.sample_exact_k_dpp(size=n_per_class)
            chosen = list(dpp.list_of_samples[0])
        except Exception as e:
            print(f"    DPP failed ({e}), using k-center")
            chosen = kcenter_greedy(cls_f, n_per_class)
        selected.extend(cls_idx[chosen].tolist())
    return selected


def kcenter_greedy(features, n_select, seed=42):
    np.random.seed(seed)
    sel   = [np.random.randint(len(features))]
    dists = np.full(len(features), np.inf)
    for _ in range(n_select - 1):
        d     = np.linalg.norm(features - features[sel[-1]], axis=1)
        dists = np.minimum(dists, d)
        dists[sel] = -np.inf
        sel.append(int(np.argmax(dists)))
    return sel


def kcenter_select(features, labels, n_per_class, num_classes):
    feat_n = features
    selected = []
    for c in range(num_classes):
        mask    = labels == c
        cls_f   = feat_n[mask]
        cls_idx = np.where(mask)[0]
        print(f"  Class {c} ({CLASS_NAMES[c]}): {len(cls_f)} → {n_per_class}")
        chosen = (cls_idx.tolist() if len(cls_f) <= n_per_class
                  else cls_idx[kcenter_greedy(cls_f, n_per_class)].tolist())
        selected.extend(chosen)
    return selected


def kmeans_select(features, labels, n_per_class, num_classes,
                  random_state=42):
    feat_n = features
    selected = []
    for c in range(num_classes):
        mask    = labels == c
        cls_f   = feat_n[mask]
        cls_idx = np.where(mask)[0]
        print(f"  Class {c} ({CLASS_NAMES[c]}): {len(cls_f)} → {n_per_class}")
        if len(cls_f) <= n_per_class:
            selected.extend(cls_idx.tolist()); continue
        km = KMeans(n_clusters=n_per_class, random_state=random_state,
                    n_init=10)
        km.fit(cls_f)
        nearest, _ = pairwise_distances_argmin_min(km.cluster_centers_, cls_f)
        nearest = list(dict.fromkeys(nearest.tolist()))
        if len(nearest) < n_per_class:
            used = set(nearest)
            for i in range(len(cls_f)):
                if i not in used: nearest.append(i)
                if len(nearest) == n_per_class: break
        selected.extend(cls_idx[nearest[:n_per_class]].tolist())
    return selected


# ═════════════════════════════════════════════════════════════
# NEW METHOD 1: HERDING
# ═════════════════════════════════════════════════════════════
def herding_select(features, labels, n_per_class, num_classes):
    """Herding: greedy class-mean matching."""
    feat_n   = features
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
            chosen.append(best)
            chosen_set.add(best)
            running_sum += cls_f[best]

        err = np.linalg.norm(class_mean - cls_f[chosen].mean(axis=0))
        print(f"    Mean approx error: {err:.4f}  (0=perfect)")
        selected.extend(cls_idx[chosen].tolist())

    return selected







def random_select(features, labels, n_per_class, num_classes,
                  random_state=42):
    rng      = np.random.default_rng(random_state)
    selected = []
    for c in range(num_classes):
        mask    = labels == c
        cls_idx = np.where(mask)[0]
        print(f"  Class {c} ({CLASS_NAMES[c]}): {len(cls_idx)} → {n_per_class}")
        if len(cls_idx) <= n_per_class:
            selected.extend(cls_idx.tolist())
        else:
            selected.extend(
                rng.choice(cls_idx, size=n_per_class, replace=False).tolist())
    return selected


# ─────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────
def make_scheduler_with_warmup(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = ((epoch - warmup_epochs) /
                    max(1, total_epochs - warmup_epochs))
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate_distilled(selected_idx, dst_train, dst_test,
                       num_classes, device,
                       epochs=500, n_runs=3,
                       label_smoothing=0.3,
                       warmup_epochs=10,
                       dropout=0.3):
    print(f"  Building distilled dataset ({len(selected_idx)} samples)...")
    dist_x = torch.stack([dst_train[i][0] for i in selected_idx])
    dist_y = torch.tensor([dst_train[i][1] for i in selected_idx],
                           dtype=torch.long)

    batch_size  = min(256, len(dist_x))
    dist_loader = DataLoader(
        torch.utils.data.TensorDataset(dist_x, dist_y),
        batch_size=batch_size, shuffle=True, drop_last=False)

    test_x = torch.stack([dst_test[i][0] for i in range(len(dst_test))])
    test_y = torch.tensor([dst_test[i][1] for i in range(len(dst_test))],
                           dtype=torch.long)
    test_loader = DataLoader(
        torch.utils.data.TensorDataset(test_x, test_y),
        batch_size=256, shuffle=False)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    accs = []

    for run in range(n_runs):
        torch.manual_seed(run * 42)
        model = Widar_CNN3D_Eval(num_classes, dropout=dropout).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3,
                                  weight_decay=1e-4)
        actual_epochs = max(epochs, 800)
        sched = make_scheduler_with_warmup(opt, warmup_epochs, actual_epochs)
        best_run_acc = 0.0

        for epoch in range(actual_epochs):
            model.train()
            for xb, yb in dist_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = criterion(model(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            if (epoch + 1) % 50 == 0 or epoch == actual_epochs - 1:
                model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for xb, yb in test_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        correct += (model(xb).argmax(1) == yb).sum().item()
                        total   += len(yb)
                best_run_acc = max(best_run_acc, 100.0 * correct / total)

        accs.append(best_run_acc)
        print(f"    Run {run+1}/{n_runs}: {best_run_acc:.2f}%")

    return float(np.mean(accs)), float(np.std(accs))





# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main(args):
    print(f"Device          : {DEVICE}")
    print(f"Selection space : {args.selection_space}")

    dst_train = Widar_Dataset(DATA_ROOT + 'train/')
    dst_test  = Widar_Dataset(DATA_ROOT + 'test/')
    print(f"Train: {len(dst_train)}  |  Test: {len(dst_test)}")

    if args.mode == 'train_extractor':
        train_extractor(dst_train, dst_test,
                        save_path=args.extractor_path,
                        epochs=args.extractor_epochs)
        return

    # ── Load extractor (needed for 'feature' space, and for diagnose) ────
    extractor = None
    if args.selection_space == 'feature' or args.mode == 'diagnose':
        if not os.path.exists(args.extractor_path):
            raise FileNotFoundError(
                f"Extractor not found: {args.extractor_path}\n"
                f"Run: python {__file__} --mode train_extractor\n"
                f"Or use --selection_space raw / original_csv.")
        extractor = Widar_CNN3D(NUM_CLASSES).to(DEVICE)
        extractor.load_state_dict(
            torch.load(args.extractor_path, map_location=DEVICE,
                       weights_only=False))
        extractor.eval()

    # ── NEW: diagnose mode — builds BOTH spaces, runs NC1 comparison ─────
    if args.mode == 'diagnose':
        print("\nBuilding feature-space matrix...")
        feature_matrix, labels_f = extract_cnn_features(extractor, dst_train, DEVICE)

        print("\nBuilding raw-space matrix (PCA-matched to feature dim)...")
        raw_matrix, labels_r = extract_raw_samples(
            dst_train, standardize=True,
            pca_dim=feature_matrix.shape[1],   # <-- match dims, see note below
            random_state=args.seed)

        assert np.array_equal(labels_f, labels_r), "label misalignment!"

        return

    print(f"\nBuilding selection matrix ({args.selection_space} space)...")
    features, labels = build_selection_matrix(args, dst_train, extractor)

    print(f"\nIPC: {args.ipc}  ({args.ipc * NUM_CLASSES} total)")
    results = {}

    def run_eval(name, idx):
        print(f"\n{'='*60}\n{name.upper()}  [{args.selection_space}]\n{'='*60}")
        save_name = f'distilled_{args.selection_space}_{name}_ipc{args.ipc}.pt'
        torch.save({
            'indices': idx, 'method': name, 'ipc': args.ipc,
            'selection_space': args.selection_space,
            'raw_pca_dim': args.raw_pca_dim,
        }, save_name)
        acc, std = evaluate_distilled(
            idx, dst_train, dst_test, NUM_CLASSES, DEVICE,
            epochs=args.eval_epochs,
            label_smoothing=args.label_smoothing,
            warmup_epochs=args.warmup_epochs,
            dropout=args.dropout)
        results[name] = (acc, std)
        print(f"{name} → {acc:.2f}% ± {std:.2f}%")

        
    if args.method in ('kmeans', 'all'):
        run_eval('kmeans', kmeans_select(features, labels,
                                         args.ipc, NUM_CLASSES))

    if args.method in ('kcenter', 'all'):
        run_eval('kcenter', kcenter_select(features, labels,
                                            args.ipc, NUM_CLASSES))

    if args.method in ('DPP', 'all'):
        run_eval('DPP', dpp_select(features, labels,
                                    args.ipc, NUM_CLASSES))

    if args.method in ('random', 'all'):
        run_eval('random', random_select(features, labels,
                                          args.ipc, NUM_CLASSES))

    if args.method in ('herding', 'all'):
        run_eval('herding', herding_select(features, labels,
                                            args.ipc, NUM_CLASSES))



    # Summary
    method_type = {
        'random':                  'lower bound',
        'kmeans':                  'prototypes',
        'kcenter':                 'max coverage',
        'DPP':                     'diverse set',
        'herding':                 'mean matching',
    }
    print("\n" + "=" * 65)
    print(f"SUMMARY  selection_space={args.selection_space}  "
          f"IPC={args.ipc}  ({args.ipc * NUM_CLASSES}/{len(dst_train)})")
    print("=" * 65)
    print(f"  {'Method':<26}  {'Type':<24}  {'Acc':>8}  {'Std':>6}")
    print("  " + "-" * 60)
    for m, (a, s) in sorted(results.items(), key=lambda x: -x[1][0]):
        t = method_type.get(m, '')
        print(f"  {m:<26}  {t:<24}  {a:>7.2f}%  {s:>5.2f}%")
    print("=" * 65)
    if 'random' in results:
        print("\nGain over random:")
        for m, (a, _) in sorted(results.items(), key=lambda x: -x[1][0]):
            if m != 'random':
                print(f"  {m:<26} {a - results['random'][0]:+.2f}%")


# ─────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='select',
                        choices=['train_extractor', 'select','diagnose'])
    parser.add_argument('--extractor_path',
                        default='***/herding_new_may2026/extractor/cnn3d_extractor_updatedFeatures.pt')
    parser.add_argument('--extractor_epochs', type=int, default=100)

    # ── Selection space ───────────────────────────────────────────────────
    parser.add_argument(
        '--selection_space', default='feature',
        choices=['feature', 'raw', 'original_csv'],
        help=(
            "Space in which to score/rank samples for selection.\n"
            "  feature      : 512-d CNN3D extractor embeddings (default).\n"
            "  raw          : normalised/reshaped model-input tensors\n"
            "                 (22, 20, 20) = 8 800 dims, then PCA.\n"
            "  original_csv : raw CSV values before any normalisation,\n"
            "                 flattened, then PCA.\n"
            "The selected indices always refer to dst_train, so evaluation\n"
            "is identical: a new CNN is trained on the original raw samples."
        )
    )
    parser.add_argument(
        '--raw_pca_dim', type=int, default=0,
        help="PCA dimension for raw/original_csv spaces. 0 = no PCA."
    )
    parser.add_argument(
        '--no_standardize_raw', action='store_true',
        help="Disable z-scoring before raw/original_csv selection."
    )
    parser.add_argument('--seed', type=int, default=42)

    # ── Core experiment settings ──────────────────────────────────────────
    parser.add_argument('--ipc',    type=int,   default=100)
    parser.add_argument('--method', default='all',
                        choices=['random', 'kmeans', 'kcenter', 'DPP',
                                 'herding', 'all'
                                 ])

    parser.add_argument('--eval_epochs',     type=int,   default=800)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--warmup_epochs',   type=int,   default=10)
    parser.add_argument('--dropout',         type=float, default=0.2)



    args = parser.parse_args()
    main(args)
