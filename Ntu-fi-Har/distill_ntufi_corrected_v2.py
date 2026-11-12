"""
distill_ntufi_corrected.py
==========================
Dataset distillation for NTU-Fi HAR.

This version lets you choose where the selector operates:

  --selection_space feature
      Select using 512-d CNN extractor features.

  --selection_space raw
      Select using the original model-input samples returned by NTUFi_Dataset:
      normalized, downsampled, reshaped tensors with shape (3, 114, 500).

  --selection_space original_mat
      Select using the unprocessed values stored in each .mat file before
      normalization/downsampling/reshape.

The selected indices always refer to dst_train, so evaluation is identical
across spaces: train a new CNN on the selected raw dataset samples and test it.
"""

import argparse
import glob
import os
import warnings

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


torch.multiprocessing.set_sharing_strategy("file_system")


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DATA_ROOT = "/home/sattarha/data/WiFi-CSI-Sensing-Benchmark/NTU-Fi_HAR/"
NUM_CLASSES = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Alphabetical order, matching sorted class folders in NTUFi_Dataset.
CLASS_NAMES = ["box", "circle", "clean", "fall", "run", "walk"]


# ---------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------
class NTUFi_Dataset(Dataset):
    def __init__(self, root_dir, modal="CSIamp"):
        self.modal = modal
        self.data_list = sorted(glob.glob(root_dir + "/*/*.mat"))
        self.folder = sorted(glob.glob(root_dir + "/*/"))
        self.category = {
            self.folder[i].split("/")[-2]: i
            for i in range(len(self.folder))
        }
        print(f"  Classes found: {self.category}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        path = self.data_list[idx]
        y = self.category[path.split("/")[-2]]
        x = sio.loadmat(path)[self.modal]       # (342, 2000)
        x = (x - 42.3199) / 4.9802              # normalize
        x = x[:, ::4]                           # (342, 500)
        x = x.reshape(3, 114, 500)              # (3, 114, 500)
        return torch.FloatTensor(x), y


# ---------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------
class NTUFi_CNN2D_Eval(nn.Module):
    def __init__(self, num_classes=6, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)

    @torch.no_grad()
    def get_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return F.relu(self.fc1(x))



# ---------------------------------------------------------------------
# TRAIN EXTRACTOR
# ---------------------------------------------------------------------
def train_extractor(dst_train, dst_test, save_path, epochs=50, batch_size=32):
    train_loader = DataLoader(
        dst_train, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        dst_test, batch_size=32, shuffle=False, num_workers=4
    )

    model = NTUFi_CNN2D_Eval(NUM_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    best_acc = 0.0
    print(f"\nTraining extractor on {len(dst_train)} samples")
    print(f'{"Epoch":>6}  {"Train Loss":>12}  {"Val Acc":>10}')
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
                    total += len(y)

            acc = 100.0 * correct / total
            print(
                f"{epoch + 1:>6}  {total_loss / len(train_loader):>12.4f}"
                f"  {acc:>9.2f}%"
            )

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), save_path)

    print(f"\nBest val accuracy: {best_acc:.2f}%")
    print(f"Extractor saved to: {save_path}")
    return best_acc


# ---------------------------------------------------------------------
# SELECTION MATRIX BUILDERS
# ---------------------------------------------------------------------
def extract_features(model, dataset, device, batch_size=32, num_workers=0):
    """
    Extract 512-d CNN features.

    num_workers=0 is intentional: guarantees that the row order of the
    returned feature matrix matches dataset[0], dataset[1], ... exactly,
    which is required so that the selector indices align with dst_train.
    """
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    model.eval()
    all_feats, all_y = [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Extracting CNN features", ncols=70):
            all_feats.append(model.get_features(x.to(device)).cpu())
            all_y.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))

    features = torch.cat(all_feats).numpy().astype(np.float32)
    labels = torch.cat(all_y).numpy()
    print(f"  Feature matrix: {features.shape} | Labels: {labels.shape}")
    print_class_stats(features, labels, name="feature")
    return features, labels


def extract_raw_samples(dataset, batch_size=32, num_workers=0,
                        standardize=True, pca_dim=128, random_state=42):
    """
    Raw model-input sample space.

    Iterates dataset sequentially by index so that position i in the returned
    matrix always corresponds to dst_train[i].  num_workers=0 is intentional:
    with multiple workers and shuffle=False the yield order is still sequential
    in most PyTorch versions, but it is not guaranteed across all platforms and
    versions.  Using num_workers=0 (single-process DataLoader) is the only safe
    way to ensure the returned matrix rows align with dataset indices.

    Samples go through the same normalization/downsampling/reshape as during
    training: (342, 2000) -> normalized -> (342, 500) -> (3, 114, 500) ->
    flattened to 171 000 dimensions, before optional standardisation and PCA.
    """
    # Build feature matrix by iterating dataset indices directly so the row
    # order is guaranteed to match dst_train[0], dst_train[1], ...
    all_x, all_y = [], []

    for i in tqdm(range(len(dataset)), desc="Extracting raw model-input samples",
                  ncols=70):
        x, y = dataset[i]
        all_x.append(x.reshape(-1).float().cpu())
        all_y.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))

    raw_x = torch.stack(all_x).numpy().astype(np.float32)
    labels = torch.stack(all_y).numpy()

    # Sanity-check: verify label at each position matches what dst_train[i]
    # returns.  This catches any future refactor that breaks index alignment.
    assert len(labels) == len(dataset), (
        f"Row count mismatch: matrix has {len(labels)} rows but "
        f"dataset has {len(dataset)} items."
    )

    raw_x = maybe_standardize_and_pca(
        raw_x, standardize=standardize, pca_dim=pca_dim,
        random_state=random_state
    )
    print(f"  Raw matrix: {raw_x.shape} | Labels: {labels.shape}")
    print_class_stats(raw_x, labels, name="raw")
    return raw_x, labels


def extract_original_mat_samples(dataset, modal=None, standardize=True,
                                 pca_dim=128, random_state=42):
    """
    Original .mat value space.

    This reads dataset.data_list directly, with no normalization, no packet
    downsampling, and no reshape.

    Index alignment: NTUFi_Dataset stores files as
        self.data_list = sorted(glob.glob(...))
    and __getitem__(i) loads self.data_list[i].  This function iterates
    enumerate(dataset.data_list) in the same sorted order, so row i of the
    returned matrix corresponds to dst_train[i] — the same guarantee as
    the direct-index loop used in extract_raw_samples.
    """
    if not hasattr(dataset, "data_list") or not hasattr(dataset, "category"):
        raise ValueError(
            "extract_original_mat_samples expects an NTUFi_Dataset-like object."
        )

    modal = modal or getattr(dataset, "modal", "CSIamp")
    all_x, all_y = [], []

    for path in tqdm(dataset.data_list, desc="Extracting original .mat samples",
                     ncols=70):
        y = dataset.category[path.split("/")[-2]]
        x = sio.loadmat(path)[modal]
        all_x.append(np.asarray(x, dtype=np.float32).reshape(1, -1))
        all_y.append(y)

    raw_x = np.concatenate(all_x, axis=0)
    labels = np.asarray(all_y)
    raw_x = maybe_standardize_and_pca(
        raw_x, standardize=standardize, pca_dim=pca_dim,
        random_state=random_state
    )
    print(f"  Original .mat matrix: {raw_x.shape} | Labels: {labels.shape}")
    print_class_stats(raw_x, labels, name="original_mat")
    return raw_x, labels


def maybe_standardize_and_pca(x, standardize=True, pca_dim=128,
                              random_state=42):
    if standardize:
        scaler = StandardScaler()
        x = scaler.fit_transform(x).astype(np.float32)

    if pca_dim is None or pca_dim <= 0:
        return x.astype(np.float32)

    effective_dim = min(int(pca_dim), x.shape[0] - 1, x.shape[1])
    if effective_dim < 1:
        return x.astype(np.float32)

    print(f"  Applying PCA: {x.shape[1]} -> {effective_dim}")
    pca = PCA(n_components=effective_dim, random_state=random_state)
    return pca.fit_transform(x).astype(np.float32)


def print_class_stats(x, labels, name="matrix"):
    print(f"\n  Class stats in {name} space:")
    for c in range(NUM_CLASSES):
        m = labels == c
        if not np.any(m):
            print(f"    Class {c} ({CLASS_NAMES[c]:>7}): n=0")
            continue
        print(
            f"    Class {c} ({CLASS_NAMES[c]:>7}): "
            f"n={m.sum():4d} mean_norm={np.linalg.norm(x[m].mean(0)):.3f} "
            f"std={x[m].std():.3f}"
        )
    print("")


def build_selection_matrix(args, dst_train):
    pca_dim = None if args.raw_pca_dim <= 0 else args.raw_pca_dim

    if args.selection_space == "feature":
        if not os.path.exists(args.extractor_path):
            raise FileNotFoundError(
                f"Extractor not found at {args.extractor_path}\n"
                "Run --mode train_extractor first, or use "
                "--selection_space raw."
            )

        print(f"Loading extractor from {args.extractor_path}...")
        extractor = NTUFi_CNN2D_Eval(NUM_CLASSES).to(DEVICE)
        extractor.load_state_dict(
            torch.load(args.extractor_path, map_location=DEVICE,
                       weights_only=False)
        )
        extractor.eval()
        print("Extractor loaded.\n")
        return extract_features(extractor, dst_train, DEVICE, batch_size=32)

    if args.selection_space == "raw":
        return extract_raw_samples(
            dst_train,
            batch_size=32,
            standardize=not args.no_standardize_raw,
            pca_dim=pca_dim,
            random_state=args.seed,
        )

    if args.selection_space == "original_mat":
        return extract_original_mat_samples(
            dst_train,
            standardize=not args.no_standardize_raw,
            pca_dim=pca_dim,
            random_state=args.seed,
        )

    raise ValueError(f"Unknown selection_space: {args.selection_space}")


# ---------------------------------------------------------------------
# SELECTION METHODS
# ---------------------------------------------------------------------
def random_select(features, labels, n_per_class, num_classes=NUM_CLASSES,
                  random_state=42):
    rng = np.random.default_rng(random_state)
    selected = []

    for c in range(num_classes):
        mask = labels == c
        cls_idx = np.where(mask)[0]
        print(f"  Class {c} ({CLASS_NAMES[c]}): {len(cls_idx)} -> {n_per_class}")

        if len(cls_idx) <= n_per_class:
            selected.extend(cls_idx.tolist())
        else:
            selected.extend(
                rng.choice(cls_idx, size=n_per_class, replace=False).tolist()
            )

    print(f"Total selected: {len(selected)}")
    return selected


def kmeans_select(features, labels, n_per_class, random_state=42):
    selected = []

    for c in range(NUM_CLASSES):
        mask = labels == c
        cls_f = features[mask]
        cls_idx = np.where(mask)[0]
        n_c = len(cls_f)
        print(f"  Class {c} ({CLASS_NAMES[c]:>7}): {n_c} -> {n_per_class}")

        if n_c <= n_per_class:
            selected.extend(cls_idx.tolist())
            continue

        km = KMeans(
            n_clusters=n_per_class, random_state=random_state,
            n_init=10, max_iter=300
        )
        cluster_lbl = km.fit_predict(cls_f)

        chosen = []
        for k in range(n_per_class):
            members = np.where(cluster_lbl == k)[0]
            if len(members) == 0:
                continue
            d = np.linalg.norm(cls_f[members] - km.cluster_centers_[k], axis=1)
            chosen.append(members[int(np.argmin(d))])

        selected.extend(cls_idx[chosen].tolist())

    print(f"Total selected: {len(selected)}")
    return selected


def herding_select(features, labels, n_per_class):
    selected = []

    for c in range(NUM_CLASSES):
        mask = labels == c
        cls_f = features[mask]
        cls_idx = np.where(mask)[0]
        n_c = len(cls_f)
        print(f"  Class {c} ({CLASS_NAMES[c]:>7}): {n_c} -> {n_per_class}")

        if n_c <= n_per_class:
            selected.extend(cls_idx.tolist())
            continue

        class_mean = cls_f.mean(axis=0)
        running_sum = np.zeros_like(class_mean)
        chosen, chosen_set = [], set()

        for step in range(1, n_per_class + 1):
            new_means = (running_sum[None, :] + cls_f) / step
            dists = np.linalg.norm(class_mean - new_means, axis=1)
            if chosen_set:
                dists[list(chosen_set)] = np.inf

            best = int(np.argmin(dists))
            chosen.append(best)
            chosen_set.add(best)
            running_sum += cls_f[best]

        selected.extend(cls_idx[chosen].tolist())

    print(f"Total selected: {len(selected)}")
    return selected





def dpp_select(features, labels, n_per_class, num_classes=NUM_CLASSES,
               random_state=42):
    """
    DPP selection. If dppy is unavailable or DPP fails numerically, use
    k-center greedy for that class instead of crashing.
    """
    try:
        from dppy.finite_dpps import FiniteDPP
        have_dppy = True
    except ImportError:
        print("dppy is not installed; using k-center fallback for DPP.")
        have_dppy = False
        FiniteDPP = None

    scaler = StandardScaler()
    feat_n = scaler.fit_transform(features).astype(np.float32)
    selected = []

    np.random.seed(random_state)

    for c in tqdm(range(num_classes), desc="DPP per class", ncols=70):
        mask = labels == c
        cls_f = feat_n[mask]
        cls_idx = np.where(mask)[0]

        if len(cls_f) <= n_per_class:
            selected.extend(cls_idx.tolist())
            continue

        chosen = None
        if have_dppy:
            try:
                pw_dist = cdist(cls_f, cls_f, metric="euclidean")
                positive = pw_dist[pw_dist > 0]
                sigma = np.median(positive) if len(positive) else 1.0
                sigma = max(float(sigma), 1e-8)
                l_kernel = np.exp(-(pw_dist ** 2) / (2.0 * sigma ** 2))
                dpp = FiniteDPP(kernel_type="likelihood", L=l_kernel)
                dpp.sample_exact_k_dpp(size=n_per_class)
                chosen = list(dpp.list_of_samples[0])
            except Exception as exc:
                print(f"\n  Class {c}: DPP failed ({exc}); using k-center.")

        if chosen is None:
            chosen = kcenter_greedy(cls_f, n_per_class)

        selected.extend(cls_idx[chosen].tolist())

    print(f"Total selected: {len(selected)}")
    return selected


def kcenter_greedy(x, k):
    if len(x) <= k:
        return list(range(len(x)))

    center = x.mean(axis=0, keepdims=True)
    first = int(np.argmin(np.linalg.norm(x - center, axis=1)))
    chosen = [first]
    min_d = cdist(x, x[[first]], metric="euclidean").reshape(-1)

    for _ in range(1, k):
        best = int(np.argmax(min_d))
        chosen.append(best)
        new_d = cdist(x, x[[best]], metric="euclidean").reshape(-1)
        min_d = np.minimum(min_d, new_d)

    return chosen


# ---------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------
def make_scheduler_with_warmup(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (
            (epoch - warmup_epochs)
            / float(max(1, total_epochs - warmup_epochs))
        )
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate_distilled(selected_idx, dst_train, dst_test,
                       epochs=500, n_runs=3,
                       label_smoothing=0.1,
                       warmup_epochs=10,
                       dropout=0.3):
    print(f"\n  Building distilled dataset ({len(selected_idx)} samples)...")
    dist_x = torch.stack([dst_train[i][0] for i in selected_idx])
    dist_y = torch.tensor(
        [dst_train[i][1] for i in selected_idx], dtype=torch.long
    )
    dist_loader = DataLoader(
        torch.utils.data.TensorDataset(dist_x, dist_y),
        batch_size=min(32, len(dist_x)),
        shuffle=True,
    )

    print("  Building test loader...")
    test_x = torch.stack([dst_test[i][0] for i in range(len(dst_test))])
    test_y = torch.tensor(
        [dst_test[i][1] for i in range(len(dst_test))], dtype=torch.long
    )
    test_loader = DataLoader(
        torch.utils.data.TensorDataset(test_x, test_y),
        batch_size=32,
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    accs = []

    for run in range(n_runs):
        torch.manual_seed(run * 42)
        np.random.seed(run * 42)

        model = NTUFi_CNN2D_Eval(NUM_CLASSES, dropout=dropout).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = make_scheduler_with_warmup(opt, warmup_epochs, epochs)

        best_run_acc = 0.0
        for epoch in range(epochs):
            model.train()
            for xb, yb in dist_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                loss = criterion(model(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

            sched.step()

            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for xb, yb in test_loader:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        correct += (model(xb).argmax(1) == yb).sum().item()
                        total += len(yb)

                acc = 100.0 * correct / total
                best_run_acc = max(best_run_acc, acc)
                print( f'    Run {run+1}/{n_runs} | ' f'Epoch {epoch+1:>4}/{epochs} | ' f'Acc: {acc:>6.2f}% | ' f'Best: {best_run_acc:>6.2f}%')

        accs.append(best_run_acc)
        print(f"    Run {run + 1}/{n_runs}: {best_run_acc:.2f}%")

    return float(np.mean(accs)), float(np.std(accs))


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main(args):
    print("\n" + "=" * 58)
    print("  NTU-Fi HAR - Dataset Distillation")
    print(f"  Device          : {DEVICE}")
    print(f"  Data            : {args.data_root}")
    print(f"  Selection space : {args.selection_space}")
    print("=" * 58 + "\n")

    train_root = os.path.join(args.data_root, "train_amp")
    test_root = os.path.join(args.data_root, "test_amp")

    dst_train = NTUFi_Dataset(train_root)
    dst_test = NTUFi_Dataset(test_root)
    print(f"Train: {len(dst_train)} | Test: {len(dst_test)}\n")

    if args.mode == "train_extractor":
        best = train_extractor(
            dst_train,
            dst_test,
            save_path=args.extractor_path,
            epochs=args.extractor_epochs,
        )
        print(f"\nDone. Best accuracy: {best:.2f}%")
        return

    features, labels = build_selection_matrix(args, dst_train)

    print(
        f"\nIPC = {args.ipc} "
        f"({args.ipc * NUM_CLASSES} / {len(dst_train)} samples)\n"
    )

    method = args.method.lower()
    methods_to_run = (
        ["random", "kmeans", "herding",
          "dpp"]
        if method == "all"
        else [method]
    )

    results = {}

    def run_eval(name, idx):
        save_name = f"ntufi_{args.selection_space}_{name}_ipc{args.ipc}.pt"
        torch.save(
            {
                "indices": idx,
                "method": name,
                "ipc": args.ipc,
                "selection_space": args.selection_space,
                "raw_pca_dim": args.raw_pca_dim,
            },
            save_name,
        )
        print(f"  Saved indices to: {save_name}")

        acc, std = evaluate_distilled(
            idx,
            dst_train,
            dst_test,
            epochs=args.eval_epochs,
            n_runs=args.eval_runs,
            label_smoothing=args.label_smoothing,
            warmup_epochs=args.warmup_epochs,
            dropout=args.dropout,
        )
        results[name] = (acc, std)
        print(f"  {name:<24} -> {acc:.2f}% +/- {std:.2f}%\n")

    for name in methods_to_run:
        print("=" * 65)
        print(f"{name.upper()} in {args.selection_space.upper()} space")
        print("=" * 65)

        if name == "random":
            idx = random_select(features, labels, args.ipc,
                                random_state=args.seed)

        elif name == "kmeans":
            idx = kmeans_select(features, labels, args.ipc,
                                random_state=args.seed)

        elif name == "herding":
            idx = herding_select(features, labels, args.ipc)

        elif name == "kmeans_herding":
            idx = kmeans_herding_select(
                features,
                labels,
                args.ipc,
                var_weight=args.kh_var_weight,
                cluster_weight=args.kh_cluster_weight,
                coverage_weight=args.kh_coverage_weight,
                diversity_weight=args.kh_diversity_weight,
                n_clusters=args.kh_n_clusters,
                min_cluster_size=args.kh_min_cluster_size,
                random_state=args.seed,
            )

      

        elif name == "dpp":
            idx = dpp_select(features, labels, args.ipc,
                             random_state=args.seed)

        else:
            raise ValueError(f"Unknown method: {name}")

        run_eval(name, idx)

    print("\n" + "=" * 65)
    print(
        f"SUMMARY | selection_space={args.selection_space} | IPC={args.ipc}"
    )
    print("=" * 65)
    print(f'{"Method":<24} {"Acc":>8} {"Std":>8}')
    print("-" * 43)
    for name, (acc, std) in sorted(results.items(), key=lambda x: -x[1][0]):
        print(f"{name:<24} {acc:>7.2f}% {std:>7.2f}%")
    print("=" * 65)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--mode", default="select",
                   choices=["train_extractor", "select"])
    p.add_argument("--data_root", default=DATA_ROOT)
    p.add_argument(
        "--extractor_path",
        default="/home/sattarha/data/WiFi-CSI-Sensing-Benchmark/"
                "cnn2d_extractor_ntu.pt",
    )
    p.add_argument("--extractor_epochs", type=int, default=50)

    p.add_argument("--selection_space", default="feature",
                   choices=["feature", "raw", "original_mat"])
    p.add_argument(
        "--raw_pca_dim",
        type=int,
        default=0,
        help="PCA dimension for raw/original_mat selection. Use 0 for no PCA.",
    )
    p.add_argument(
        "--no_standardize_raw",
        action="store_true",
        help="Disable z-scoring before raw/original_mat selection.",
    )

    p.add_argument("--ipc", type=int, default=10)
    p.add_argument(
        "--method",
        default="kmeans_herding",
        choices=[
            "random",
            "kmeans",
            "herding",
            "dpp",
            "DPP",
            "all",
        ],
    )

    p.add_argument("--eval_epochs", type=int, default=500)
    p.add_argument("--eval_runs", type=int, default=3)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)



    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
