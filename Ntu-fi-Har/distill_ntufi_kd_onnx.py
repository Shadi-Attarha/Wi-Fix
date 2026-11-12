"""
distill_ntufi_kd.py
====================
NTU-Fi HAR dataset distillation — SAME PROCEDURE as widar_distill.py:

  STEP A  Train a full-data feature extractor (NTUFi_CNN2D)          — CE only
  STEP B  Extract embeddings from that extractor, run coreset selection
          (kmeans / herding / DPP / random / )
  PHASE 1 Train a KD TEACHER (NTUFi_CNN2D_Teacher) FROM SCRATCH on the
          selected coreset only (never sees the full dataset)
  PHASE 2 Distill a smaller STUDENT (NTUFi_CNN2D_Student) from that
          coreset-trained teacher, on the same coreset, using DKD +
          attention transfer + a feature-hint loss

Everything is evaluated against a test set that is cached ONCE into
tensors (build_test_tensors) and reused everywhere — no repeated CSV/MAT
re-reads. Both teacher and student track a full accuracy curve during
training and keep the BEST epoch's weights, not just the final epoch.

Run
---
    # Step A — train the full-data feature extractor
    python distill_ntufi_kd.py --mode train_extractor \
        --extractor_path ntufi_extractor.pt --extractor_epochs 70

    # Step B — select coreset, train coreset-teacher, distill student
    python distill_ntufi_kd.py --mode select \
        --ipc 10 --method herding --extractor_path ntufi_extractor.pt

    # Compare all selection methods
    python distill_ntufi_kd.py --mode select \
        --ipc 10 --method all --extractor_path ntufi_extractor.pt

    # Baselines
    python distill_ntufi_kd.py --mode select --baseline full_data
    python distill_ntufi_kd.py --mode select --ipc 10 --baseline random_ce
    python distill_ntufi_kd.py --mode select --ipc 10 --baseline random_kd
    python distill_ntufi_kd.py --mode select --ipc 10 --baseline coreset_ce --method herding
    python distill_ntufi_kd.py --mode select --ipc 10 --baseline teacher_only --method herding
"""

import os
import glob
import argparse
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from tqdm import tqdm
import torch.multiprocessing
from kd_diagnostics import confusion_matrix_report, compare_models_per_class, per_run_accuracy_check

torch.multiprocessing.set_sharing_strategy('file_system')

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
DATA_ROOT   = '/home/sattarha/data/WiFi-CSI-Sensing-Benchmark/NTU-Fi_HAR/'
NUM_CLASSES = 6
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

# Alphabetical order — matches glob folder sorting in NTUFi_Dataset
CLASS_NAMES = ['box', 'circle', 'clean', 'fall', 'run', 'walk']

TEMPERATURE  = 4.0
ALPHA        = 1.0
BETA         = 2.0
LAMBDA_FEAT  = 0.1
LAMBDA_AT    = 0.5


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


def dataset_labels(dataset):
    """Labels only, no disk reads of the .mat payloads — used for
    stratified splitting before we touch any tensors."""
    labels = []
    for path in dataset.data_list:
        labels.append(dataset.category[path.split('/')[-2]])
    return np.array(labels, dtype=np.int64)


def stratified_split_indices(indices, labels, val_fraction=0.1, seed=0):
    rng = np.random.default_rng(seed)
    indices = np.array(indices, dtype=np.int64)
    labels = np.asarray(labels)
    train_idx, val_idx = [], []

    for c in np.unique(labels[indices]):
        cls_idx = indices[labels[indices] == c].copy()
        rng.shuffle(cls_idx)

        if len(cls_idx) <= 1:
            train_idx.extend(cls_idx.tolist())
            continue

        n_val = max(1, int(round(len(cls_idx) * val_fraction)))
        n_val = min(n_val, len(cls_idx) - 1)
        val_idx.extend(cls_idx[:n_val].tolist())
        train_idx.extend(cls_idx[n_val:].tolist())

    return train_idx, val_idx


# ─────────────────────────────────────────────────────────────
# Build the test set ONCE, reuse everywhere as (test_x, test_y) tensors.
# All evaluation AND all diagnostics (confusion matrices, per-class
# comparisons) go through these cached tensors — never dst_test directly
# — so nothing re-reads .mat files from disk after this point.
# ─────────────────────────────────────────────────────────────
def build_test_tensors(dst_test):
    print(f"  Caching {len(dst_test)} test samples into memory (one-time)...")
    test_x = torch.stack([dst_test[i][0]
                          for i in tqdm(range(len(dst_test)), desc='Caching test set')])
    test_y = torch.tensor([dst_test[i][1] for i in range(len(dst_test))],
                          dtype=torch.long)
    return test_x, test_y


def _eval_acc(model, test_x, test_y, device, batch_size=32):
    """Evaluates on already-cached test tensors — no disk I/O here."""
    loader = DataLoader(
        torch.utils.data.TensorDataset(test_x, test_y),
        batch_size=batch_size, shuffle=False
    )
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb  = xb.to(device), yb.to(device)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total   += len(yb)
    return 100.0 * correct / total


def print_accuracy_curve(history, label, top_n=10):
    """
    history: list of (epoch, acc) tuples, in order.
    Prints the full curve compactly and highlights the best epoch.
    """
    if not history:
        return
    best_epoch, best_acc = max(history, key=lambda h: h[1])
    print(f"\n  {label} — accuracy curve ({len(history)} checkpoints):")
    for epoch, acc in history:
        marker = "  <-- best" if epoch == best_epoch else ""
        print(f"    epoch {epoch:>4}  {acc:>6.2f}%{marker}")
    print(f"  {label} best: epoch {best_epoch} @ {best_acc:.2f}%")


# ─────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────

class NTUFi_CNN2D(nn.Module):
    """Full-data FEATURE EXTRACTOR ONLY — embeddings for coreset selection.
    Input: (B, 3, 114, 500) — 3 antenna pairs as channels.
    get_features() returns 512-d, L2-normalized penultimate activations."""
    def __init__(self, num_classes=6):
        super().__init__()
        self.conv1         = nn.Conv2d(3,   64,  kernel_size=3, padding=1)
        self.conv2         = nn.Conv2d(64,  128, kernel_size=3, padding=1)
        self.conv3         = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool          = nn.MaxPool2d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1           = nn.Linear(256 * 4 * 4, 512)
        self.fc2           = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    @torch.no_grad()
    def get_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        feat = F.relu(self.fc1(x))
        return F.normalize(feat, p=2, dim=1)


class NTUFi_CNN2D_Teacher(nn.Module):
    """KD TEACHER — trained ONLY on the selected coreset, not the full dataset."""
    def __init__(self, num_classes=6, dropout=0.3):
        super().__init__()
        self.conv1         = nn.Conv2d(3,   64,  kernel_size=3, padding=1)
        self.conv2         = nn.Conv2d(64,  128, kernel_size=3, padding=1)
        self.conv3         = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool          = nn.MaxPool2d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.drop          = nn.Dropout(dropout)
        self.fc1           = nn.Linear(256 * 4 * 4, 512)
        self.fc2           = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)

    def get_attention_maps(self, x):
        a1 = F.relu(self.conv1(x));  x = self.pool(a1)
        a2 = F.relu(self.conv2(x));  x = self.pool(a2)
        a3 = F.relu(self.conv3(x))
        pooled = self.adaptive_pool(a3)
        feat = F.relu(self.fc1(pooled.view(pooled.size(0), -1)))
        return a1, a2, a3, feat


class NTUFi_CNN2D_Student(nn.Module):
    """Student model — reduced channel widths + narrower fc1, same
    ratio as the Widar student relative to its teacher."""
    def __init__(self, num_classes=6):
        super().__init__()
        self.conv1         = nn.Conv2d(3,  48,  kernel_size=3, padding=1)
        self.conv2         = nn.Conv2d(48, 96,  kernel_size=3, padding=1)
        self.conv3         = nn.Conv2d(96, 192, kernel_size=3, padding=1)
        self.pool          = nn.MaxPool2d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1           = nn.Linear(192 * 4 * 4, 384)
        self.fc2           = nn.Linear(384, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def get_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return F.relu(self.fc1(x))

    def get_attention_maps(self, x):
        a1 = F.relu(self.conv1(x));  x = self.pool(a1)
        a2 = F.relu(self.conv2(x));  x = self.pool(a2)
        a3 = F.relu(self.conv3(x))
        pooled = self.adaptive_pool(a3)
        feat = F.relu(self.fc1(pooled.view(pooled.size(0), -1)))
        return a1, a2, a3, feat


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ─────────────────────────────────────────────────────────────
# KD LOSSES  (identical to widar_distill.py)
# ─────────────────────────────────────────────────────────────
def dkd_loss(student_logits, teacher_logits, targets, T, alpha, beta,
             label_smoothing=0.1):
    B, C = student_logits.shape
    hard_loss = F.cross_entropy(student_logits, targets,
                                label_smoothing=label_smoothing)

    p_t = F.softmax(teacher_logits / T, dim=1)
    p_s = F.softmax(student_logits / T, dim=1)

    target_mask = F.one_hot(targets, C).bool()
    pt_t = p_t[target_mask].unsqueeze(1)
    pt_s = p_s[target_mask].unsqueeze(1)
    bern_t = torch.cat([pt_t, 1.0 - pt_t], dim=1)
    bern_s = torch.cat([pt_s, 1.0 - pt_s], dim=1).clamp_min(1e-8)
    tckd   = F.kl_div(bern_s.log(), bern_t, reduction='batchmean') * (T ** 2)

    t_masked = teacher_logits.clone()
    s_masked = student_logits.clone()
    t_masked[target_mask] = -1e9
    s_masked[target_mask] = -1e9
    nckd = F.kl_div(
        F.log_softmax(s_masked / T, dim=1),
        F.softmax(t_masked    / T, dim=1),
        reduction='batchmean'
    ) * (T ** 2)

    return hard_loss + alpha * tckd + beta * nckd


def attention_map(feat):
    att = feat.pow(2).mean(dim=1)
    att = att.view(att.size(0), -1)
    return F.normalize(att, p=2, dim=1)


def attention_transfer_loss(t_maps, s_maps):
    loss = 0.0
    for t_a, s_a in zip(t_maps[:3], s_maps[:3]):
        loss = loss + F.mse_loss(attention_map(s_a), attention_map(t_a).detach())
    return loss


# ─────────────────────────────────────────────────────────────
# STEP A: Train FEATURE EXTRACTOR on FULL dataset
# ─────────────────────────────────────────────────────────────
def train_extractor(dst_train, test_x, test_y, save_path,
                    epochs=70, batch_size=32, val_fraction=0.1):
    labels = dataset_labels(dst_train)
    all_idx = list(range(len(dst_train)))
    train_idx, val_idx = stratified_split_indices(
        all_idx, labels, val_fraction=val_fraction, seed=0
    )
    n_train, n_val = len(train_idx), len(val_idx)
    train_sub = torch.utils.data.Subset(dst_train, train_idx)
    val_sub = torch.utils.data.Subset(dst_train, val_idx)
    print(f"  Train split : {n_train}  |  Val split : {n_val}")

    train_loader = DataLoader(train_sub, batch_size=batch_size,
                              shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_sub,   batch_size=32,
                              shuffle=False, num_workers=4)

    model = NTUFi_CNN2D(NUM_CLASSES).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    best_val_acc = -float('inf')
    print(f"\nTraining feature extractor on {n_train} samples...")
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
                for x, y in val_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    correct += (model(x).argmax(1) == y).sum().item()
                    total   += len(y)
            val_acc = 100.0 * correct / total
            print(f"{epoch+1:>6}  {total_loss/len(train_loader):>12.4f}"
                  f"  {val_acc:>9.2f}%  (val)")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), save_path)

    model.load_state_dict(torch.load(save_path, map_location=DEVICE,
                                     weights_only=False))
    model.eval()
    test_acc = _eval_acc(model, test_x, test_y, DEVICE)

    print(f"\nBest val accuracy  : {best_val_acc:.2f}%")
    print(f"Test accuracy      : {test_acc:.2f}%")
    print(f"Extractor saved to : {save_path}")
    if test_acc < 60:
        print("WARNING: test accuracy < 60% — try --extractor_epochs 100")
    return test_acc


# ─────────────────────────────────────────────────────────────
# STEP B: Extract discriminative features (for selection only)
# ─────────────────────────────────────────────────────────────
def extract_features(model, dataset, device, batch_size=32):
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

    all_feats = torch.cat(all_feats).numpy()
    all_y     = torch.cat(all_y).numpy()
    print(f"Extracted CNN2D features: {all_feats.shape}")

    print("\nClass separation check (larger spread = better features):")
    for c in range(NUM_CLASSES):
        mask = all_y == c
        print(f"  Class {c} ({CLASS_NAMES[c]}): "
              f"mean_norm={np.linalg.norm(all_feats[mask].mean(0)):.3f}  "
              f"std={all_feats[mask].std():.3f}")

    return all_feats, all_y


def reduce_feature_dim(features, n_components=128, random_state=42):
    scaler = StandardScaler()
    feat_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=n_components, random_state=random_state)
    feat_reduced = pca.fit_transform(feat_scaled)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA reduced to {n_components} dims "
          f"(explained variance = {explained:.3f})")
    return feat_reduced


# ─────────────────────────────────────────────────────────────
# SELECTION METHODS
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


def dpp_select(features, labels, n_per_class, num_classes=NUM_CLASSES,
               random_state=42):
    try:
        from dppy.finite_dpps import FiniteDPP
    except ImportError:
        raise ImportError("Run: pip install dppy --break-system-packages")

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


def random_select(features, labels, n_per_class, num_classes=NUM_CLASSES,
                   random_state=42):
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


def random_select_from_labels(labels, n_per_class, num_classes=NUM_CLASSES,
                              random_state=42):
    rng = np.random.default_rng(random_state)
    selected = []
    for c in range(num_classes):
        cls_idx = np.where(labels == c)[0]
        print(f"  Class {c} ({CLASS_NAMES[c]}): {len(cls_idx)} → {n_per_class}")
        if len(cls_idx) <= n_per_class:
            selected.extend(cls_idx.tolist())
        else:
            selected.extend(rng.choice(cls_idx, n_per_class,
                                       replace=False).tolist())
    print(f"Random selected: {len(selected)}")
    return selected





def select_coreset(method, features, labels, n_per_class, num_classes=NUM_CLASSES,
                   random_state=42):
    if method == 'herding':
        return herding_select(features, labels, n_per_class)
    if method == 'DPP':
        return dpp_select(features, labels, n_per_class, num_classes,
                          random_state=random_state)
    if method == 'kmeans':
        return kmeans_select(features, labels, n_per_class)
    if method == 'random':
        return random_select(features, labels, n_per_class, num_classes,
                             random_state=random_state)
    raise ValueError(f"Unknown selection method for single-method dispatch: {method}")


# ─────────────────────────────────────────────────────────────
# SCHEDULER (warmup + cosine decay)
# ─────────────────────────────────────────────────────────────
def make_scheduler_with_warmup(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────
# PHASE 1: Train the KD TEACHER on the coreset
#          Tracks (epoch, test_acc) at every eval checkpoint and
#          explicitly reports + selects the best epoch.
# ─────────────────────────────────────────────────────────────
def train_teacher_on_coreset(dist_x, dist_y, test_x, test_y, num_classes, device,
                             epochs=300, label_smoothing=0.1,
                             warmup_epochs=10, dropout=0.3, seed=0,
                             eval_every=25, print_curve=True):
    torch.manual_seed(seed)

    n_coreset = len(dist_x)
    loader = DataLoader(
        torch.utils.data.TensorDataset(dist_x, dist_y),
        batch_size=min(32, n_coreset), shuffle=True
    )

    teacher = NTUFi_CNN2D_Teacher(num_classes, dropout=dropout).to(device)
    opt   = torch.optim.Adam(teacher.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = make_scheduler_with_warmup(opt, warmup_epochs, epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    history = []
    best_acc, best_state = -1.0, None

    for epoch in range(epochs):
        teacher.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(teacher(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            acc = _eval_acc(teacher, test_x, test_y, device)
            history.append((epoch + 1, acc))
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.detach().cpu().clone()
                              for k, v in teacher.state_dict().items()}

    if print_curve:
        print_accuracy_curve(history, label="Teacher")

    teacher.load_state_dict(best_state)
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    return teacher, best_acc, history


# ─────────────────────────────────────────────────────────────
# EVALUATION: train teacher on coreset (Phase 1), then distil
#             student from it, same coreset (Phase 2).
#             Student is ALSO evaluated periodically and the
#             best-accuracy epoch is selected, instead of always
#             using the final epoch.
#             Returns the trained teacher/student objects (last run)
#             in addition to the accuracy summary, so callers can run
#             diagnostics (confusion matrices, per-class comparisons)
#             without retraining.
# ─────────────────────────────────────────────────────────────
def evaluate_distilled(selected_idx, dst_train, test_x, test_y,
                       num_classes, device,
                       n_runs=3,
                       teacher_epochs=300,
                       teacher_dropout=0.3,
                       teacher_eval_every=25,
                       label_smoothing=0.1,
                       warmup_epochs=10,
                       student_epochs=100,
                       student_eval_every=10,
                       temperature=4.0,
                       alpha=1.0,
                       beta=2.0,
                       lambda_feat=0.1,
                       lambda_at=0.5):
    print(f"  Building coreset ({len(selected_idx)} samples)...")
    dist_x = torch.stack([dst_train[i][0] for i in selected_idx])
    dist_y = torch.tensor([dst_train[i][1] for i in selected_idx],
                           dtype=torch.long)

    def per_class_acc(model):
        loader = DataLoader(
            torch.utils.data.TensorDataset(test_x, test_y),
            batch_size=32, shuffle=False
        )
        model.eval()
        tp  = torch.zeros(num_classes)
        cnt = torch.zeros(num_classes)
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                preds  = model(xb).argmax(1)
                for c in range(num_classes):
                    mask    = (yb == c)
                    tp[c]  += (preds[mask] == c).sum().item()
                    cnt[c] += mask.sum().item()
        return {CLASS_NAMES[c]: f"{100.*tp[c]/cnt[c]:.1f}%"
                if cnt[c] > 0 else 'N/A'
                for c in range(num_classes)}

    use_feat_hint = lambda_feat > 0
    use_at        = lambda_at   > 0

    print(f"\n  DKD config : T={temperature}  α(TCKD)={alpha}  β(NCKD)={beta}")
    if use_at:
        print(f"  AT loss    : enabled  (λ_at={lambda_at})")
    if use_feat_hint:
        print(f"  Hint loss  : enabled  (λ_feat={lambda_feat}, "
              f"student 384 → teacher 512 adapter)")
    print(f"  Teacher    : trained FROM SCRATCH on the coreset each run "
          f"({teacher_epochs} epochs, dropout={teacher_dropout}, "
          f"eval every {teacher_eval_every})")
    print(f"  Student    : best-epoch selected over {student_epochs} epochs "
          f"(eval every {student_eval_every})")

    teacher_accs = []
    student_accs = []
    teacher, student = None, None

    for run in range(n_runs):
        seed = run * 42
        print(f"\n  ── Run {run+1}/{n_runs} ──────────────────────────────")

        teacher, teacher_acc, _ = train_teacher_on_coreset(
            dist_x, dist_y, test_x, test_y, num_classes, device,
            epochs=teacher_epochs, label_smoothing=label_smoothing,
            warmup_epochs=warmup_epochs, dropout=teacher_dropout, seed=seed,
            eval_every=teacher_eval_every
        )
        teacher_total = count_params(teacher)
        teacher_accs.append(teacher_acc)
        print(f"  Teacher (coreset-trained) best test acc : {teacher_acc:.2f}%")
        print(f"  Teacher params                            : {teacher_total:,}")

        n_coreset = len(dist_x)
        train_loader_s = DataLoader(
            torch.utils.data.TensorDataset(dist_x, dist_y),
            batch_size=min(32, n_coreset), shuffle=True
        )

        student = NTUFi_CNN2D_Student(num_classes).to(device)
        student_total = count_params(student)
        print(f"  Student params : {student_total:,}  "
              f"({teacher_total / student_total:.1f}× smaller than teacher)")

        adapter = None
        if use_feat_hint:
            adapter = nn.Linear(384, 512).to(device)

        s_params = list(student.parameters())
        if adapter is not None:
            s_params += list(adapter.parameters())

        s_opt   = torch.optim.Adam(s_params, lr=1e-3, weight_decay=1e-4)
        s_sched = make_scheduler_with_warmup(s_opt, warmup_epochs,
                                             student_epochs)

        student_history = []
        best_student_acc, best_student_state = -1.0, None

        for epoch in range(student_epochs):
            student.train()
            if adapter is not None:
                adapter.train()

            for xb, yb in train_loader_s:
                xb, yb = xb.to(device), yb.to(device)

                with torch.no_grad():
                    t_a1, t_a2, t_a3, t_feats = teacher.get_attention_maps(xb)
                    t_logits = teacher.fc2(t_feats)

                s_a1, s_a2, s_a3, s_feats = student.get_attention_maps(xb)
                s_logits = student.fc2(s_feats)

                loss = dkd_loss(s_logits, t_logits, yb,
                                T=temperature, alpha=alpha, beta=beta,
                                label_smoothing=label_smoothing)

                if use_at:
                    loss = loss + lambda_at * attention_transfer_loss(
                        (t_a1, t_a2, t_a3), (s_a1, s_a2, s_a3)
                    )

                if use_feat_hint and adapter is not None:
                    t_feats_norm = F.normalize(t_feats, p=2, dim=1)
                    s_feats_adapted_norm = F.normalize(adapter(s_feats), p=2, dim=1)
                    loss = loss + lambda_feat * F.mse_loss(
                        s_feats_adapted_norm, t_feats_norm.detach()
                    )

                s_opt.zero_grad()
                loss.backward()
                s_opt.step()

            s_sched.step()

            if (epoch + 1) % student_eval_every == 0 or epoch == student_epochs - 1:
                acc = _eval_acc(student, test_x, test_y, device)
                student_history.append((epoch + 1, acc))
                if acc > best_student_acc:
                    best_student_acc = acc
                    best_student_state = {
                        k: v.detach().cpu().clone()
                        for k, v in student.state_dict().items()
                    }

        print_accuracy_curve(student_history, label="Student")

        student.load_state_dict(best_student_state)
        test_acc = best_student_acc
        student_accs.append(test_acc)
        print(f"  Student best test acc: {test_acc:.2f}%  "
              f"(gap vs this run's teacher = {teacher_acc - test_acc:+.2f}%)")

    t_mean, t_std = float(np.mean(teacher_accs)), float(np.std(teacher_accs))
    s_mean, s_std = float(np.mean(student_accs)), float(np.std(student_accs))

    print(f"\n  Teacher (coreset-trained, {n_runs} runs) : "
          f"{t_mean:.2f}% ± {t_std:.2f}%")
    print(f"  Student ({n_runs} runs)                   : "
          f"{s_mean:.2f}% ± {s_std:.2f}%  (gap = {t_mean - s_mean:+.2f}%)")

    print("\n  Per-class accuracy (student, last run, best epoch):")
    for cls, acc in per_class_acc(student).items():
        print(f"    {cls:<10} {acc}")

    print("\n  Per-class accuracy (teacher, last run, best epoch):")
    for cls, acc in per_class_acc(teacher).items():
        print(f"    {cls:<10} {acc}")

    return t_mean, t_std, s_mean, s_std, teacher, student


def train_student_ce_baseline(dst_train, test_x, test_y, num_classes, device,
                              source_indices=None, n_runs=3, epochs=100,
                              batch_size=32, label_smoothing=0.1,
                              warmup_epochs=10, val_fraction=0.1,
                              name='student_ce', use_validation=None,
                              eval_every=10):
    full_data_training = source_indices is None
    labels = dataset_labels(dst_train)

    if full_data_training:
        source_indices = list(range(len(dst_train)))
        print("\nCE baseline: student trained on full training data")
    else:
        source_indices = list(source_indices)
        print(f"\nCE baseline: student trained on {len(source_indices)} samples")

    if use_validation is None:
        use_validation = full_data_training

    accs = []
    student = None
    for run in range(n_runs):
        seed = run * 42
        torch.manual_seed(seed)
        np.random.seed(seed)

        if use_validation:
            train_idx, val_idx = stratified_split_indices(
                source_indices, labels, val_fraction=val_fraction, seed=seed
            )
            if not val_idx:
                print("  Empty validation split; using final-epoch checkpoint.")
                train_idx = source_indices
                use_val_this_run = False
            else:
                use_val_this_run = True
        else:
            train_idx = source_indices
            val_idx = []
            use_val_this_run = False

        train_x = torch.stack([dst_train[i][0] for i in train_idx])
        train_y = torch.tensor([dst_train[i][1] for i in train_idx],
                               dtype=torch.long)
        train_loader = DataLoader(
            torch.utils.data.TensorDataset(train_x, train_y),
            batch_size=min(batch_size, len(train_idx)), shuffle=True
        )

        val_loader = None
        if use_val_this_run:
            val_x = torch.stack([dst_train[i][0] for i in val_idx])
            val_y = torch.tensor([dst_train[i][1] for i in val_idx],
                                 dtype=torch.long)
            val_loader = DataLoader(
                torch.utils.data.TensorDataset(val_x, val_y),
                batch_size=32, shuffle=False
            )

        student = NTUFi_CNN2D_Student(num_classes).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        opt = torch.optim.Adam(student.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = make_scheduler_with_warmup(opt, warmup_epochs, epochs)

        history = []
        best_val_acc = -float('inf')
        best_state = None

        for epoch in range(epochs):
            student.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = criterion(student(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()

            if use_val_this_run and ((epoch + 1) % eval_every == 0 or epoch == epochs - 1):
                student.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        correct += (student(xb).argmax(1) == yb).sum().item()
                        total += len(yb)
                val_acc = 100.0 * correct / total
                history.append((epoch + 1, val_acc))
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in student.state_dict().items()
                    }
            elif not use_val_this_run and epoch == epochs - 1:
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in student.state_dict().items()
                }

        if history:
            print_accuracy_curve(history, label=f"{name} (val)")

        student.load_state_dict(best_state)
        student.to(device)
        test_acc = _eval_acc(student, test_x, test_y, device)
        accs.append(test_acc)
        if use_val_this_run:
            print(f"  {name} run {run+1}/{n_runs}: "
                  f"best_val={best_val_acc:.2f}% test={test_acc:.2f}%")
        else:
            print(f"  {name} run {run+1}/{n_runs}: "
                  f"final_epoch test={test_acc:.2f}%")

    mean_acc = float(np.mean(accs))
    std_acc = float(np.std(accs))
    print(f"\n{name}: {mean_acc:.2f}% +/- {std_acc:.2f}%")
    return mean_acc, std_acc, student


def export_student_onnx(student, path):
    """Export the NTU-Fi student. Input is (batch, 3, 114, 500)."""
    student.eval()
    example = torch.randn(
        1, 3, 114, 500,
        device=next(student.parameters()).device,
    )
    torch.onnx.export(
        student,
        example,
        path,
        input_names=["x"],
        output_names=["logits"],
        opset_version=18,
        dynamo=True,
        external_data=False,
    )
    print(f"Student exported to: {os.path.abspath(path)}", flush=True)
    return path


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main(args):
    print(f"Device: {DEVICE}")

    train_root = os.path.join(args.data_root, 'train_amp')
    test_root  = os.path.join(args.data_root, 'test_amp')

    dst_train = NTUFi_Dataset(train_root)
    dst_test  = NTUFi_Dataset(test_root)
    print(f"Train: {len(dst_train)}  |  Test: {len(dst_test)}")

    test_x, test_y = build_test_tensors(dst_test)

    if args.mode == 'train_extractor':
        print("\n=== TRAINING FULL-DATA FEATURE EXTRACTOR ===")
        best = train_extractor(dst_train, test_x, test_y,
                               save_path=args.extractor_path,
                               epochs=args.extractor_epochs,
                               val_fraction=args.val_fraction)
        print(f"\nDone. Extractor test accuracy: {best:.2f}%")
        print("Now run with --mode select to do selection + coreset-teacher + KD.")
        return

    if args.baseline in ('student_full_data', 'full_data'):
        train_student_ce_baseline(
            dst_train, test_x, test_y, NUM_CLASSES, DEVICE,
            source_indices=None,
            n_runs=args.n_runs,
            epochs=args.student_epochs,
            label_smoothing=args.label_smoothing,
            warmup_epochs=args.warmup_epochs,
            val_fraction=args.val_fraction,
            name='student_full_data_ce',
            eval_every=args.student_eval_every,
        )
        return

    print(f"\nIPC    : {args.ipc} per class ({args.ipc * NUM_CLASSES} total)")
    print(f"Method : {args.method}")
    print(f"\nKD config (DKD):")
    print(f"  Temperature  : {args.temperature}")
    print(f"  Alpha (TCKD) : {args.alpha}")
    print(f"  Beta  (NCKD) : {args.beta}")
    print(f"  Lambda AT    : {args.lambda_at}")
    print(f"  Lambda feat  : {args.lambda_feat}")
    print(f"  Teacher      : {args.teacher_epochs} epochs, "
          f"dropout={args.teacher_dropout}, eval every {args.teacher_eval_every}")
    print(f"  Student      : {args.student_epochs} epochs, "
          f"eval every {args.student_eval_every}")
    print(f"  Baseline      : {args.baseline}")

    if args.baseline in ('random_ce', 'random_kd'):
        train_labels = dataset_labels(dst_train)
        idx = random_select_from_labels(
            train_labels, args.ipc, NUM_CLASSES, random_state=args.seed
        )
        if args.baseline == 'random_ce':
            train_student_ce_baseline(
                dst_train, test_x, test_y, NUM_CLASSES, DEVICE,
                source_indices=idx,
                n_runs=args.n_runs,
                epochs=args.student_epochs,
                label_smoothing=args.label_smoothing,
                warmup_epochs=args.warmup_epochs,
                val_fraction=args.val_fraction,
                name=f'random_ce_ipc{args.ipc}',
                use_validation=False,
                eval_every=args.student_eval_every,
            )
        else:
            evaluate_distilled(
                idx, dst_train, test_x, test_y,
                NUM_CLASSES, DEVICE,
                n_runs=args.n_runs,
                teacher_epochs=args.teacher_epochs,
                teacher_dropout=args.teacher_dropout,
                teacher_eval_every=args.teacher_eval_every,
                label_smoothing=args.label_smoothing,
                warmup_epochs=args.warmup_epochs,
                student_epochs=args.student_epochs,
                student_eval_every=args.student_eval_every,
                temperature=args.temperature,
                alpha=args.alpha,
                beta=args.beta,
                lambda_feat=args.lambda_feat,
                lambda_at=args.lambda_at,
            )
        return

    if not os.path.exists(args.extractor_path):
        raise FileNotFoundError(
            f"Extractor checkpoint not found at '{args.extractor_path}'.\n"
            f"Run first:  python {__file__} --mode train_extractor"
        )
    print(f"\nLoading feature extractor from {args.extractor_path}...")
    extractor = NTUFi_CNN2D(NUM_CLASSES).to(DEVICE)
    extractor.load_state_dict(
        torch.load(args.extractor_path, map_location=DEVICE,
                   weights_only=False)
    )
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad = False
    print(f"Extractor loaded ✓  params={count_params(extractor):,}")

    if args.baseline == 'teacher_only':
        features, labels = extract_features(extractor, dst_train, DEVICE)
        if args.feat_dim < features.shape[1]:
            features = reduce_feature_dim(features, n_components=args.feat_dim)
        method_for_teacher_only = (args.method if args.method != 'all'
                                   else 'herding')
        idx = select_coreset(method_for_teacher_only, features, labels,
                             args.ipc, NUM_CLASSES, random_state=args.seed)

        dist_x = torch.stack([dst_train[i][0] for i in idx])
        dist_y = torch.tensor([dst_train[i][1] for i in idx], dtype=torch.long)

        accs = []
        for run in range(args.n_runs):
            _, acc, _ = train_teacher_on_coreset(
                dist_x, dist_y, test_x, test_y, NUM_CLASSES, DEVICE,
                epochs=args.teacher_epochs,
                label_smoothing=args.label_smoothing,
                warmup_epochs=args.warmup_epochs,
                dropout=args.teacher_dropout, seed=run * 42,
                eval_every=args.teacher_eval_every,
            )
            accs.append(acc)
        print(f"\nCoreset-trained teacher ({method_for_teacher_only}): "
              f"{np.mean(accs):.2f}% ± {np.std(accs):.2f}%")
        return

    print("\nExtracting features for coreset selection...")
    features, labels = extract_features(extractor, dst_train, DEVICE)
    if args.feat_dim < features.shape[1]:
        print("Reducing feature dimensionality via PCA...")
        features = reduce_feature_dim(features, n_components=args.feat_dim)

    kd_kwargs = dict(
        n_runs=args.n_runs,
        teacher_epochs=args.teacher_epochs,
        teacher_dropout=args.teacher_dropout,
        teacher_eval_every=args.teacher_eval_every,
        label_smoothing=args.label_smoothing,
        warmup_epochs=args.warmup_epochs,
        student_epochs=args.student_epochs,
        student_eval_every=args.student_eval_every,
        temperature=args.temperature,
        alpha=args.alpha,
        beta=args.beta,
        lambda_feat=args.lambda_feat,
        lambda_at=args.lambda_at,
    )



    results = {}

    def run_method(name, idx):
        print(f"\n{'='*60}\n{name.upper()}\n{'='*60}")
        torch.save({'indices': idx, 'method': name, 'ipc': args.ipc},
                   f'ntufi_distilled_{name}_ipc{args.ipc}.pt')

        if args.baseline == 'coreset_ce':
            s_m, s_s, student = train_student_ce_baseline(
                dst_train, test_x, test_y, NUM_CLASSES, DEVICE,
                source_indices=idx,
                n_runs=args.n_runs,
                epochs=args.student_epochs,
                label_smoothing=args.label_smoothing,
                warmup_epochs=args.warmup_epochs,
                val_fraction=args.val_fraction,
                name=f'{name}_ce_ipc{args.ipc}',
                use_validation=False,
                eval_every=args.student_eval_every,
            )
            results[name] = (float('nan'), s_m, s_s)

            confusion_matrix_report(student, test_x, test_y, DEVICE,
                                    NUM_CLASSES, CLASS_NAMES,
                                    title=f"Coreset student CE-only "
                                          f"({name}, ipc={args.ipc})")
        else:
            t_m, t_s, s_m, s_s, teacher, student = evaluate_distilled(
                idx, dst_train, test_x, test_y, NUM_CLASSES, DEVICE,
                **kd_kwargs
            )
            results[name] = (t_m, s_m, s_s)

            confusion_matrix_report(student, test_x, test_y, DEVICE,
                                    NUM_CLASSES, CLASS_NAMES,
                                    title=f"Coreset student "
                                          f"({name}, ipc={args.ipc})")
            confusion_matrix_report(teacher, test_x, test_y, DEVICE,
                                    NUM_CLASSES, CLASS_NAMES,
                                    title=f"Coreset teacher "
                                          f"({name}, ipc={args.ipc})")

        # Export the student returned by the selected training path (KD or CE).
        # With n_runs > 1 this is the final run's best-epoch student.
        onnx_path = f"ntufi_student_{name}_ipc{args.ipc}.onnx"
        export_student_onnx(student, onnx_path)

        print(f"{name}: teacher={results[name][0]:.2f}%  "
              f"student={results[name][1]:.2f}%±{results[name][2]:.2f}%")

    if args.method in ('herding', 'all'):
        idx = herding_select(features, labels, args.ipc)
        run_method('herding', idx)

    if args.method in ('DPP', 'all'):
        idx = dpp_select(features, labels, args.ipc, NUM_CLASSES,
                         random_state=args.seed)
        run_method('DPP', idx)

    if args.method in ('random', 'all'):
        idx = random_select(features, labels, args.ipc, NUM_CLASSES,
                            random_state=args.seed)
        run_method('random', idx)

    if args.method in ('kmeans', 'all'):
        idx = kmeans_select(features, labels, args.ipc)
        run_method('kmeans', idx)


    print("\n" + "=" * 72)
    print(f"SUMMARY  IPC={args.ipc}  "
          f"T={args.temperature}  α={args.alpha}  β={args.beta}  "
          f"λ_feat={args.lambda_feat}  λ_at={args.lambda_at}")
    print("=" * 72)
    print("  NOTE: both teacher and student now select their BEST epoch "
          "across training, not just the final one.")
    print(f"{'Method':<24}  {'Teacher Acc':>12}  "
          f"{'Student Acc':>12}  {'Student Std':>11}  {'Gap':>8}")
    print("-" * 74)
    for method, (t_m, s_m, s_s) in sorted(results.items(),
                                            key=lambda x: -x[1][1]):
        gap_str = f"{t_m-s_m:>+7.2f}%" if not np.isnan(t_m) else "   n/a"
        t_str = f"{t_m:>11.2f}%" if not np.isnan(t_m) else "      n/a"
        print(f"{method:<24}  {t_str}  "
              f"{s_m:>11.2f}%  {s_s:>10.2f}%  {gap_str}")
    print("=" * 72)
    print("Coreset indices saved to ntufi_distilled_<method>_ipc<N>.pt")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='NTU-Fi HAR coreset selection + coreset-trained-teacher KD '
                    '(cached test set, full accuracy-curve tracking, '
                    'best-epoch selection for both teacher and student) '
                    '— same procedure as widar_distill.py'
    )

    parser.add_argument('--mode', type=str, default='select',
                        choices=['train_extractor', 'select'])

    parser.add_argument('--data_root', type=str, default=DATA_ROOT)
    parser.add_argument('--extractor_path',   type=str,
                        default="cnn2d_extractor_ntu.pt")
    parser.add_argument('--extractor_epochs', type=int, default=70)
    parser.add_argument('--feat_dim', type=int, default=512,
                        help='PCA-reduce features to this dim before selection '
                             '(default = no reduction, 512-d)')

    parser.add_argument('--ipc',    type=int, default=10)
    parser.add_argument('--method', type=str, default='all',
                        choices=['DPP', 'kmeans', 'all',
                                'herding', 'random'])
    parser.add_argument('--baseline', type=str, default='none',
                        choices=[
                            'none', 'teacher_only', 'student_full_data',
                            'full_data', 'random_ce', 'random_kd',
                            'coreset_ce',
                        ])

    parser.add_argument('--teacher_epochs', type=int, default=300,
                        help='Epochs to train the KD teacher on the coreset')
    parser.add_argument('--teacher_dropout', type=float, default=0.3)
    parser.add_argument('--teacher_eval_every', type=int, default=25,
                        help='Evaluate + record teacher test acc every N epochs')

    parser.add_argument('--student_eval_every', type=int, default=10,
                        help='Evaluate + record student test acc every N epochs')

    parser.add_argument('--n_runs',          type=int,   default=3)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--warmup_epochs',   type=int,   default=10)
    parser.add_argument('--val_fraction',    type=float, default=0.1)
    parser.add_argument('--student_epochs',  type=int,   default=100)
    parser.add_argument('--temperature',     type=float, default=4.0)
    parser.add_argument('--alpha',           type=float, default=1.0)
    parser.add_argument('--beta',            type=float, default=2.0)
    parser.add_argument('--lambda_feat',     type=float, default=0.0)
    parser.add_argument('--lambda_at',       type=float, default=0.0)
    parser.add_argument('--seed',            type=int, default=42)


    args = parser.parse_args()
    main(args)
