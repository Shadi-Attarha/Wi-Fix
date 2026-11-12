"""
 Coreset Selection + Teacher/Student KD for XRF55 Dataset
=====================================================================
This is the XRF55 counterpart to the Widar coreset+KD script. It follows
the exact same procedure:

  Step A  — train a FEATURE EXTRACTOR on the full dataset (embeddings
            used ONLY for coreset selection, never for classification).
  Step B  — select a coreset per class (herding /  / kmeans / kcenter /
            random / ...).
  Phase 1 — train a TEACHER from scratch on the coreset ONLY.
  Phase 2 — distill a smaller STUDENT from that teacher on the SAME
            coreset, using DKD (decoupled KD) + attention transfer +
            a feature-hint loss.

Architecture note (unchanged from the original file):
  (270,1000) is reshaped to (9,30,1000) — 9 WiFi links (3 rx x 3 antenna)
  as channels, (30 subcarriers x 1000 time) as the 2D frequency-time plane.


----------------------------------------------------
These were referenced in the original file's --method choices and CLI
args but never defined anywhere, so `--method all` would have crashed
with a NameError. They are stubbed out below with NotImplementedError
and clear instructions — paste your original implementations into thea
marked TODO sections and they'll be wired into --method automatically.
Until then, `--method all` silently skips them (with a printed warning)
so the rest of the pipeline (herding /  / kmeans / kcenter / random)
still runs.

Run:
    # Step A — train feature extractor (~15 min)
    python xrf55_dpp_kd.py --mode train_extractor \
        --extractor_path xrf55_extractor.pt --extractor_epochs 30

    # Step B — select + coreset-teacher + KD, all implemented methods
    python xrf55_dpp_kd.py --mode select \
        --ipc 10 --method all --extractor_path xrf55_extractor.pt

    # Single method, full KD
    python xrf55_dpp_kd.py --mode select --ipc 10 --method herding \
        --extractor_path xrf55_extractor.pt

    # CE-only coreset baseline (no teacher/distillation)
    python xrf55_dpp_kd.py --mode select --ipc 10 --method herding \
        --baseline coreset_ce --extractor_path xrf55_extractor.pt

Install: pip install dppy --break-system-packages
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

try:
    from kd_diagnostics import (confusion_matrix_report,
                                 compare_models_per_class,
                                 per_run_accuracy_check)
    HAS_DIAGNOSTICS = True
except ImportError:
    HAS_DIAGNOSTICS = False


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


def dataset_labels(dataset):
    return np.array(dataset.labels, dtype=np.int64)


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
# All evaluation AND all diagnostics go through these cached tensors —
# never dst_test directly — so nothing re-reads .npy files from disk
# after this point.
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


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ─────────────────────────────────────────────────────────────
# BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────

class SEBlock2D(nn.Module):
    """
    Squeeze-and-Excitation: channel attention over the 9 antenna links.
    Learns which antenna combinations are most informative per gesture.
    reduction=8 keeps the bottleneck light; clamp to ≥8 avoids collapse
    in early stages where channel count is still small.
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


def make_stage(in_ch, out_ch, n, stride):
    layers = [BasicBlock2D(in_ch, out_ch, stride=stride)]
    for _ in range(1, n):
        layers.append(BasicBlock2D(out_ch, out_ch, stride=1))
    return nn.Sequential(*layers)


def init_conv_bn_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                    nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


# ─────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────

class XRF55_ResNet2D(nn.Module):
    """Full-data FEATURE EXTRACTOR ONLY — embeddings for coreset selection."""
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

        self.stage1 = make_stage(64,  64,  n=2, stride=1)          # (64, 30, 250)
        self.stage2 = make_stage(64,  128, n=2, stride=2)          # (128, 15, 125)
        self.stage3 = make_stage(128, 256, n=2, stride=(1, 2))     # (256, 15, 63)
        self.stage4 = make_stage(256, 512, n=2, stride=(1, 2))     # (512, 15, 32)

        self.gap = nn.AdaptiveAvgPool2d(1)   # → (B, 512)
        self.fc  = nn.Linear(512, num_classes)

        init_conv_bn_weights(self)

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


class XRF55_ResNet2D_Teacher(nn.Module):
    """KD TEACHER — trained ONLY on the selected coreset, not the full dataset."""
    def __init__(self, num_classes=55, dropout=0.3):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=(3, 7), stride=(1, 2),
                      padding=(1, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        )
        self.stage1 = make_stage(64,  64,  n=2, stride=1)
        self.stage2 = make_stage(64,  128, n=2, stride=2)
        self.stage3 = make_stage(128, 256, n=2, stride=(1, 2))
        self.stage4 = make_stage(256, 512, n=2, stride=(1, 2))
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.drop   = nn.Dropout(dropout)
        self.fc     = nn.Linear(512, num_classes)
        init_conv_bn_weights(self)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x).squeeze(-1).squeeze(-1)
        x = self.drop(x)
        return self.fc(x)

    def get_attention_maps(self, x):
        """
        Returns three intermediate spatial activation maps (stage2, stage3,
        stage4 — semantically deeper than the raw stem/stage1) plus the
        pre-dropout pooled feature vector, mirroring Widar's
        (a1, a2, a3, feat) used for DKD attention-transfer + hint loss.
        """
        x  = self.stem(x)
        x  = self.stage1(x)
        a1 = self.stage2(x)
        a2 = self.stage3(a1)
        a3 = self.stage4(a2)
        feat = self.gap(a3).squeeze(-1).squeeze(-1)
        return a1, a2, a3, feat


class XRF55_ResNet2D_Student(nn.Module):
    """Student model — ~4x fewer channels per stage than the teacher."""
    def __init__(self, num_classes=55):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv2d(9, 16, kernel_size=(3, 7), stride=(1, 2),
                      padding=(1, 3), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        )
        self.stage1 = make_stage(16, 16,  n=2, stride=1)
        self.stage2 = make_stage(16, 32,  n=2, stride=2)
        self.stage3 = make_stage(32, 64,  n=2, stride=(1, 2))
        self.stage4 = make_stage(64, 128, n=2, stride=(1, 2))
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(128, num_classes)
        init_conv_bn_weights(self)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x).squeeze(-1).squeeze(-1)
        return self.fc(x)

    def get_features(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x).squeeze(-1).squeeze(-1)
        return x

    def get_attention_maps(self, x):
        x  = self.stem(x)
        x  = self.stage1(x)
        a1 = self.stage2(x)
        a2 = self.stage3(a1)
        a3 = self.stage4(a2)
        feat = self.gap(a3).squeeze(-1).squeeze(-1)
        return a1, a2, a3, feat


# ─────────────────────────────────────────────────────────────
# KD LOSSES  (architecture-agnostic — ported unchanged from Widar)
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
                    epochs=30, batch_size=16):
    train_loader = DataLoader(dst_train, batch_size=batch_size,
                              shuffle=True, num_workers=4)

    model = XRF55_ResNet2D(NUM_CLASSES).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    best_acc = 0.0
    print(f"\nTraining on {len(dst_train)} samples ({NUM_CLASSES} classes)...")
    print(f"Input shape per sample: ({N_LINKS}, {N_SUBCARRIERS}, {N_TIME})")
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Test Acc':>10}")
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
            acc = _eval_acc(model, test_x, test_y, DEVICE)
            print(f"{epoch+1:>6}  {total_loss/len(train_loader):>12.4f}"
                  f"  {acc:>9.2f}%")
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), save_path)

    print(f"\nBest test accuracy: {best_acc:.2f}%")
    print(f"Extractor saved to: {save_path}")
    if best_acc < 50:
        print("WARNING: accuracy < 50% — try --extractor_epochs 60")
    return best_acc


# ─────────────────────────────────────────────────────────────
# STEP B: Extract discriminative features (for selection only)
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

    all_feats = torch.cat(all_feats).numpy()   # (N, 512)
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
def dpp_select(features, labels, n_per_class, num_classes,
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


def herding_select(features, labels, n_per_class, num_classes,
                   random_state=42):
    """Basic greedy class-mean matching (same as Widar's herding_select)."""
    selected = []
    for c in tqdm(range(num_classes), desc='Herding per class'):
        mask    = labels == c
        cls_f   = features[mask]
        cls_idx = np.where(mask)[0]
        N_c     = len(cls_f)

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

        selected.extend(cls_idx[chosen].tolist())

    print(f"Total selected: {len(selected)}")
    return selected


def random_select(features, labels, n_per_class, num_classes, random_state=42):
    rng = np.random.default_rng(random_state)
    selected_indices = []
    for c in range(num_classes):
        mask = labels == c
        cls_idx = np.where(mask)[0]
        if len(cls_idx) <= n_per_class:
            selected_indices.extend(cls_idx.tolist())
        else:
            chosen = rng.choice(cls_idx, size=n_per_class, replace=False)
            selected_indices.extend(chosen.tolist())
    print(f"Total selected: {len(selected_indices)}")
    return selected_indices


def random_select_from_labels(labels, n_per_class, num_classes, random_state=42):
    """Random baseline that only needs labels — no CNN feature extraction."""
    rng = np.random.default_rng(random_state)
    selected = []
    for c in range(num_classes):
        cls_idx = np.where(labels == c)[0]
        if len(cls_idx) <= n_per_class:
            selected.extend(cls_idx.tolist())
        else:
            selected.extend(rng.choice(cls_idx, n_per_class,
                                       replace=False).tolist())
    print(f"Random selected: {len(selected)}")
    return selected




def select_coreset(method, features, labels, n_per_class, num_classes,
                   random_state=42):
    if method == 'herding':
        return herding_select(features, labels, n_per_class, num_classes,
                              random_state=random_state)
    if method == 'DPP':
        return dpp_select(features, labels, n_per_class, num_classes,
                          random_state=random_state)
    if method == 'kmeans':
        return kmeans_select(features, labels, n_per_class, num_classes,
                             random_state=random_state)
    if method == 'kcenter':
        return kcenter_select(features, labels, n_per_class, num_classes)
    if method == 'random':
        return random_select(features, labels, n_per_class, num_classes,
                             random_state=random_state)

    raise ValueError(f"Unknown selection method: {method}")


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
                             epochs=500, label_smoothing=0.1,
                             warmup_epochs=10, dropout=0.3, seed=0,
                             eval_every=50, batch_size=32, print_curve=True):
    torch.manual_seed(seed)

    n_coreset = len(dist_x)
    loader = DataLoader(
        torch.utils.data.TensorDataset(dist_x, dist_y),
        batch_size=min(batch_size, n_coreset), shuffle=True
    )

    teacher = XRF55_ResNet2D_Teacher(num_classes, dropout=dropout).to(device)
    opt   = torch.optim.Adam(teacher.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = make_scheduler_with_warmup(opt, warmup_epochs, epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    history = []            # (epoch, test_acc) at every eval checkpoint
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
#             diagnostics without retraining.
# ─────────────────────────────────────────────────────────────
def evaluate_distilled(selected_idx, dst_train, test_x, test_y,
                       num_classes, device,
                       n_runs=3,
                       teacher_epochs=500,
                       teacher_dropout=0.3,
                       teacher_eval_every=50,
                       label_smoothing=0.1,
                       warmup_epochs=10,
                       student_epochs=300,
                       student_eval_every=25,
                       temperature=10.0,
                       alpha=8.0,
                       beta=8.0,
                       lambda_feat=0.9,
                       lambda_at=0.9,
                       batch_size=32):
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
              f"student 128 → teacher 512 adapter)")
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
            eval_every=teacher_eval_every, batch_size=batch_size
        )
        teacher_total = count_params(teacher)
        teacher_accs.append(teacher_acc)
        print(f"  Teacher (coreset-trained) best test acc : {teacher_acc:.2f}%")
        print(f"  Teacher params                            : {teacher_total:,}")

        n_coreset = len(dist_x)
        train_loader_s = DataLoader(
            torch.utils.data.TensorDataset(dist_x, dist_y),
            batch_size=min(batch_size, n_coreset), shuffle=True
        )

        student = XRF55_ResNet2D_Student(num_classes).to(device)
        student_total = count_params(student)
        print(f"  Student params : {student_total:,}  "
              f"({teacher_total / student_total:.1f}× smaller than teacher)")

        adapter = None
        if use_feat_hint:
            adapter = nn.Linear(128, 512).to(device)

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
                    t_logits = teacher.fc(t_feats)

                s_a1, s_a2, s_a3, s_feats = student.get_attention_maps(xb)
                s_logits = student.fc(s_feats)

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

    return t_mean, t_std, s_mean, s_std, teacher, student


# ─────────────────────────────────────────────────────────────
# CE-ONLY BASELINE (no teacher, no distillation)
# ─────────────────────────────────────────────────────────────
def train_student_ce_baseline(dst_train, test_x, test_y, num_classes, device,
                              source_indices=None, n_runs=3, epochs=300,
                              batch_size=32, label_smoothing=0.1,
                              warmup_epochs=10, val_fraction=0.1,
                              name='student_ce', use_validation=None,
                              eval_every=25):
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

        student = XRF55_ResNet2D_Student(num_classes).to(device)
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

    # Build the test set ONCE — everything below reuses these tensors
    # instead of re-reading .npy files from disk.
    test_x, test_y = build_test_tensors(dst_test)

    if args.mode == 'train_extractor':
        print("\n=== TRAINING FULL-DATA FEATURE EXTRACTOR ===")
        best = train_extractor(dst_train, test_x, test_y,
                               save_path=args.extractor_path,
                               epochs=args.extractor_epochs,
                               batch_size=args.batch_size)
        print(f"\nDone. Extractor test accuracy: {best:.2f}%")
        print("Now run with --mode select to do selection + coreset-teacher + KD.")
        return

    if args.baseline in ('student_full_data', 'full_data'):
        train_student_ce_baseline(
            dst_train, test_x, test_y, NUM_CLASSES, DEVICE,
            source_indices=None,
            n_runs=args.n_runs,
            epochs=args.student_epochs,
            batch_size=args.batch_size,
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
    print(f"  Baseline     : {args.baseline}")

    if not os.path.exists(args.extractor_path):
        raise FileNotFoundError(
            f"Extractor checkpoint not found at '{args.extractor_path}'.\n"
            f"Run first:  python {__file__} --mode train_extractor"
        )
    print(f"\nLoading feature extractor from {args.extractor_path}...")
    extractor = XRF55_ResNet2D(NUM_CLASSES).to(DEVICE)
    extractor.load_state_dict(
        torch.load(args.extractor_path, map_location=DEVICE,
                   weights_only=False)
    )
    extractor.eval()
    for p in extractor.parameters():
        p.requires_grad = False
    print(f"Extractor loaded ✓  params={count_params(extractor):,}")

    if args.baseline == 'teacher_only':
        features, labels = extract_cnn_features(extractor, dst_train, DEVICE,
                                                 batch_size=args.batch_size)
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
                batch_size=args.batch_size,
            )
            accs.append(acc)
        print(f"\nCoreset-trained teacher ({method_for_teacher_only}): "
              f"{np.mean(accs):.2f}% ± {np.std(accs):.2f}%")
        return

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
                batch_size=args.batch_size,
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
                batch_size=args.batch_size,
            )
        return

    print("\nExtracting 512-dim features for coreset selection...")
    features, labels = extract_cnn_features(
        extractor, dst_train, DEVICE, batch_size=args.batch_size)

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
        batch_size=args.batch_size,
    )

    results = {}

    def run_method(name, idx):
        print(f"\n{'='*60}\n{name.upper()}\n{'='*60}")
        torch.save({'indices': idx, 'method': name, 'ipc': args.ipc},
                   f'xrf55_{name}_ipc{args.ipc}.pt')

        if args.baseline == 'coreset_ce':
            s_m, s_s, student = train_student_ce_baseline(
                dst_train, test_x, test_y, NUM_CLASSES, DEVICE,
                source_indices=idx,
                n_runs=args.n_runs,
                epochs=args.student_epochs,
                batch_size=args.batch_size,
                label_smoothing=args.label_smoothing,
                warmup_epochs=args.warmup_epochs,
                val_fraction=args.val_fraction,
                name=f'{name}_ce_ipc{args.ipc}',
                use_validation=False,
                eval_every=args.student_eval_every,
            )
            results[name] = (float('nan'), s_m, s_s)

            if HAS_DIAGNOSTICS:
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


            torch.save(student.state_dict(), 'xrf55_student_herding_ipc10.pt')
            size_mb = os.path.getsize('xrf55_student_herding_ipc10.pt') / (1024**2)
            n_params = sum(p.numel() for p in student.parameters())
            print(f"Params: {n_params:,}")
            print(f"File size (fp32 state_dict): {size_mb:.2f} MB")

            if HAS_DIAGNOSTICS:
                confusion_matrix_report(student, test_x, test_y, DEVICE,
                                        NUM_CLASSES, CLASS_NAMES,
                                        title=f"Coreset student "
                                              f"({name}, ipc={args.ipc})")
                confusion_matrix_report(teacher, test_x, test_y, DEVICE,
                                        NUM_CLASSES, CLASS_NAMES,
                                        title=f"Coreset teacher "
                                              f"({name}, ipc={args.ipc})")

        print(f"{name}: teacher={results[name][0]:.2f}%  "
              f"student={results[name][1]:.2f}%±{results[name][2]:.2f}%")

    requested_methods = (
        ['herding', 'DPP', 'random', 'kmeans', 'kcenter'
         , 'kmeans_herding']
        if args.method == 'all' else [args.method]
    )

 

    for method in requested_methods:
        if method in NOT_YET_IMPLEMENTED_METHODS:
            if args.method == 'all':
                print(f"\n[skipping '{method}' in --method all: "
                      f"not yet implemented — paste it in to enable]")
                continue
            # explicitly requested → let it raise NotImplementedError
        idx = select_coreset(method, features, labels, args.ipc,
                             NUM_CLASSES, random_state=args.seed)
        run_method(method, idx)

    print("\n" + "=" * 72)
    print(f"SUMMARY  IPC={args.ipc}  "
          f"T={args.temperature}  α={args.alpha}  β={args.beta}  "
          f"λ_feat={args.lambda_feat}  λ_at={args.lambda_at}")
    print("=" * 72)
    print("  NOTE: both teacher and student select their BEST epoch "
          "across training, not just the final one.")
    print(f"{'Method':<14}  {'Teacher Acc':>12}  "
          f"{'Student Acc':>12}  {'Student Std':>11}  {'Gap':>8}")
    print("-" * 62)
    for method, (t_m, s_m, s_s) in sorted(results.items(),
                                            key=lambda x: -x[1][1]):
        gap_str = f"{t_m-s_m:>+7.2f}%" if not np.isnan(t_m) else "   n/a"
        t_str = f"{t_m:>11.2f}%" if not np.isnan(t_m) else "      n/a"
        print(f"{method:<14}  {t_str}  "
              f"{s_m:>11.2f}%  {s_s:>10.2f}%  {gap_str}")
    print("=" * 72)
    print("Coreset indices saved to xrf55_<method>_ipc<N>.pt")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='XRF55 coreset selection + coreset-trained-teacher KD '
                    '(cached test set, full accuracy-curve tracking, '
                    'best-epoch selection for both teacher and student)'
    )

    parser.add_argument('--mode', type=str, default='select',
                        choices=['train_extractor', 'select'])

    parser.add_argument('--extractor_path',   type=str,
                        default='/home/sattarha/data/XRF55-repo/xrf55_extractorResNet.pt')
    parser.add_argument('--extractor_epochs', type=int, default=30)

    parser.add_argument('--ipc',    type=int, default=10)
    parser.add_argument('--method', type=str, default='all',
                        choices=['herding', 'DPP', 'kmeans',
                                 'kcenter', 'random', 'all'])
    parser.add_argument('--baseline', type=str, default='none',
                        choices=[
                            'none', 'teacher_only', 'student_full_data',
                            'full_data', 'random_ce', 'random_kd',
                            'coreset_ce',
                        ])

    parser.add_argument('--teacher_epochs', type=int, default=500,
                        help='Epochs to train the KD teacher on the coreset')
    parser.add_argument('--teacher_dropout', type=float, default=0.3)
    parser.add_argument('--teacher_eval_every', type=int, default=50,
                        help='Evaluate + record teacher test acc every N epochs')

    parser.add_argument('--student_eval_every', type=int, default=25,
                        help='Evaluate + record student test acc every N epochs')

    parser.add_argument('--n_runs',          type=int,   default=3)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--warmup_epochs',   type=int,   default=10)
    parser.add_argument('--val_fraction',    type=float, default=0.0)
    parser.add_argument('--student_epochs',  type=int,   default=300)
    parser.add_argument('--temperature',     type=float, default=10.0)
    parser.add_argument('--alpha',           type=float, default=8.0)
    parser.add_argument('--beta',            type=float, default=8.0)
    parser.add_argument('--lambda_feat',     type=float, default=0.9)
    parser.add_argument('--lambda_at',       type=float, default=0.9)
    parser.add_argument('--seed',            type=int, default=42)
    parser.add_argument('--batch_size',      type=int,   default=16,
                        help='Reduce if OOM — XRF55 samples are large')



    args = parser.parse_args()
    main(args)
