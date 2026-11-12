"""Widar3.0 sample selection and evaluation with the user's ResNet-18.

Expected data layout:
    DATA_ROOT/train/<class_name>/*.csv
    DATA_ROOT/test/<class_name>/*.csv

Each CSV must contain 22*20*20 values (8,800), as in the original script.
Run --help for options. Requires torch, numpy, scikit-learn, and tqdm.
"""
import argparse
import glob
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class WidarDataset(Dataset):
    def __init__(self, root, class_names=None):
        self.paths = sorted(glob.glob(os.path.join(root, '*', '*.csv')))
        if not self.paths:
            raise FileNotFoundError(f'No CSV files found under {root!r}')
        found = sorted({os.path.basename(os.path.dirname(p)) for p in self.paths})
        self.class_names = class_names or found
        unknown = set(found) - set(self.class_names)
        if unknown:
            raise ValueError(f'Classes absent from training set: {sorted(unknown)}')
        self.label = {name: i for i, name in enumerate(self.class_names)}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        name = os.path.basename(os.path.dirname(p))
        x = np.genfromtxt(p, delimiter=',').astype(np.float32)
        if x.size != 8800:
            raise ValueError(f'{p} has {x.size} values; expected 8800 (22x20x20)')
        x = ((x.reshape(22, 20, 20) - 0.0025) / 0.0119).clip(-3, 3) / 3.0
        return torch.from_numpy(x), self.label[name]


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, planes, i_downsample=None, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.i_downsample = i_downsample
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.i_downsample is not None:
            identity = self.i_downsample(x)
        return self.relu(out + identity)


class WidarResNet(nn.Module):
    """ResNet-18 adapted from the supplied code; input shape is (B,22,20,20)."""
    def __init__(self, num_classes, dropout=0.0):
        super().__init__()
        self.reshape = nn.Sequential(
            nn.ConvTranspose2d(22, 3, 7, stride=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(3, 3, 7, stride=1), nn.ReLU(inplace=True))
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(2, 64)
        self.layer2 = self._make_layer(2, 128, stride=2)
        self.layer3 = self._make_layer(2, 256, stride=2)
        self.layer4 = self._make_layer(2, 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, blocks, planes, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != planes:
            downsample = nn.Sequential(nn.Conv2d(self.in_channels, planes, 1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes))
        layers = [BasicBlock(self.in_channels, planes, downsample, stride)]
        self.in_channels = planes
        layers.extend(BasicBlock(self.in_channels, planes) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def get_features(self, x):
        x = self.reshape(x)
        x = self.max_pool(self.relu(self.batch_norm1(self.conv1(x))))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return torch.flatten(self.avgpool(x), 1)

    def forward(self, x):
        return self.fc(self.dropout(self.get_features(x)))


def train(model, data, device, epochs, batch_size, save_path):
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(epochs):
        model.train(); loss_sum = 0.
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = nn.functional.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); loss_sum += loss.item()
        print(f'Epoch {epoch+1}/{epochs}: loss={loss_sum/max(1,len(loader)):.4f}')
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f'Saved weights: {save_path}')


@torch.no_grad()
def extract(model, data, device, batch_size=128):
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval(); fs=[]; ys=[]
    for x,y in tqdm(loader, desc='Extracting features'):
        fs.append(model.get_features(x.to(device)).cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(fs), np.concatenate(ys)


def extract_raw(data, original_csv=False, standardize=True, pca_dim=0, seed=42):
    """Build a row-aligned selection matrix in normalized-input or original CSV space."""
    rows=[]; labels=[]
    for i in tqdm(range(len(data)), desc='Extracting raw samples'):
        if original_csv:
            x=np.genfromtxt(data.paths[i], delimiter=',').astype(np.float32).reshape(-1)
            name=os.path.basename(os.path.dirname(data.paths[i]))
            y=data.label[name]
        else:
            x,y=data[i]
            x=x.numpy().reshape(-1)
        rows.append(x); labels.append(y)
    # CSVs must have a consistent width to form a feature matrix.
    widths={len(row) for row in rows}
    if len(widths)!=1:
        raise ValueError(f'Raw samples have inconsistent flattened sizes: {sorted(widths)}')
    matrix=np.stack(rows).astype(np.float32)
    labels=np.asarray(labels)
    if standardize:
        matrix=StandardScaler().fit_transform(matrix).astype(np.float32)
    if pca_dim>0:
        dim=min(pca_dim, matrix.shape[0]-1, matrix.shape[1])
        if dim>0:
            print(f'Applying PCA: {matrix.shape[1]} -> {dim} dimensions')
            matrix=PCA(n_components=dim, random_state=seed).fit_transform(matrix).astype(np.float32)
    print(f'Selection matrix: {matrix.shape}; labels: {labels.shape}')
    return matrix,labels


def select(features, labels, ipc, method, seed):
    rng = np.random.default_rng(seed); chosen=[]
    for c in sorted(np.unique(labels)):
        idx = np.flatnonzero(labels == c); f = features[idx]
        n = min(ipc, len(idx))
        if method == 'random' or n == len(idx):
            local = rng.choice(len(idx), n, replace=False) if n < len(idx) else np.arange(len(idx))
        elif method == 'kmeans':
            km = KMeans(n_clusters=n, random_state=seed, n_init=10).fit(f)
            local, _ = pairwise_distances_argmin_min(km.cluster_centers_, f)
            local = list(dict.fromkeys(local.tolist()))
            if len(local) < n:
                local.extend(i for i in range(len(idx)) if i not in set(local))
            local = np.asarray(local[:n])
        elif method == 'kcenter':
            local = [int(rng.integers(len(f)))]; d = np.full(len(f), np.inf)
            for _ in range(n-1):
                d = np.minimum(d, np.linalg.norm(f-f[local[-1]], axis=1)); d[local] = -np.inf
                local.append(int(np.argmax(d)))
        else:  # greedy mean matching / herding
            mean=f.mean(0); local=[]; total=np.zeros_like(mean)
            for k in range(1,n+1):
                score=np.linalg.norm(mean-(total[None,:]+f)/k, axis=1)
                if local: score[local]=np.inf
                j=int(np.argmin(score)); local.append(j); total += f[j]
        chosen.extend(idx[np.asarray(local)].tolist())
        print(f'Class {c}: selected {min(ipc,len(idx))} of {len(idx)}')
    return chosen


def evaluate(model, train_data, test_data, indices, device, epochs, batch_size, seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tr = DataLoader(Subset(train_data, indices), batch_size=batch_size, shuffle=True, num_workers=0)
    te = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
    model.to(device); opt=torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_acc=0.0
    for epoch in range(epochs):
        model.train()
        for x,y in tr:
            x,y=x.to(device),y.to(device); loss=nn.functional.cross_entropy(model(x),y)
            opt.zero_grad(); loss.backward(); opt.step()
        if (epoch+1)%10==0 or epoch+1==epochs:
            model.eval(); good=total=0
            with torch.no_grad():
                for x,y in te:
                    pred=model(x.to(device)).argmax(1).cpu(); good += (pred==y).sum().item(); total += len(y)
            acc=100*good/max(1,total)
            best_acc=max(best_acc,acc)
            print(f'Epoch {epoch+1}/{epochs}: test accuracy={acc:.2f}%')
    return best_acc


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_root', required=True, help='Folder containing train/ and test/')
    p.add_argument('--mode', choices=['train_extractor','select'], default='train_extractor')
    p.add_argument('--weights', default='widar_resnet18.pt')
    p.add_argument('--ipc', type=int, default=100)
    p.add_argument('--method', choices=['random','kmeans','kcenter','herding'], default='kmeans')
    p.add_argument('--selection_space', choices=['feature','raw','original_csv'], default='feature',
                   help='feature: ResNet embeddings; raw: normalized (22,20,20) inputs; original_csv: CSV values before normalization/reshape')
    p.add_argument('--raw_pca_dim', type=int, default=128,
                   help='PCA dimensions for raw spaces; use 0 to disable PCA')
    p.add_argument('--no_standardize_raw', action='store_true',
                   help='Skip standardization for raw and original_csv spaces')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--eval_epochs', type=int, default=100)
    p.add_argument('--n_runs', type=int, default=3,
                   help='Number of independent selection and evaluation runs')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    a=p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr=WidarDataset(os.path.join(a.data_root,'train'))
    te=WidarDataset(os.path.join(a.data_root,'test'), tr.class_names)
    print(f'Device: {device}; classes ({len(tr.class_names)}): {tr.class_names}; train={len(tr)}, test={len(te)}')
    if a.mode=='train_extractor':
        train(WidarResNet(len(tr.class_names)), tr, device, a.epochs, a.batch_size, a.weights)
        return
    if a.selection_space=='feature':
        if not os.path.isfile(a.weights):
            raise FileNotFoundError(f'{a.weights} not found; run --mode train_extractor first')
        extractor=WidarResNet(len(tr.class_names)).to(device)
        extractor.load_state_dict(torch.load(a.weights, map_location=device, weights_only=True))
        features, labels=extract(extractor,tr,device)
    else:
        features, labels=extract_raw(
            tr, original_csv=(a.selection_space=='original_csv'),
            standardize=not a.no_standardize_raw, pca_dim=a.raw_pca_dim, seed=a.seed)
    scores=[]
    for run in range(a.n_runs):
        run_seed=a.seed+run
        print(f'\n===== Run {run+1}/{a.n_runs} (seed={run_seed}) =====')
        indices=select(features,labels,a.ipc,a.method,run_seed)
        out=f'selected_{a.selection_space}_{a.method}_ipc{a.ipc}_run{run+1}.npz'
        np.savez(out, indices=np.asarray(indices), class_names=np.asarray(tr.class_names))
        print(f'Saved selected dataset indices to {out}')
        print('Training evaluation ResNet-18 on selected samples:')
        acc=evaluate(WidarResNet(len(tr.class_names), dropout=0.2),tr,te,indices,
                     device,a.eval_epochs,a.batch_size,seed=run_seed)
        scores.append(acc)
        print(f'Run {run+1} best accuracy: {acc:.2f}%')
    print(f'\n{a.method} / {a.selection_space}, IPC={a.ipc}: '
          f'{np.mean(scores):.2f}% ± {np.std(scores, ddof=1) if len(scores)>1 else 0.0:.2f}% '
          f'(mean ± std over {len(scores)} runs)')

if __name__=='__main__':
    main()
