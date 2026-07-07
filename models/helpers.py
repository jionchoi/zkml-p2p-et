"""Shared constants and per-site data/training helpers for the LTSF-Linear models.

Split out of the original ltsf.py so each model variant lives in its own file
(models/Linear.py, NLinear.py, DLinear.py). Anything used by more than one of
those files -- or by train_export.py -- lives here: the sequence lengths, the
per-site train/val/test split (matching LTSF-Linear's Dataset_Custom, 70/10/20,
StandardScaler fit on train), and the training loop.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

SEQ_LEN = 192
PRED_LEN = 24


def _windows(series, seq_len, pred_len):
    """All (X, Y) sliding windows of a 1-D series -> arrays [n, seq_len/pred_len, 1]."""
    n = len(series) - seq_len - pred_len + 1
    X = np.stack([series[i:i + seq_len] for i in range(n)])
    Y = np.stack([series[i + seq_len:i + seq_len + pred_len] for i in range(n)])
    return X[..., None].astype(np.float32), Y[..., None].astype(np.float32)


def column_splits(col, seq_len=SEQ_LEN, pred_len=PRED_LEN):
    """Replicates LTSF-Linear Dataset_Custom for one site's 1-D series.

    `col` is that site's raw values (any length). Returns train/val/test (X, Y)
    windows plus the scaler, with scaling fit on the train split only — exactly
    as in the paper's loader. Split off from `site_splits` so the dataset can be
    read once and reused across all sites.
    """
    col = np.asarray(col, dtype=np.float64).reshape(-1, 1)
    n = len(col)
    num_train = int(n * 0.7)
    num_test = int(n * 0.2)
    num_vali = n - num_train - num_test

    border1s = [0, num_train - seq_len, n - num_test - seq_len]
    border2s = [num_train, num_train + num_vali, n]

    scaler = StandardScaler().fit(col[border1s[0]:border2s[0]])
    data = scaler.transform(col).ravel()

    splits = {}
    for name, b1, b2 in zip(("train", "val", "test"), border1s, border2s):
        splits[name] = _windows(data[b1:b2], seq_len, pred_len)
    return splits, scaler


def site_splits(csv_path, site, seq_len=SEQ_LEN, pred_len=PRED_LEN):
    """Convenience wrapper: read one site column from the CSV and split it."""
    col = pd.read_csv(csv_path, usecols=[str(site)])[str(site)].values
    return column_splits(col, seq_len, pred_len)


def train_site(model, splits, epochs=100, patience=3, lr=1e-3, batch_size=16):
    """Train one site model with early stopping on validation MSE."""
    Xtr, Ytr = (torch.from_numpy(a) for a in splits["train"])
    Xva, Yva = (torch.from_numpy(a) for a in splits["val"])

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best, best_state, waited = float("inf"), None, 0
    n = len(Xtr)

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), Ytr[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xva), Yva).item()
        if vloss < best - 1e-7:
            best, best_state, waited = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            waited += 1
            if waited >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best
