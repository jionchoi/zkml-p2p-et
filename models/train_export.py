
from __future__ import annotations

import argparse
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from helpers import SEQ_LEN, PRED_LEN, column_splits, train_site
from Linear import Linear
from NLinear import NLinear
from DLinear import DLinear

MODELS = {"linear": Linear, "nlinear": NLinear, "dlinear": DLinear}

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO / "data" / "electricity.csv"


# Sites are CSV column names: "0".."319" plus "OT" (321 in total). They stay
# strings so the OT target is handled like any other site.
def parse_sites(spec: str):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part and part.replace("-", "").isdigit():
            a, b = part.split("-")
            out.extend(str(i) for i in range(int(a), int(b) + 1))
        else:
            out.append(part)
    return out


def site_label(site):
    """Filesystem label: zero-pad numeric sites for tidy sorting, keep OT as is."""
    return f"{int(site):03d}" if str(site).isdigit() else str(site)


def export_site(model_key, site, col, out_root, epochs, patience):
    splits, scaler = column_splits(col, SEQ_LEN, PRED_LEN)

    torch.manual_seed(0)
    model = MODELS[model_key]()
    val_mse = train_site(model, splits, epochs=epochs, patience=patience)
    model.eval()

    # First test window is the private input; its true future is the accuracy target.
    Xte, Yte = splits["test"]
    x0 = torch.from_numpy(Xte[:1])                  # [1, seq_len, 1]
    with torch.no_grad():
        pred = model(x0).numpy().ravel()
    true = Yte[0].ravel()
    mse_float = float(np.mean((pred - true) ** 2))

    site_dir = out_root / model_key / f"site_{site_label(site)}"
    site_dir.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model, x0, str(site_dir / "network.onnx"),
        opset_version=15, input_names=["input"], output_names=["output"],
        dynamo=False,   # fixed batch=1; EZKL's parser rejects symbolic dims
    )

    json.dump({"input_data": [x0.numpy().ravel().tolist()]},
              open(site_dir / "input.json", "w"))

    params = sum(p.numel() for p in model.parameters())
    json.dump({
        "model": model_key, "site": site, "seq_len": SEQ_LEN, "pred_len": PRED_LEN,
        "params": params, "val_mse": round(val_mse, 6), "mse_float": round(mse_float, 6),
        "true_future": true.tolist(),
        "scaler_mean": float(scaler.mean_[0]), "scaler_scale": float(scaler.scale_[0]),
    }, open(site_dir / "meta.json", "w"), indent=2)

    return params, mse_float


# Worker globals: the dataset is loaded once per process (not once per site).
_DATA = _OUT = _EPOCHS = _PATIENCE = None


def _init(csv_path, out_root, epochs, patience):
    global _DATA, _OUT, _EPOCHS, _PATIENCE
    torch.set_num_threads(1)            # one model per core; no intra-op contention
    df = pd.read_csv(csv_path)
    _DATA = {c: df[c].values.astype(np.float64)
             for c in df.columns if c != "date"}
    _OUT, _EPOCHS, _PATIENCE = Path(out_root), epochs, patience


def _job(spec):
    model_key, site = spec
    params, mse = export_site(model_key, site, _DATA[site], _OUT, _EPOCHS, _PATIENCE)
    return model_key, site, params, mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV), help="path to electricity.csv")
    ap.add_argument("--models", default="linear,nlinear,dlinear")
    ap.add_argument("--sites", default="0,1,2",
                    help="'all' (321 sites), a range '0-319', or a list '0,1,OT'")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6, help="parallel processes")
    ap.add_argument("--out", default=str(REPO / "models"))
    args = ap.parse_args()

    csv_path = Path(args.csv)
    assert csv_path.exists(), f"dataset not found: {csv_path}"
    out_root = Path(args.out)
    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]

    all_cols = [c for c in pd.read_csv(csv_path, nrows=0).columns if c != "date"]
    sites = all_cols if args.sites.strip() == "all" else parse_sites(args.sites)
    unknown = [s for s in sites if s not in all_cols]
    assert not unknown, f"unknown site columns: {unknown} (have {len(all_cols)})"
    jobs = [(m, s) for m in model_keys for s in sites]

    print(f"{len(model_keys)} models x {len(sites)} sites = {len(jobs)} models "
          f"-> {out_root} ({args.workers} workers)")
    init_args = (csv_path, out_root, args.epochs, args.patience)
    done = 0
    if args.workers > 1:
        with Pool(args.workers, initializer=_init, initargs=init_args) as pool:
            for mk, site, params, mse in pool.imap_unordered(_job, jobs):
                done += 1
                print(f"  [{done}/{len(jobs)}] {mk} site {site:>3}: "
                      f"params={params} mse_float={mse:.4f}", flush=True)
    else:
        _init(*init_args)
        for spec in jobs:
            mk, site, params, mse = _job(spec)
            done += 1
            print(f"  [{done}/{len(jobs)}] {mk} site {site:>3}: "
                  f"params={params} mse_float={mse:.4f}", flush=True)


if __name__ == "__main__":
    main()
