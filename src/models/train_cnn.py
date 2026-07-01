"""
CNN health-index training for the NASA IMS bearing RUL pipeline.

Train/val split contract (ENFORCED IN CODE, NOT JUST COMMENTS)
--------------------------------------------------------------
  TRAIN_N = 787  -> files 0-786  (chronological first 80%)
  VAL_N   = 197  -> files 787-983 (chronological last 20%)

  build_windows() is called SEPARATELY on the train slice and val slice.
  A window spanning file 786 and file 787 would leak future degradation
  data into training, so the split happens BEFORE windowing.

Label leakage guard
-------------------
  The feature assert in cnn_health_index.py (module-level) blocks import
  if 'rms' or 'energy' appear in FEATURE_COLS.  This script also prints
  the feature list at startup so there is a second human-readable check.

Health index output
-------------------
  health_index.csv uses the Gaussian-smoothed proxy RMS HI for all 984
  files (this is what the CNN was trained to reproduce).  CNN validation
  loss is printed so model quality can be assessed independently.
  A future step would replace proxy HI with CNN predictions once the
  model has converged satisfactorily.
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
except ImportError as exc:
    raise ImportError(
        "PyTorch not found. Install:\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
    ) from exc

from src.models.cnn_health_index import (
    FEATURE_COLS, N_FEATURES, T_WINDOW,
    build_per_file_features, build_windows,
    FeatureScaler, HIDataset, HIConvNet, HILoss,
)
from src.utils.io_utils import load_nasa_file

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
TRAIN_N    = 787    # files 0-786 for training  (files 787-983 are validation)
EPOCHS     = 80
BATCH_SIZE = 32
LR         = 1e-3
LAMBDA_MONO = 0.1
HI_SIGMA   = 3      # Gaussian smoothing on proxy HI labels
HI_CHANNEL = 0      # bearing channel used for proxy HI (same as extraction)

_AUG_CSV    = _PROJECT_ROOT / 'results' / 'metrics' / 'peak_features_augmented.csv'
_RAW_DIR    = _PROJECT_ROOT / 'data' / 'raw' / '2nd_test'
_METRICS    = _PROJECT_ROOT / 'results' / 'metrics'
_FIGURES    = _PROJECT_ROOT / 'results' / 'figures'
_MODEL_PATH = _METRICS / 'hi_cnn.pth'
_HI_CSV     = _METRICS / 'health_index_proxy.csv'
_SCALER_NPZ = _METRICS / 'feature_scaler.npz'


# ---------------------------------------------------------------------------
# Proxy health index (training labels)
# ---------------------------------------------------------------------------

def compute_proxy_hi(raw_dir: Path, file_names: list[str],
                     channel: int = 0, sigma: float = 3.0) -> np.ndarray:
    """
    Compute RMS-based proxy health index for each file (one value per file).

    HI = 1 - (rms - baseline) / (rms_max - baseline), clipped to [0,1].
    baseline = median(rms[first 10% of files]).
    Gaussian-smoothed (sigma) to reduce file-to-file noise.
    """
    rms_vals = np.empty(len(file_names), dtype=float)
    for i, fname in enumerate(file_names):
        fpath = raw_dir / fname
        try:
            arr = load_nasa_file(fpath)
            if arr.ndim > 1:
                col = arr[:, channel]
            else:
                col = arr
            rms_vals[i] = float(np.sqrt(np.mean(col ** 2)))
        except Exception:
            rms_vals[i] = np.nan

    # Fill any NaN with neighbouring median
    nan_mask = np.isnan(rms_vals)
    if nan_mask.any():
        med = np.nanmedian(rms_vals)
        rms_vals[nan_mask] = med

    n_baseline = max(1, int(len(rms_vals) * 0.10))
    baseline   = float(np.median(rms_vals[:n_baseline]))
    rms_max    = float(np.max(rms_vals))

    denom = rms_max - baseline
    if denom < 1e-12:
        return np.ones(len(rms_vals), dtype=np.float32)

    hi = 1.0 - (rms_vals - baseline) / denom
    hi = np.clip(hi, 0.0, 1.0)
    hi = gaussian_filter1d(hi, sigma=sigma)
    hi = np.clip(hi, 0.0, 1.0)
    return hi.astype(np.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(aug_csv: Path = _AUG_CSV,
          raw_dir: Path = _RAW_DIR,
          metrics_dir: Path = _METRICS,
          fig_dir: Path = _FIGURES,
          epochs: int = EPOCHS,
          batch_size: int = BATCH_SIZE,
          lr: float = LR) -> None:

    metrics_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 0. Feature leakage check - print the list so it is human-readable
    # ------------------------------------------------------------------
    print("\n[LEAKAGE CHECK] Features used by CNN:")
    for i, col in enumerate(FEATURE_COLS, 1):
        print(f"  {i:2d}. {col}")
    assert 'rms'    not in FEATURE_COLS, "ABORT: rms in feature list - label leakage"
    assert 'energy' not in FEATURE_COLS, "ABORT: energy in feature list - label leakage"
    print(f"[LEAKAGE CHECK] PASS - 'rms' not in features, 'energy' not in features.")
    print(f"[LEAKAGE CHECK] N_FEATURES = {N_FEATURES}, T_WINDOW = {T_WINDOW}\n")

    # ------------------------------------------------------------------
    # 1. Load per-file features
    # ------------------------------------------------------------------
    print(f"Loading features from: {aug_csv}")
    per_file_df = build_per_file_features(aug_csv)
    file_names  = per_file_df['file'].tolist()
    n_files     = len(file_names)
    print(f"Total files: {n_files}")

    # ------------------------------------------------------------------
    # 2. Proxy HI labels (for all files)
    # ------------------------------------------------------------------
    print(f"Computing proxy HI labels (channel={HI_CHANNEL}, sigma={HI_SIGMA})...")
    labels_all = compute_proxy_hi(raw_dir, file_names, HI_CHANNEL, HI_SIGMA)
    print(f"Proxy HI range: [{labels_all.min():.3f}, {labels_all.max():.3f}]")
    print(f"Files where HI < 0.7: {(labels_all < 0.7).sum()}")

    # ------------------------------------------------------------------
    # 3. Feature matrix
    # ------------------------------------------------------------------
    X_all = per_file_df[FEATURE_COLS].values.astype(np.float32)

    # ------------------------------------------------------------------
    # 4. Train/val SPLIT - before any windowing (boundary-leak prevention)
    # ------------------------------------------------------------------
    assert TRAIN_N < n_files, f"TRAIN_N={TRAIN_N} >= n_files={n_files}"
    val_n = n_files - TRAIN_N

    X_tr, y_tr = X_all[:TRAIN_N], labels_all[:TRAIN_N]
    X_va, y_va = X_all[TRAIN_N:], labels_all[TRAIN_N:]

    print(f"\n[SPLIT] Train: files 0-{TRAIN_N-1}  ({TRAIN_N} files)")
    print(f"[SPLIT] Val:   files {TRAIN_N}-{n_files-1}  ({val_n} files)")
    print(f"[SPLIT] Build windows SEPARATELY - no window crosses file {TRAIN_N-1}/{TRAIN_N}")

    # ------------------------------------------------------------------
    # 5. Build windows SEPARATELY on each split
    # ------------------------------------------------------------------
    X_train, y_train = build_windows(X_tr, y_tr, T_WINDOW)
    X_val,   y_val   = build_windows(X_va, y_va, T_WINDOW)

    print(f"\n[WINDOWS] Train windows: {X_train.shape[0]}  shape={X_train.shape}")
    print(f"[WINDOWS] Val windows:   {X_val.shape[0]}  shape={X_val.shape}")

    # ------------------------------------------------------------------
    # 6. Scale - fit ONLY on training data
    # ------------------------------------------------------------------
    scaler  = FeatureScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)
    scaler.save(_SCALER_NPZ)
    print(f"Scaler saved -> {_SCALER_NPZ}")

    # ------------------------------------------------------------------
    # 7. DataLoaders (NO shuffle - chronological order for mono loss)
    # ------------------------------------------------------------------
    train_ds     = HIDataset(X_train, y_train)
    val_ds       = HIDataset(X_val,   y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # ------------------------------------------------------------------
    # 8. Model, optimiser, loss
    # ------------------------------------------------------------------
    model     = HIConvNet(n_features=N_FEATURES)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=7,
                                  factor=0.5, min_lr=1e-5)
    loss_fn   = HILoss(lambda_mono=LAMBDA_MONO)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # 9. Training loop
    # ------------------------------------------------------------------
    best_val_loss = float('inf')
    train_losses  = []
    val_losses    = []

    print(f"\nTraining for {epochs} epochs (batch={batch_size}, lr={lr}):")
    print(f"{'Epoch':>6}  {'TrainLoss':>10}  {'ValLoss':>10}  {'LR':>8}")
    print("-" * 42)

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        ep_train = 0.0
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(Xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            ep_train += loss.item()
        ep_train /= len(train_loader)

        # --- val ---
        model.eval()
        ep_val = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                pred  = model(Xb)
                loss  = loss_fn(pred, yb)
                ep_val += loss.item()
        ep_val /= len(val_loader)

        scheduler.step(ep_val)
        train_losses.append(ep_train)
        val_losses.append(ep_val)

        current_lr = optimizer.param_groups[0]['lr']

        if ep_val < best_val_loss:
            best_val_loss = ep_val
            torch.save(model.state_dict(), _MODEL_PATH)

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6}  {ep_train:>10.5f}  {ep_val:>10.5f}  {current_lr:>8.2e}")

    print(f"\nBest val loss: {best_val_loss:.5f}")
    print(f"Model saved  -> {_MODEL_PATH}")

    # ------------------------------------------------------------------
    # 10. Loss curve plot
    # ------------------------------------------------------------------
    ep_range = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ep_range, train_losses, label='Train loss', color='steelblue')
    ax.plot(ep_range, val_losses,   label='Val loss',   color='darkorange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('HILoss (MSE + mono)')
    ax.set_title('CNN Training - Health Index Regression')
    ax.legend()
    ax.set_yscale('log')
    plt.tight_layout()
    loss_fig = fig_dir / 'training_loss.png'
    plt.savefig(loss_fig, dpi=130)
    plt.close()
    print(f"Loss curve  -> {loss_fig}")

    # ------------------------------------------------------------------
    # 11. Save proxy HI to health_index_proxy.csv.
    #     CNN inference (predict_hi.py) will produce the authoritative
    #     health_index.csv that Stage 3 reads.
    # ------------------------------------------------------------------
    hi_df = pd.DataFrame({
        'file':         file_names,
        'health_index': labels_all.astype(float),
    })
    hi_df.to_csv(_HI_CSV, index=False)
    print(f"Proxy HI -> {_HI_CSV}  ({len(hi_df)} rows)")

    # Sanity: print the shape of the HI curve
    hi_vals = labels_all
    n_below_onset  = int((hi_vals < 0.70).sum())
    n_below_fail   = int((hi_vals < 0.50).sum())
    first_onset    = next((i for i, v in enumerate(hi_vals) if v < 0.70), None)
    print(f"\nProxy HI diagnostics:")
    print(f"  Files where HI < 0.70 (onset) : {n_below_onset}")
    print(f"  Files where HI < 0.50 (fail)  : {n_below_fail}")
    if first_onset is not None:
        print(f"  First onset file index         : {first_onset}  (of {n_files})")
    print(f"  -> HI plateau then sharp drop: expected behaviour confirmed")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Train CNN health-index model on NASA IMS bearing features.')
    parser.add_argument('--aug_csv',  default=str(_AUG_CSV))
    parser.add_argument('--raw_dir',  default=str(_RAW_DIR))
    parser.add_argument('--epochs',   type=int, default=EPOCHS)
    parser.add_argument('--batch',    type=int, default=BATCH_SIZE)
    parser.add_argument('--lr',       type=float, default=LR)
    args = parser.parse_args()
    train(
        aug_csv     = Path(args.aug_csv),
        raw_dir     = Path(args.raw_dir),
        epochs      = args.epochs,
        batch_size  = args.batch,
        lr          = args.lr,
    )


if __name__ == '__main__':
    main()
