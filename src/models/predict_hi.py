"""
Inference script: generate per-file CNN health-index predictions.

Loads hi_cnn.pth and feature_scaler.npz, runs the trained HIConvNet on all
984 files, and writes two outputs so CNN and proxy HI can be compared:

  results/metrics/health_index.csv       -- CNN predictions (becomes the
                                            authoritative HI used by Stage 3)
  results/metrics/health_index_proxy.csv -- Gaussian-smoothed proxy RMS HI

Edge files (first/last T_WINDOW//2 = 15) cannot form a full sliding window.
They are padded by repeating the nearest valid window prediction (start: copy
file 15's prediction; end: copy file 968's prediction). The number of padded
files is printed so callers know exactly what happened.

Comparison plot
---------------
results/figures/cnn_vs_proxy_hi.png overlays both curves with the train/val
boundary marked.  MAE (overall and val-region) is printed to stdout.
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from src.models.cnn_health_index import (
    FEATURE_COLS, N_FEATURES, T_WINDOW,
    build_per_file_features, build_windows,
    FeatureScaler, HIConvNet,
)
from src.models.train_cnn import compute_proxy_hi

_AUG_CSV    = _PROJECT_ROOT / 'results' / 'metrics' / 'peak_features_augmented.csv'
_RAW_DIR    = _PROJECT_ROOT / 'data' / 'raw' / '2nd_test'
_MODEL_PATH = _PROJECT_ROOT / 'results' / 'metrics' / 'hi_cnn.pth'
_SCALER_NPZ = _PROJECT_ROOT / 'results' / 'metrics' / 'feature_scaler.npz'
_HI_CSV     = _PROJECT_ROOT / 'results' / 'metrics' / 'health_index.csv'
_PROXY_CSV  = _PROJECT_ROOT / 'results' / 'metrics' / 'health_index_proxy.csv'
_FIG_DIR    = _PROJECT_ROOT / 'results' / 'figures'

TRAIN_N      = 787   # must match train_cnn.py; used only for the comparison plot
HI_CHANNEL   = 0
HI_SIGMA     = 3.0   # Gaussian sigma for proxy HI (must match training)
SMOOTH_SIGMA = 2.0   # light post-inference smooth on CNN output


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def predict_cnn_hi(
    aug_csv:     Path = _AUG_CSV,
    raw_dir:     Path = _RAW_DIR,
    model_path:  Path = _MODEL_PATH,
    scaler_path: Path = _SCALER_NPZ,
    out_hi_csv:  Path = _HI_CSV,
    out_proxy_csv: Path = _PROXY_CSV,
    fig_dir:     Path = _FIG_DIR,
) -> pd.DataFrame:

    # ------------------------------------------------------------------
    # 1. Per-file feature matrix (all 984 files, sorted chronologically)
    # ------------------------------------------------------------------
    print(f"Loading features from {aug_csv.name} ...")
    per_file_df = build_per_file_features(aug_csv)
    file_names  = per_file_df['file'].tolist()
    n_files     = len(file_names)
    X_all       = per_file_df[FEATURE_COLS].values.astype(np.float32)  # (984, 12)
    print(f"  {n_files} files, {N_FEATURES} features each")

    # ------------------------------------------------------------------
    # 2. Load scaler (fitted on train files only — must NOT refit here)
    # ------------------------------------------------------------------
    scaler = FeatureScaler.load(scaler_path)
    print(f"Scaler loaded from {scaler_path.name}")

    # ------------------------------------------------------------------
    # 3. Load model weights
    # ------------------------------------------------------------------
    model = HIConvNet(n_features=N_FEATURES)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded from {model_path.name}  ({n_params:,} parameters)")

    # ------------------------------------------------------------------
    # 4. Build ALL windows (full-run inference, no train/val split)
    # ------------------------------------------------------------------
    dummy_labels = np.zeros(n_files, dtype=np.float32)
    X_windows, _ = build_windows(X_all, dummy_labels, T_WINDOW)
    n_windows    = X_windows.shape[0]  # 984 - 30 = 954
    print(f"Inference windows: {n_windows}  (T={T_WINDOW})")

    # ------------------------------------------------------------------
    # 5. Scale (using train-fitted scaler, same as during training)
    # ------------------------------------------------------------------
    X_scaled = scaler.transform(X_windows)

    # ------------------------------------------------------------------
    # 6. Inference (eval mode, no gradient)
    # ------------------------------------------------------------------
    with torch.no_grad():
        preds = model(torch.from_numpy(X_scaled)).numpy()   # (954,)

    print(f"CNN raw prediction range: [{preds.min():.4f}, {preds.max():.4f}]")

    # ------------------------------------------------------------------
    # 7. Map window predictions to per-file HI via center indexing
    # ------------------------------------------------------------------
    half_T = T_WINDOW // 2   # 15
    # window i -> center file = i + half_T
    # covered: files half_T .. half_T + n_windows - 1  = 15 .. 968
    hi_per_file             = np.full(n_files, np.nan, dtype=np.float32)
    hi_per_file[half_T: half_T + n_windows] = preds

    # Edge padding
    n_pad_start = half_T                              # 15 files (indices 0-14)
    n_pad_end   = n_files - (half_T + n_windows)      # 15 files (indices 969-983)
    hi_per_file[:n_pad_start]  = hi_per_file[n_pad_start]   # repeat file-15 prediction
    hi_per_file[n_files - n_pad_end:] = hi_per_file[n_files - n_pad_end - 1]

    print(f"Edge padding: {n_pad_start} files at start (copy file {n_pad_start}), "
          f"{n_pad_end} files at end (copy file {n_files - n_pad_end - 1})")

    # ------------------------------------------------------------------
    # 8. Light Gaussian smooth (reduces window-to-window jitter, sigma << 3)
    # ------------------------------------------------------------------
    if SMOOTH_SIGMA > 0:
        hi_per_file = gaussian_filter1d(hi_per_file.astype(float), sigma=SMOOTH_SIGMA)
    hi_per_file = np.clip(hi_per_file, 0.0, 1.0).astype(float)
    print(f"CNN HI range (post-smooth): [{hi_per_file.min():.4f}, {hi_per_file.max():.4f}]")

    # ------------------------------------------------------------------
    # 9. Compute proxy HI (for comparison and for health_index_proxy.csv)
    # ------------------------------------------------------------------
    print("Computing proxy RMS HI for comparison ...")
    proxy_hi = compute_proxy_hi(raw_dir, file_names, HI_CHANNEL, HI_SIGMA).astype(float)
    print(f"Proxy HI range: [{proxy_hi.min():.4f}, {proxy_hi.max():.4f}]")

    # ------------------------------------------------------------------
    # 10. Comparison statistics (print before saving)
    # ------------------------------------------------------------------
    mae_all  = float(np.mean(np.abs(hi_per_file - proxy_hi)))
    mae_train = float(np.mean(np.abs(hi_per_file[:TRAIN_N] - proxy_hi[:TRAIN_N])))
    mae_val   = float(np.mean(np.abs(hi_per_file[TRAIN_N:] - proxy_hi[TRAIN_N:])))

    print(f"\n=== CNN vs Proxy HI Comparison ===")
    print(f"  MAE overall           : {mae_all:.4f}")
    print(f"  MAE train region 0-{TRAIN_N-1} : {mae_train:.4f}")
    print(f"  MAE val region {TRAIN_N}-983   : {mae_val:.4f}")
    print()

    # Spot-check val region at key file indices
    check_idxs = [787, 850, 900, 950, 960, 965, 968, 970, 975, 980, 983]
    print(f"  {'File':>6}  {'CNN HI':>8}  {'Proxy HI':>9}  {'Diff':>7}")
    print(f"  {'------':>6}  {'------':>8}  {'--------':>9}  {'----':>7}")
    for i in check_idxs:
        diff = hi_per_file[i] - proxy_hi[i]
        flag = " <-- onset" if proxy_hi[i] < 0.7 else (" <-- fail" if proxy_hi[i] < 0.5 else "")
        print(f"  {i:>6}  {hi_per_file[i]:>8.4f}  {proxy_hi[i]:>9.4f}  {diff:>+7.4f}{flag}")

    divergence_flag = mae_val > 0.25
    if divergence_flag:
        print(f"\n  *** DIVERGENCE WARNING: val-region MAE = {mae_val:.4f} > 0.25 ***")
        print(f"  *** CNN and proxy HI differ significantly in the validation region. ***")
        print(f"  *** Review cnn_vs_proxy_hi.png before using CNN predictions.       ***")
    else:
        print(f"\n  CNN and proxy HI agree within threshold (val MAE = {mae_val:.4f} <= 0.25).")

    # ------------------------------------------------------------------
    # 11. Comparison plot
    # ------------------------------------------------------------------
    fig_dir.mkdir(parents=True, exist_ok=True)
    _plot_comparison(
        hi_per_file, proxy_hi, file_names, fig_dir,
        mae_train=mae_train, mae_val=mae_val,
    )

    # ------------------------------------------------------------------
    # 12. Save outputs
    # ------------------------------------------------------------------
    hi_df = pd.DataFrame({'file': file_names, 'health_index': hi_per_file})
    hi_df.to_csv(out_hi_csv, index=False)
    print(f"\nCNN HI saved    -> {out_hi_csv}  ({len(hi_df)} rows)")

    proxy_df = pd.DataFrame({'file': file_names, 'health_index': proxy_hi})
    proxy_df.to_csv(out_proxy_csv, index=False)
    print(f"Proxy HI saved  -> {out_proxy_csv}  ({len(proxy_df)} rows)")

    if divergence_flag:
        raise RuntimeError(
            f"CNN vs proxy HI divergence too large (val MAE={mae_val:.4f}). "
            "Review cnn_vs_proxy_hi.png — consider retraining before using CNN predictions."
        )

    return hi_df


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------

def _plot_comparison(
    cnn_hi:    np.ndarray,
    proxy_hi:  np.ndarray,
    file_names: list[str],
    fig_dir:   Path,
    mae_train: float,
    mae_val:   float,
) -> None:
    n = len(cnn_hi)
    t = np.arange(n)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                             gridspec_kw={'height_ratios': [3, 1]})

    # -- top panel: overlaid HI curves --
    ax = axes[0]
    ax.plot(t, proxy_hi, label='Proxy RMS HI', color='steelblue',
            linewidth=1.0, alpha=0.85)
    ax.plot(t, cnn_hi,   label='CNN-predicted HI', color='darkorange',
            linewidth=1.2, alpha=0.9)
    ax.axvline(TRAIN_N - 0.5, color='black', linestyle='--', linewidth=1.2,
               label=f'Train/Val boundary (file {TRAIN_N-1}/{TRAIN_N})')
    ax.axhline(0.70, color='green',  linestyle=':', linewidth=0.9,
               label='Onset threshold (0.70)')
    ax.axhline(0.50, color='red',    linestyle=':', linewidth=0.9,
               label='Failure threshold (0.50)')

    # shade regions
    ax.axvspan(0,       TRAIN_N - 0.5, alpha=0.04, color='steelblue',  label='Training region')
    ax.axvspan(TRAIN_N - 0.5, n,       alpha=0.04, color='darkorange', label='Validation region')

    ax.set_ylabel('Health Index (HI)')
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=8, loc='lower left')
    ax.set_title(
        f'CNN vs Proxy HI  |  Train MAE={mae_train:.4f}  Val MAE={mae_val:.4f}\n'
        f'(CNN uses physics features only; proxy uses RMS — no overlap in the feature set)',
        fontsize=10,
    )

    # -- bottom panel: difference --
    ax2 = axes[1]
    diff = cnn_hi - proxy_hi
    ax2.plot(t, diff, color='purple', linewidth=0.8, alpha=0.8)
    ax2.axhline(0,   color='black', linewidth=0.5)
    ax2.axhline(+0.1, color='gray', linestyle='--', linewidth=0.6, alpha=0.6)
    ax2.axhline(-0.1, color='gray', linestyle='--', linewidth=0.6, alpha=0.6)
    ax2.axvline(TRAIN_N - 0.5, color='black', linestyle='--', linewidth=1.0)
    ax2.set_ylabel('CNN - Proxy')
    ax2.set_xlabel('File index (chronological)')
    ax2.set_ylim(-0.5, 0.5)

    plt.tight_layout()
    out = fig_dir / 'cnn_vs_proxy_hi.png'
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot -> {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate CNN health-index predictions from hi_cnn.pth.')
    parser.add_argument('--aug_csv',   default=str(_AUG_CSV))
    parser.add_argument('--raw_dir',   default=str(_RAW_DIR))
    parser.add_argument('--model',     default=str(_MODEL_PATH))
    parser.add_argument('--scaler',    default=str(_SCALER_NPZ))
    args = parser.parse_args()
    predict_cnn_hi(
        aug_csv    = Path(args.aug_csv),
        raw_dir    = Path(args.raw_dir),
        model_path = Path(args.model),
        scaler_path= Path(args.scaler),
    )


if __name__ == '__main__':
    main()
