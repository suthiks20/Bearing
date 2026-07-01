"""
CNN health-index model and feature specification for the NASA IMS bearing pipeline.

Feature set (12 features — leakage audit)
------------------------------------------
Included:
  f1_hz, f2_hz, f3_hz  — FrSST dominant frequencies (spectral content)
  a1, a2, a3            — FrSST relative magnitudes  (spectral weight)
  kurtosis              — excess kurtosis (4th-moment shape, NOT amplitude)
  entropy               — Shannon entropy of amplitude histogram (shape, NOT amplitude)
  bpfo_match_count      — windows matching a BPFO harmonic (Rexnord ZA-2115)
  bpfi_match_count      — same, BPFI
  bsf_match_count       — same, BSF
  ftf_match_count       — same, FTF

Excluded by design (label leakage):
  rms    — proxy HI label = f(rms); a CNN feature that directly determines
            the label gives near-zero training loss while learning nothing.
  energy — sum(x²) = n * rms²; same leakage path as rms.

The exclusion is enforced by module-level asserts (below FEATURE_COLS).
The module will not import if rms or energy are ever added to that list.

Architecture
------------
Input: (batch, T_WINDOW=30, N_FEATURES=12)
  — 30 consecutive per-file feature vectors fed as a temporal sequence.
  — Conv1D expects channels-first; the forward() method transposes internally.

Conv1D(12->32, k=5, p=2) + BN + ReLU  — local 50-min slope patterns
Conv1D(32->32, k=3, p=1) + BN + ReLU  — fine-grain transition sharpening
GlobalAveragePool over time             — collapse temporal dim
Dense(32) + ReLU + Dropout(0.3)
Dense(1)  + Sigmoid                    — HI ∈ [0, 1]

~5,600 trainable parameters (intentionally small for 787-file training set).

Loss
----
HILoss = MSE + lambda_mono * soft_monotonicity_penalty
Monotonicity penalty assumes batches are fed in chronological order
(no shuffle in DataLoader during training).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for cnn_health_index.py.\n"
        "Install it with:  pip install torch --index-url https://download.pytorch.org/whl/cpu"
    ) from exc

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Authoritative feature specification
# ---------------------------------------------------------------------------

FEATURE_COLS: list[str] = [
    # FrSST dominant frequencies — spectral fingerprint of fault excitation.
    # Normalized to [0,1] by dividing by Nyquist (10240 Hz) before scaling.
    'f1_hz', 'f2_hz', 'f3_hz',
    # Corresponding relative magnitudes from the FrSST TF map.
    'a1', 'a2', 'a3',
    # Signal shape statistics — capture impulsiveness and distribution spread
    # without encoding signal amplitude (which is the label's information source).
    'kurtosis',  # excess kurtosis (Fisher); Gaussian = 0; fault impacts >> 0
    'entropy',   # Shannon entropy [bits] of 50-bin amplitude histogram
    # Physics-informed fault-frequency match counts (Rexnord ZA-2115 @ 2000 RPM).
    # Max-aggregated per file: presence of any matching window is the signal.
    'bpfo_match_count',
    'bpfi_match_count',
    'bsf_match_count',
    'ftf_match_count',
]

# --- Label-leakage guard: enforced at import time, not just in comments. ---
# proxy HI = 1 - (rms - baseline) / (rms_max - baseline), so rms and energy
# (= n * rms²) are monotonically related to the label and must never be features.
assert 'rms'    not in FEATURE_COLS, \
    "LEAKAGE: 'rms' is in FEATURE_COLS — proxy HI is derived from rms. Remove it."
assert 'energy' not in FEATURE_COLS, \
    "LEAKAGE: 'energy' is in FEATURE_COLS — energy = n*rms² shares the leakage path."

N_FEATURES: int = len(FEATURE_COLS)   # 12
T_WINDOW:   int = 30                  # consecutive per-file vectors per sample
FS_NYQUIST: float = 10240.0           # Hz — used to pre-normalize freq features

# Aggregation rule per feature when collapsing window-level rows -> one per file.
# max for fault-match and kurtosis: a single impulsive/matching window is the signal.
# mean (nanmean) for everything else: stable central tendency.
AGGS: dict[str, str] = {
    'f1_hz': 'mean', 'f2_hz': 'mean', 'f3_hz': 'mean',
    'a1':    'mean', 'a2':    'mean', 'a3':    'mean',
    'kurtosis':         'max',
    'entropy':          'mean',
    'bpfo_match_count': 'max',
    'bpfi_match_count': 'max',
    'bsf_match_count':  'max',
    'ftf_match_count':  'max',
}
# 'mean' is used (not 'nanmean') because pandas groupby.agg silently skips NaN
# by default — equivalent to np.nanmean. Using 'nanmean' as a string raises
# AttributeError in pandas.

assert set(AGGS.keys()) == set(FEATURE_COLS), \
    "AGGS keys must exactly match FEATURE_COLS — update both together."


# ---------------------------------------------------------------------------
# Data preparation utilities
# ---------------------------------------------------------------------------

def build_per_file_features(aug_csv: Path | str) -> pd.DataFrame:
    """
    Aggregate window-level features from peak_features_augmented.csv to one
    row per file, sorted chronologically by filename.

    Returns a DataFrame with columns = FEATURE_COLS, index = file name,
    sorted so that row 0 is the earliest file.
    """
    df = pd.read_csv(aug_csv)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"peak_features_augmented.csv is missing columns: {missing}\n"
            f"Run add_physics_features() first."
        )

    # Pre-normalise frequency features before aggregation so scale is consistent.
    for col in ('f1_hz', 'f2_hz', 'f3_hz'):
        df[col] = df[col] / FS_NYQUIST

    per_file = (
        df.groupby('file', sort=False)
          .agg(**{col: pd.NamedAgg(column=col, aggfunc=agg)
                  for col, agg in AGGS.items()})
          .reset_index()
          .sort_values('file')
          .reset_index(drop=True)
    )

    # Fill any remaining NaN (e.g. files where all windows had no f3 peak)
    # with the per-column median computed across all files.
    for col in FEATURE_COLS:
        med = per_file[col].median()
        per_file[col] = per_file[col].fillna(med if not np.isnan(med) else 0.0)

    return per_file


class FeatureScaler:
    """
    StandardScaler wrapper that operates on numpy arrays.
    Fit on training data, apply to both train and val.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_:  np.ndarray | None = None

    def fit(self, X: np.ndarray) -> 'FeatureScaler':
        # X shape: (n_samples, T_WINDOW, N_FEATURES)
        flat = X.reshape(-1, X.shape[-1])
        self.mean_ = flat.mean(axis=0)
        self.std_  = flat.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0    # avoid zero-division on constant cols
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: Path | str) -> None:
        np.savez(path, mean=self.mean_, std=self.std_)

    @classmethod
    def load(cls, path: Path | str) -> 'FeatureScaler':
        data = np.load(path)
        sc = cls()
        sc.mean_ = data['mean']
        sc.std_  = data['std']
        return sc


def build_windows(
    per_file_X:      np.ndarray,   # (n_files, N_FEATURES)
    per_file_labels: np.ndarray,   # (n_files,)  — proxy HI, one per file
    t_window:        int = T_WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window of length t_window over the per-file feature matrix.

    Label for window i is the proxy HI at the CENTER of that window
    (file index i + t_window // 2).

    Returns
    -------
    X : (n_windows, t_window, N_FEATURES)
    y : (n_windows,)
    """
    n_files  = per_file_X.shape[0]
    n_windows = n_files - t_window
    if n_windows <= 0:
        raise ValueError(
            f"Not enough files ({n_files}) to build windows of size {t_window}."
        )

    X = np.stack([per_file_X[i: i + t_window] for i in range(n_windows)])
    y = per_file_labels[t_window // 2: t_window // 2 + n_windows]
    return X.astype(np.float32), y.astype(np.float32)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class HIDataset(Dataset):
    """Wraps (X, y) numpy arrays for DataLoader consumption."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X)   # (n, T, F)
        self.y = torch.from_numpy(y)   # (n,)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class HIConvNet(nn.Module):
    """
    1D-CNN health-index regressor.

    Input  : (batch, T=30, F=12)   — temporal window of per-file feature vectors
    Output : (batch,)               — predicted HI ∈ [0, 1]

    The time dimension is the convolution axis; features are the channels.
    Internally, the tensor is transposed to (batch, F, T) for Conv1D,
    then transposed back after pooling.

    Architecture
    ------------
    Conv1D(12->32, k=5, pad=2) + BN + ReLU
    Conv1D(32->32, k=3, pad=1) + BN + ReLU
    GlobalAvgPool (mean over T)
    Linear(32->32) + ReLU + Dropout(0.3)
    Linear(32->1)  + Sigmoid
    """

    def __init__(self, n_features: int = N_FEATURES, dropout: float = 0.3) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(32)
        self.fc1   = nn.Linear(32, 32)
        self.drop  = nn.Dropout(dropout)
        self.fc2   = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, F) — transpose to (batch, F, T) for Conv1D
        x = x.transpose(1, 2)            # -> (batch, F, T)
        x = F.relu(self.bn1(self.conv1(x)))   # -> (batch, 32, T)
        x = F.relu(self.bn2(self.conv2(x)))   # -> (batch, 32, T)
        x = x.mean(dim=2)                # GlobalAvgPool -> (batch, 32)
        x = F.relu(self.fc1(x))          # -> (batch, 32)
        x = self.drop(x)
        x = torch.sigmoid(self.fc2(x))   # -> (batch, 1)
        return x.squeeze(1)              # -> (batch,)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

class HILoss(nn.Module):
    """
    MSE + soft monotonicity penalty.

    Monotonicity term penalises consecutive predictions in the batch where
    HI increases (pred[t+1] > pred[t]).  This requires that training batches
    are fed in chronological order — set shuffle=False in DataLoader.

    lambda_mono=0.1 is the design value; reduce toward 0 if train/val loss
    diverge, since the penalty fights the gradient on recovery predictions.
    """

    def __init__(self, lambda_mono: float = 0.1) -> None:
        super().__init__()
        self.lambda_mono = lambda_mono

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, target)
        if pred.shape[0] > 1 and self.lambda_mono > 0:
            delta = pred[1:] - pred[:-1]             # positive = HI rising (penalise)
            mono  = F.relu(delta).pow(2).mean()
        else:
            mono = torch.zeros(1, device=pred.device)
        return mse + self.lambda_mono * mono
