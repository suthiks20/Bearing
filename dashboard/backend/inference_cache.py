"""
Loads pipeline artifacts once at process startup and holds them in memory.

Read-only with respect to src/, results/, data/ -- nothing in this module
writes back into those directories. The CNN health-index curve is not
persisted anywhere on disk (health_index.csv was reverted to the proxy RMS
HI after the divergence investigation -- see README "Model Validation"), so
it is recomputed here via the existing predict_cnn_hi() inference path and
cached in memory for the lifetime of the server.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

METRICS_DIR = PROJECT_ROOT / 'results' / 'metrics'
FIGURES_DIR = PROJECT_ROOT / 'results' / 'figures'
RAW_DIR     = PROJECT_ROOT / 'data' / 'raw' / '2nd_test'
AUG_CSV     = METRICS_DIR / 'peak_features_augmented.csv'

from src.models.cnn_health_index import FEATURE_COLS, build_per_file_features


class PipelineCache:
    """Holds every dataset the dashboard needs, loaded once at startup."""

    def __init__(self) -> None:
        self.health_index_df: pd.DataFrame | None = None   # file, hi_proxy, hi_cnn
        self.rul_df: pd.DataFrame | None = None
        self.bpfo_trend_df: pd.DataFrame | None = None
        self.per_file_features_df: pd.DataFrame | None = None
        self.file_names: list[str] = []
        self.cnn_mae_val: float | None = None
        self.cnn_mae_train: float | None = None

    def load(self) -> None:
        print('[cache] Loading rul_estimates.csv ...')
        self.rul_df = pd.read_csv(METRICS_DIR / 'rul_estimates.csv')

        print('[cache] Loading bpfo_trend_stats.csv ...')
        self.bpfo_trend_df = pd.read_csv(METRICS_DIR / 'bpfo_trend_stats.csv')

        print('[cache] Loading peak_features_augmented.csv for per-file aggregates ...')
        self.per_file_features_df = build_per_file_features(AUG_CSV)
        self.file_names = self.per_file_features_df['file'].tolist()

        print('[cache] Loading proxy health_index_proxy.csv ...')
        proxy_df = pd.read_csv(METRICS_DIR / 'health_index_proxy.csv')

        print('[cache] Running CNN inference once (hi_cnn.pth, in-memory only) ...')
        hi_cnn = self._run_cnn_inference()

        self.health_index_df = pd.DataFrame({
            'file':     proxy_df['file'],
            'hi_proxy': proxy_df['health_index'].astype(float),
            'hi_cnn':   hi_cnn.astype(float),
        })

        mae_all = float(np.mean(np.abs(self.health_index_df['hi_cnn'] - self.health_index_df['hi_proxy'])))
        train_n = 787
        self.cnn_mae_train = float(np.mean(np.abs(
            self.health_index_df['hi_cnn'][:train_n] - self.health_index_df['hi_proxy'][:train_n])))
        self.cnn_mae_val = float(np.mean(np.abs(
            self.health_index_df['hi_cnn'][train_n:] - self.health_index_df['hi_proxy'][train_n:])))
        print(f'[cache] CNN vs proxy MAE: overall={mae_all:.4f} '
              f'train={self.cnn_mae_train:.4f} val={self.cnn_mae_val:.4f}')
        print('[cache] Ready.')

    def _run_cnn_inference(self) -> np.ndarray:
        """Reproduces predict_hi.py's inference path without writing any files."""
        import torch
        from scipy.ndimage import gaussian_filter1d
        from src.models.cnn_health_index import (
            N_FEATURES, T_WINDOW, build_windows, FeatureScaler, HIConvNet,
        )

        X_all = self.per_file_features_df[FEATURE_COLS].values.astype(np.float32)
        n_files = len(self.file_names)

        scaler = FeatureScaler.load(METRICS_DIR / 'feature_scaler.npz')
        model = HIConvNet(n_features=N_FEATURES)
        model.load_state_dict(torch.load(METRICS_DIR / 'hi_cnn.pth', map_location='cpu', weights_only=True))
        model.eval()

        dummy_labels = np.zeros(n_files, dtype=np.float32)
        X_windows, _ = build_windows(X_all, dummy_labels, T_WINDOW)
        n_windows = X_windows.shape[0]

        X_scaled = scaler.transform(X_windows)
        with torch.no_grad():
            preds = model(torch.from_numpy(X_scaled)).numpy()

        half_T = T_WINDOW // 2
        hi_per_file = np.full(n_files, np.nan, dtype=np.float32)
        hi_per_file[half_T: half_T + n_windows] = preds
        n_pad_start = half_T
        n_pad_end = n_files - (half_T + n_windows)
        hi_per_file[:n_pad_start] = hi_per_file[n_pad_start]
        hi_per_file[n_files - n_pad_end:] = hi_per_file[n_files - n_pad_end - 1]

        hi_per_file = gaussian_filter1d(hi_per_file.astype(float), sigma=2.0)
        return np.clip(hi_per_file, 0.0, 1.0)


cache = PipelineCache()
