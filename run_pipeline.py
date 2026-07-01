"""
Single entrypoint for the NASA IMS Bearing RUL pipeline.

Stages
------
1. Feature extraction  — FrSST peak-window features + physics augmentation
                         → results/metrics/peak_features_augmented.csv
2. CNN health index    — train HI regression model
                         → results/metrics/health_index.csv
3. RUL estimation      — derive RUL from HI trend
                         → results/figures/rul_*.png

Run from the project root (ts_frsst_project/):
    python run_pipeline.py
    python run_pipeline.py --skip-extraction    # reuse existing peak_features_augmented.csv
    python run_pipeline.py --skip-training      # reuse existing health_index.csv
    python run_pipeline.py --method cwt         # use plain CWT instead of FrSST
    python run_pipeline.py --channel 3          # use bearing channel 3 (0-indexed)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR      = PROJECT_ROOT / 'data' / 'raw' / '2nd_test'
FEATURES_CSV  = PROJECT_ROOT / 'results' / 'metrics' / 'peak_features.csv'
AUG_CSV       = PROJECT_ROOT / 'results' / 'metrics' / 'peak_features_augmented.csv'
HI_CSV        = PROJECT_ROOT / 'results' / 'metrics' / 'health_index.csv'
FIGURES_DIR   = PROJECT_ROOT / 'results' / 'figures'
METRICS_DIR   = PROJECT_ROOT / 'results' / 'metrics'

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------

def stage1_extract_features(method: str = 'frsst', channel: int = 0) -> None:
    print('\n=== Stage 1a: FrSST Feature Extraction ===')
    from src.features.extract_features import process_folder
    process_folder(
        input_folder=DATA_DIR,
        out_csv=FEATURES_CSV,
        fs=20480,
        channel=channel,
        method=method,
        top_n=3,
    )
    print('\n=== Stage 1b: Physics Feature Augmentation ===')
    from src.features.physics_features import add_physics_features
    add_physics_features(
        peak_csv=FEATURES_CSV,
        raw_dir=DATA_DIR,
        out_csv=AUG_CSV,
    )


def stage2_train_cnn() -> None:
    print('\n=== Stage 2a: CNN Training ===')
    from src.models.train_cnn import train
    train(
        aug_csv     = AUG_CSV,
        raw_dir     = DATA_DIR,
        metrics_dir = METRICS_DIR,
        fig_dir     = FIGURES_DIR,
    )
    # NOTE: health_index.csv is populated from proxy RMS HI (health_index_proxy.csv).
    # CNN inference (predict_hi.py) is available as a diagnostic tool, but
    # the trained model does not generalise to the degradation period because
    # all training files are in the healthy regime -- see README Model Validation.
    import shutil
    shutil.copy(str(METRICS_DIR / 'health_index_proxy.csv'),
                str(METRICS_DIR / 'health_index.csv'))
    print(f'health_index.csv <- proxy RMS HI  '
          f'(CNN val MAE=0.14; model does not capture degradation -- see README)')


def stage3_estimate_rul() -> None:
    print('\n=== Stage 3: RUL Estimation ===')
    from src.models.rul_estimation import estimate_rul
    estimate_rul(hi_csv=HI_CSV, fig_dir=FIGURES_DIR)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='NASA IMS Bearing RUL pipeline.')
    parser.add_argument('--skip-extraction', action='store_true',
                        help='Skip Stage 1 (reuse existing peak_features.csv)')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip Stage 2 (CNN training)')
    parser.add_argument('--method', choices=['frsst', 'sst', 'cwt'], default='frsst',
                        help='TF method for feature extraction (default: frsst)')
    parser.add_argument('--channel', type=int, default=0,
                        help='Bearing channel index, 0-indexed (default: 0)')
    args = parser.parse_args()

    if not args.skip_extraction:
        stage1_extract_features(method=args.method, channel=args.channel)
    else:
        print('\n[Stage 1 skipped - using existing peak_features_augmented.csv]')

    if not args.skip_training:
        stage2_train_cnn()
    else:
        print('\n[Stage 2 skipped]')

    stage3_estimate_rul()


if __name__ == '__main__':
    main()
