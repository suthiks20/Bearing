"""
Peak-window feature extraction pipeline.

Reads raw NASA IMS bearing files, detects vibration peak windows,
computes FrSST (or CWT fallback) time-frequency maps, and writes
dominant-frequency features to a CSV.

Usage (standalone):
    python -m src.features.extract_features [--input DIR] [--out CSV] ...

Or imported and called via run_pipeline.py.
"""
from __future__ import annotations

import sys
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# Allow running as `python src/features/extract_features.py` from any CWD
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.io_utils import load_nasa_file
from src.features.peak_detection import envelope, adaptive_peak_windows
from src.features.tf_extraction import (
    FRSST_AVAILABLE,
    compute_cwt_tf,
    compute_sst_tf,
    extract_top_freqs_from_tf,
)

_DEFAULT_INPUT  = _PROJECT_ROOT / 'data' / 'raw' / '2nd_test'
_DEFAULT_OUTPUT = _PROJECT_ROOT / 'results' / 'metrics' / 'peak_features.csv'


def process_folder(
    input_folder: Path | str = _DEFAULT_INPUT,
    out_csv:      Path | str = _DEFAULT_OUTPUT,
    fs:           float = 20480,
    channel:      int   = 0,
    method:       str   = 'frsst',
    top_n:        int   = 3,
    debug_plots:  bool  = False,
    plots_dir:    Path | str | None = None,
) -> Path:
    input_folder = Path(input_folder)
    out_csv      = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Degrade gracefully but NEVER silently
    if method in ('frsst', 'sst') and not FRSST_AVAILABLE:
        warnings.warn(
            f"[extract_features] method='{method}' requested but ssqueezepy "
            "is not installed. Degrading to CWT — THIS IS NOT FrSST. "
            "Install ssqueezepy to use the intended method.",
            RuntimeWarning,
            stacklevel=2,
        )
        method = 'cwt'

    if debug_plots:
        plots_dir = Path(plots_dir) if plots_dir else \
            _PROJECT_ROOT / 'results' / 'figures' / 'debug'
        plots_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in input_folder.iterdir() if p.is_file())
    rows: list[dict] = []

    for fpath in tqdm(files, desc='Extracting features'):
        try:
            arr = load_nasa_file(fpath)
        except Exception as exc:
            print(f'  [SKIP] {fpath.name}: {exc}')
            continue

        sig = arr[:, channel] if arr.ndim > 1 else arr
        sig = sig - sig.mean()

        env = envelope(sig, win_len=max(3, int(0.005 * fs)))
        windows = adaptive_peak_windows(
            env, fs,
            height_factor=0.6,
            distance_s=0.01,
            rel_height_for_width=0.5,
            min_win_samples=512,
            max_win_samples=4096,
        )
        if not windows:
            continue

        for (pidx, lidx, ridx) in windows:
            wsig = sig[lidx: ridx + 1]

            if method in ('frsst', 'sst'):
                tf_mag, freqs = compute_sst_tf(wsig, fs)
            else:
                tf_mag, freqs = compute_cwt_tf(wsig, fs)

            top = extract_top_freqs_from_tf(tf_mag, freqs, top_n=top_n)
            peak_time_s = float(pidx) / fs

            row: dict = {
                'file':        fpath.name,
                'channel':     int(channel),
                'peak_time_s': peak_time_s,
                'win_left':    int(lidx),
                'win_right':   int(ridx),
            }
            for i in range(top_n):
                if i < len(top):
                    row[f'f{i+1}_hz'] = float(top[i][0])
                    row[f'a{i+1}']    = float(top[i][1])
                else:
                    row[f'f{i+1}_hz'] = np.nan
                    row[f'a{i+1}']    = np.nan
            rows.append(row)

            if debug_plots:
                _save_debug_plot(tf_mag, freqs, wsig, top, fpath.name,
                                 channel, pidx, peak_time_s, fs, plots_dir)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        print(f'Saved {len(df)} rows -> {out_csv}')
    else:
        print('No peak windows found — check input data and parameters.')

    return out_csv


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_debug_plot(tf_mag, freqs, wsig, top, fname, channel,
                     pidx, peak_time_s, fs, plots_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    extent = [0, len(wsig) / fs, float(freqs[-1]), float(freqs[0])]
    axes[0].imshow(tf_mag, aspect='auto', extent=extent)
    axes[0].set_title(f'{fname} ch{channel} @{peak_time_s:.3f}s')
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Frequency [Hz]')
    for fr, _ in top:
        axes[0].axhline(y=fr, color='r', linestyle='--', linewidth=0.8)

    avg = np.mean(tf_mag, axis=1)
    axes[1].plot(freqs, avg)
    axes[1].scatter([t[0] for t in top],
                    [t[1] * float(np.max(avg)) for t in top],
                    color='r', zorder=5)
    axes[1].set_xlabel('Frequency [Hz]')
    axes[1].set_ylabel('Mean magnitude')

    plt.tight_layout()
    out = plots_dir / f'plot_{fname}_ch{channel}_p{pidx}.png'
    plt.savefig(out, dpi=80)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract FrSST/CWT peak-window features from NASA IMS data.')
    parser.add_argument('--input',  '-i', default=str(_DEFAULT_INPUT))
    parser.add_argument('--out',    '-o', default=str(_DEFAULT_OUTPUT))
    parser.add_argument('--fs',          type=float, default=20480)
    parser.add_argument('--channel',     type=int,   default=0)
    parser.add_argument('--method',
                        choices=['frsst', 'sst', 'cwt'], default='frsst')
    parser.add_argument('--top_n',       type=int,   default=3)
    parser.add_argument('--debug_plots', action='store_true')
    args = parser.parse_args()

    process_folder(
        input_folder=args.input,
        out_csv=args.out,
        fs=args.fs,
        channel=args.channel,
        method=args.method,
        top_n=args.top_n,
        debug_plots=args.debug_plots,
    )
