"""
One-time prebuild: generate dark-themed FrSST scalogram PNGs for a sparse
set of files and cache them under dashboard/backend/cache/scalograms/.

Read-only with respect to data/ and src/ -- only reads raw NASA files and
imports existing extraction functions. Writes only inside dashboard/.

Sampled indices: every 20th file (0, 20, 40, ... 980) plus every file in
the degradation tail 960-983, so the ScalogramViewer crossfades smoothly
through the part of the run that tells the story.

Run once before starting the frontend:
    python precompute_scalograms.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_utils import load_nasa_file
from src.features.peak_detection import envelope, adaptive_peak_windows
from src.features.tf_extraction import FRSST_AVAILABLE, compute_sst_tf, extract_top_freqs_from_tf

RAW_DIR  = PROJECT_ROOT / 'data' / 'raw' / '2nd_test'
OUT_DIR  = Path(__file__).resolve().parent / 'cache' / 'scalograms'
FS       = 20480
CHANNEL  = 0

# Dark theme to match the dashboard
_BG       = '#0a0a0f'
_FG       = '#cbd5e1'
_CMAP     = 'inferno'
_ACCENT   = '#22d3ee'   # cyan, matches "healthy" accent color


def select_indices(n_files: int) -> list[int]:
    sparse = set(range(0, n_files, 20))
    tail = set(range(max(0, n_files - 24), n_files))
    return sorted(sparse | tail)


def render_one(fpath: Path, idx: int) -> bool:
    arr = load_nasa_file(fpath)
    sig = arr[:, CHANNEL] if arr.ndim > 1 else arr
    sig = sig - sig.mean()

    env = envelope(sig, win_len=max(3, int(0.005 * FS)))
    windows = adaptive_peak_windows(
        env, FS,
        height_factor=0.6,
        distance_s=0.01,
        rel_height_for_width=0.5,
        min_win_samples=512,
        max_win_samples=4096,
    )
    if not windows:
        print(f'  [SKIP] {fpath.name}: no peak windows found')
        return False

    pidx, lidx, ridx = windows[0]
    wsig = sig[lidx: ridx + 1]
    tf_mag, freqs = compute_sst_tf(wsig, FS)
    top = extract_top_freqs_from_tf(tf_mag, freqs, top_n=3)

    fig, ax = plt.subplots(figsize=(6, 4), facecolor=_BG)
    ax.set_facecolor(_BG)
    extent = [0, len(wsig) / FS, float(freqs[-1]), float(freqs[0])]
    ax.imshow(tf_mag, aspect='auto', extent=extent, cmap=_CMAP)
    for fr, _ in top:
        ax.axhline(y=fr, color=_ACCENT, linestyle='--', linewidth=0.7, alpha=0.85)
    ax.set_xlabel('Time [s]', color=_FG, fontsize=9)
    ax.set_ylabel('Frequency [Hz]', color=_FG, fontsize=9)
    ax.set_title(f'{fpath.name}  (file {idx})', color=_FG, fontsize=9)
    ax.tick_params(colors=_FG, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color('#334155')

    plt.tight_layout()
    out_path = OUT_DIR / f'file_{idx:04d}.png'
    plt.savefig(out_path, dpi=120, facecolor=_BG)
    plt.close(fig)
    return True


def main() -> None:
    if not FRSST_AVAILABLE:
        raise RuntimeError(
            'ssqueezepy not available -- FrSST scalograms cannot be generated. '
            'Install ssqueezepy before running this script (no silent CWT fallback).'
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in RAW_DIR.iterdir() if p.is_file())
    n_files = len(files)
    indices = select_indices(n_files)
    print(f'Precomputing {len(indices)} scalograms (of {n_files} files) -> {OUT_DIR}')

    n_ok = 0
    for idx in indices:
        ok = render_one(files[idx], idx)
        n_ok += int(ok)
        print(f'  [{idx:4d}] {files[idx].name} -> {"OK" if ok else "SKIPPED"}')

    print(f'\nDone: {n_ok}/{len(indices)} scalograms written to {OUT_DIR}')


if __name__ == '__main__':
    main()
