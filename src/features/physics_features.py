"""
Physics-based per-window feature extraction for NASA IMS bearing data.

Two feature classes
-------------------
1. Time-domain statistics (computed from the raw window signal):
     energy   = sum(x[n]^2)
     entropy  = Shannon entropy of normalized amplitude histogram
     rms      = sqrt(mean(x[n]^2))
     kurtosis = excess kurtosis (Fisher, Gaussian=0; impulsive faults >> 0)

2. Bearing fault frequency matching (FrSST dominant freqs vs. known fault harmonics):
     Fault frequencies are derived from exact Rexnord ZA-2115 geometry.
     Any FrSST-extracted dominant frequency that lands within
         tol = max(TOL_MIN_HZ, TOL_PCT * harmonic_freq)
     of a BPFO / BPFI / BSF / FTF harmonic is flagged.

Bearing geometry — Rexnord ZA-2115, double-row
-----------------------------------------------
  N_ROLLERS   = 16        rollers per row
  PITCH_DIA   = 2.815     inches (pitch diameter)
  ROLLER_DIA  = 0.331     inches (roller/ball diameter)
  CONTACT_ANG = 15.17     degrees
  SHAFT_RPM   = 2000      RPM
  FS          = 20480     Hz

Fault frequency formulas
------------------------
  beta  = (d/D) * cos(alpha)        d = roller dia, D = pitch dia
  FTF   = f_r/2      * (1 - beta)   cage / train frequency
  BPFO  = (N/2)*f_r  * (1 - beta)   ball pass frequency, outer race
  BPFI  = (N/2)*f_r  * (1 + beta)   ball pass frequency, inner race
  BSF   = (D/2d)*f_r * (1 - beta²)  ball spin frequency
  where f_r = shaft_rpm / 60

Computed values (from the geometry above):
  FTF  =  14.7752 Hz
  BPFO = 236.4035 Hz
  BPFI = 296.9299 Hz
  BSF  = 139.9167 Hz
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as scipy_kurtosis

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.io_utils import load_nasa_file

# ---------------------------------------------------------------------------
# Bearing geometry constants — do NOT change these without updating the
# docstring above and re-verifying against the NASA IMS dataset documentation.
# ---------------------------------------------------------------------------
_N_ROLLERS   = 16
_PITCH_DIA   = 2.815    # inches
_ROLLER_DIA  = 0.331    # inches
_CONTACT_ANG = 15.17    # degrees
_SHAFT_RPM   = 2000
_FS          = 20480

# Matching tolerance: tol = max(TOL_MIN_HZ, TOL_PCT * harmonic_freq)
# At 3 % the tolerance grows with harmonic number, matching the FrSST
# frequency resolution which also grows with frequency (log-spaced bins).
TOL_PCT     = 0.03   # 3 % of the harmonic frequency
TOL_MIN_HZ  = 20.0   # absolute floor (covers FrSST bin-width at low freqs)
N_HARMONICS = 20     # highest harmonic to check per fault type


# ---------------------------------------------------------------------------
# Fault frequency computation
# ---------------------------------------------------------------------------

def compute_fault_frequencies(
    shaft_rpm:        float = _SHAFT_RPM,
    n_rollers:        int   = _N_ROLLERS,
    pitch_dia:        float = _PITCH_DIA,
    roller_dia:       float = _ROLLER_DIA,
    contact_angle_deg: float = _CONTACT_ANG,
) -> dict[str, float]:
    """
    Return {'FTF': ..., 'BPFO': ..., 'BPFI': ..., 'BSF': ...} in Hz.
    All geometry parameters default to Rexnord ZA-2115 / 2000 RPM.
    """
    f_r  = shaft_rpm / 60.0
    beta = (roller_dia / pitch_dia) * np.cos(np.radians(contact_angle_deg))
    return {
        'ftf':  f_r / 2.0 * (1.0 - beta),
        'bpfo': (n_rollers / 2.0) * f_r * (1.0 - beta),
        'bpfi': (n_rollers / 2.0) * f_r * (1.0 + beta),
        'bsf':  (pitch_dia / (2.0 * roller_dia)) * f_r * (1.0 - beta ** 2),
    }


# ---------------------------------------------------------------------------
# Time-domain feature extraction
# ---------------------------------------------------------------------------

def window_physics_features(signal: np.ndarray) -> dict[str, float]:
    """
    Compute time-domain statistics for a single window of raw vibration data.

    Returns
    -------
    dict with keys: energy, entropy, rms, kurtosis
    """
    x = np.asarray(signal, dtype=float)

    energy = float(np.sum(x ** 2))
    rms    = float(np.sqrt(np.mean(x ** 2)))

    # Excess kurtosis (Fisher, normal = 0).  Healthy bearings ≈ 0;
    # impulsive fault signals typically >> 3.
    kurt = float(scipy_kurtosis(x, fisher=True, bias=False))

    # Shannon entropy of the amplitude histogram (50 bins).
    # Spread-out (non-Gaussian) distributions yield higher entropy.
    hist, _ = np.histogram(x, bins=50)
    hist    = hist.astype(float)
    total   = hist.sum()
    if total > 0:
        p    = hist[hist > 0] / total
        entr = float(-np.sum(p * np.log2(p)))
    else:
        entr = 0.0

    return {'energy': energy, 'entropy': entr, 'rms': rms, 'kurtosis': kurt}


# ---------------------------------------------------------------------------
# Fault frequency matching
# ---------------------------------------------------------------------------

def match_dominant_freqs(
    top_freqs_hz: list[float],
    fault_freqs:  dict[str, float],
    fs:           float = _FS,
    n_harmonics:  int   = N_HARMONICS,
    tol_pct:      float = TOL_PCT,
    tol_min_hz:   float = TOL_MIN_HZ,
) -> dict:
    """
    Check each observed frequency against every harmonic of every fault type.

    Returns
    -------
    dict with:
      '<TYPE>_match_count'  — int, number of observed freqs hitting this type
      'any_fault_match'     — bool
      'fault_match_detail'  — str, human-readable list of all matches
    """
    counts  = {t: 0 for t in fault_freqs}
    details = []

    nyquist = fs / 2.0
    for obs in top_freqs_hz:
        if obs is None or (isinstance(obs, float) and np.isnan(obs)):
            continue
        for fault_type, ff in fault_freqs.items():
            for n in range(1, n_harmonics + 1):
                harm_hz = n * ff
                if harm_hz > nyquist:
                    break
                tol  = max(tol_min_hz, tol_pct * harm_hz)
                diff = abs(obs - harm_hz)
                if diff <= tol:
                    counts[fault_type] += 1
                    details.append(
                        f'{obs:.1f}Hz~{fault_type}x{n}'
                        f'({harm_hz:.1f}Hz,d={diff:.1f}Hz)'
                    )

    return {
        **{f'{t}_match_count': counts[t] for t in counts},
        'any_fault_match':    any(v > 0 for v in counts.values()),
        'fault_match_detail': '; '.join(details) if details else '',
    }


# ---------------------------------------------------------------------------
# Main augmentation function
# ---------------------------------------------------------------------------

def add_physics_features(
    peak_csv:    Path | str,
    raw_dir:     Path | str,
    out_csv:     Path | str | None = None,
    fault_freqs: dict[str, float] | None = None,
    n_harmonics: int   = N_HARMONICS,
    tol_pct:     float = TOL_PCT,
    tol_min_hz:  float = TOL_MIN_HZ,
) -> pd.DataFrame:
    """
    Read peak_features.csv, add time-domain + fault-match columns, write CSV.

    Re-reads raw signal files (grouping by filename to load each file once)
    to extract window signals for time-domain computation.  Fault-frequency
    matching uses the dominant frequencies already stored in the CSV
    (f1_hz, f2_hz, f3_hz, …).

    Parameters
    ----------
    peak_csv    : path to peak_features.csv produced by extract_features.py
    raw_dir     : directory containing the raw NASA IMS files
    out_csv     : output path; defaults to peak_csv parent / 'peak_features_augmented.csv'
    fault_freqs : override computed fault frequencies (default: Rexnord ZA-2115)
    """
    peak_csv = Path(peak_csv)
    raw_dir  = Path(raw_dir)
    out_csv  = Path(out_csv) if out_csv else \
               peak_csv.parent / 'peak_features_augmented.csv'

    if fault_freqs is None:
        fault_freqs = compute_fault_frequencies()

    df = pd.read_csv(peak_csv)

    # Identify dominant-frequency columns (f1_hz, f2_hz, ...)
    freq_cols = sorted(c for c in df.columns if c.startswith('f') and c.endswith('_hz'))

    # Accumulate new columns row-by-row, loading each raw file once per group
    new_rows: list[dict] = []
    grouped = df.groupby(['file', 'channel'])
    n_groups = len(grouped)

    for g_idx, ((fname, channel), grp) in enumerate(grouped):
        raw_path = raw_dir / fname
        try:
            arr = load_nasa_file(raw_path)
        except Exception as exc:
            print(f'  [SKIP] {fname}: {exc}')
            for _, row in grp.iterrows():
                new_rows.append({'energy': np.nan, 'entropy': np.nan,
                                 'rms': np.nan, 'kurtosis': np.nan,
                                 **{f'{t}_match_count': 0 for t in fault_freqs},
                                 'any_fault_match': False,
                                 'fault_match_detail': 'load_error'})
            continue

        sig = arr[:, channel] if arr.ndim > 1 else arr
        sig = sig - sig.mean()

        if (g_idx + 1) % 100 == 0 or g_idx == n_groups - 1:
            print(f'  [{g_idx+1}/{n_groups}] {fname}')

        for _, row in grp.iterrows():
            l = int(row['win_left'])
            r = int(row['win_right'])
            wsig = sig[l: r + 1]

            phys = window_physics_features(wsig)

            top_freqs = [row[c] for c in freq_cols
                         if not (isinstance(row[c], float) and np.isnan(row[c]))]
            match = match_dominant_freqs(
                top_freqs, fault_freqs,
                n_harmonics=n_harmonics,
                tol_pct=tol_pct,
                tol_min_hz=tol_min_hz,
            )
            new_rows.append({**phys, **match})

    aug = df.assign(**pd.DataFrame(new_rows).to_dict(orient='list'))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    aug.to_csv(out_csv, index=False)
    print(f'\nSaved {len(aug)} rows -> {out_csv}')
    return aug


# ---------------------------------------------------------------------------
# Explicit verification: Step 3 match demo
# ---------------------------------------------------------------------------

def verify_match_on_file(
    raw_file:    Path | str,
    channel:     int   = 0,
    fs:          float = _FS,
    top_n:       int   = 5,
    n_harmonics: int   = N_HARMONICS,
    tol_pct:     float = TOL_PCT,
    tol_min_hz:  float = TOL_MIN_HZ,
) -> None:
    """
    Load a raw file, run FrSST on the first peak window, compute physics
    features, and print a full match table showing exactly which fault
    frequencies the dominant FrSST peaks correspond to.
    """
    from src.features.peak_detection import envelope, adaptive_peak_windows
    from src.features.tf_extraction import (
        FRSST_AVAILABLE, compute_sst_tf, compute_cwt_tf,
        extract_top_freqs_from_tf,
    )

    raw_file    = Path(raw_file)
    fault_freqs = compute_fault_frequencies()

    print('=' * 68)
    print(f'Physics Feature Verification')
    print(f'File    : {raw_file.name}')
    print(f'Channel : {channel}')
    print(f'Method  : {"FrSST (ssqueezepy)" if FRSST_AVAILABLE else "CWT (fallback — ssqueezepy absent)"}')
    print('=' * 68)

    # Load & detect peaks
    arr = load_nasa_file(raw_file)
    sig = arr[:, channel] if arr.ndim > 1 else arr
    sig = sig - sig.mean()

    env     = envelope(sig, win_len=max(3, int(0.005 * fs)))
    windows = adaptive_peak_windows(
        env, fs,
        height_factor=0.6, distance_s=0.01,
        rel_height_for_width=0.5,
        min_win_samples=512, max_win_samples=4096,
    )
    print(f'\nPeak windows detected : {len(windows)}')
    if not windows:
        print('ERROR: no peak windows found.')
        return

    pidx, lidx, ridx = windows[0]
    wsig = sig[lidx: ridx + 1]
    print(f'Using window 0        : samples {lidx}-{ridx} '
          f'({1000*len(wsig)/fs:.1f} ms)')

    # Time-frequency extraction
    if FRSST_AVAILABLE:
        tf_mag, freqs = compute_sst_tf(wsig, fs)
    else:
        tf_mag, freqs = compute_cwt_tf(wsig, fs)
    top = extract_top_freqs_from_tf(tf_mag, freqs, top_n=top_n, exclusion_hz=10.0)
    top_hz = [f for f, _ in top]

    # Time-domain features
    phys = window_physics_features(wsig)

    # ---- Print fault frequencies ----------------------------------------
    print()
    print('Rexnord ZA-2115 fault frequencies (2000 RPM):')
    for name, ff in fault_freqs.items():
        print(f'  {name:4s} = {ff:9.4f} Hz')

    # ---- Print dominant freqs -------------------------------------------
    print()
    print(f'Top {top_n} FrSST dominant frequencies:')
    for i, (f, a) in enumerate(top):
        print(f'  [{i+1}] {f:9.2f} Hz   rel_mag = {a:.4f}')

    # ---- Print time-domain features ------------------------------------
    print()
    print('Time-domain features (window 0):')
    print(f'  energy   = {phys["energy"]:12.4f}')
    print(f'  rms      = {phys["rms"]:12.6f}')
    print(f'  kurtosis = {phys["kurtosis"]:12.4f}  '
          f'(excess; healthy~0, impulsive fault >> 0)')
    print(f'  entropy  = {phys["entropy"]:12.4f}  bits (Shannon, 50-bin histogram)')

    # ---- Fault frequency match table -----------------------------------
    print()
    print(f'Fault frequency matching '
          f'(tol = max({tol_min_hz:.0f} Hz, {tol_pct*100:.0f}% of harmonic)):')
    print()

    nyquist  = fs / 2.0
    any_hit  = False
    hit_rows = []

    for obs, rel_mag in top:
        obs_hits = []
        for fault_type, ff in fault_freqs.items():
            for n in range(1, n_harmonics + 1):
                harm_hz = n * ff
                if harm_hz > nyquist:
                    break
                tol  = max(tol_min_hz, tol_pct * harm_hz)
                diff = abs(obs - harm_hz)
                if diff <= tol:
                    obs_hits.append((fault_type, n, harm_hz, diff, tol))
                    any_hit = True

        hit_rows.append((obs, rel_mag, obs_hits))

    # Print in a compact aligned table
    header = (f'  {"Observed":>10}  {"rel_mag":>8}  '
              f'{"Fault":>4}  {"N":>3}  {"Harmonic Hz":>12}  '
              f'{"diff Hz":>8}  {"tol Hz":>8}  Match')
    print(header)
    print('  ' + '-' * (len(header) - 2))

    for obs, rel_mag, hits in hit_rows:
        if not hits:
            print(f'  {obs:>10.2f}  {rel_mag:>8.4f}  '
                  f'{"--":>4}  {"--":>3}  {"--":>12}  '
                  f'{"--":>8}  {"--":>8}  no match')
        else:
            first = True
            for (ft, n, harm, diff, tol) in hits:
                obs_str    = f'{obs:.2f}' if first else ''
                mag_str    = f'{rel_mag:.4f}' if first else ''
                first      = False
                print(f'  {obs_str:>10}  {mag_str:>8}  '
                      f'{ft:>4}  {n:>3}  {harm:>12.2f}  '
                      f'{diff:>8.2f}  {tol:>8.2f}  MATCH')

    print()
    # Summary
    counts = {t: 0 for t in fault_freqs}
    for _, _, hits in hit_rows:
        for ft, *_ in hits:
            counts[ft] += 1
    print('Summary — match count per fault type across all dominant freqs:')
    for ft, cnt in counts.items():
        bar = '#' * cnt + ('  (dominant)' if cnt == max(counts.values()) and cnt > 0 else '')
        print(f'  {ft:4s}: {cnt}  {bar}')

    if not any_hit:
        print('  No fault frequency matches found.')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Add physics features to peak_features.csv '
                    'or verify fault-frequency matches on a single file.')
    sub = parser.add_subparsers(dest='cmd')

    # sub-command: augment
    aug_p = sub.add_parser('augment', help='Augment peak_features.csv')
    aug_p.add_argument('--peak_csv', default=str(
        _PROJECT_ROOT / 'results' / 'metrics' / 'peak_features.csv'))
    aug_p.add_argument('--raw_dir', default=str(
        _PROJECT_ROOT / 'data' / 'raw' / '2nd_test'))
    aug_p.add_argument('--out_csv', default=None)
    aug_p.add_argument('--n_harmonics', type=int,   default=N_HARMONICS)
    aug_p.add_argument('--tol_pct',     type=float, default=TOL_PCT)
    aug_p.add_argument('--tol_min_hz',  type=float, default=TOL_MIN_HZ)

    # sub-command: verify
    ver_p = sub.add_parser('verify', help='Verify matches on one raw file')
    ver_p.add_argument('--file', required=True, help='Raw bearing file path')
    ver_p.add_argument('--channel',     type=int,   default=0)
    ver_p.add_argument('--top_n',       type=int,   default=5)
    ver_p.add_argument('--n_harmonics', type=int,   default=N_HARMONICS)
    ver_p.add_argument('--tol_pct',     type=float, default=TOL_PCT)
    ver_p.add_argument('--tol_min_hz',  type=float, default=TOL_MIN_HZ)

    args = parser.parse_args()

    if args.cmd == 'augment':
        add_physics_features(
            peak_csv=args.peak_csv,
            raw_dir=args.raw_dir,
            out_csv=args.out_csv,
            n_harmonics=args.n_harmonics,
            tol_pct=args.tol_pct,
            tol_min_hz=args.tol_min_hz,
        )
    elif args.cmd == 'verify':
        verify_match_on_file(
            raw_file=args.file,
            channel=args.channel,
            top_n=args.top_n,
            n_harmonics=args.n_harmonics,
            tol_pct=args.tol_pct,
            tol_min_hz=args.tol_min_hz,
        )
    else:
        parser.print_help()
