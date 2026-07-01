"""
RUL estimation from a CNN-derived health index curve.

Algorithm
---------
For each evaluation point t (on a sliding stride):
  1. Take the last FIT_WINDOW health-index values ending at t.
  2. Try to fit an exponential decay: HI(t) = A * exp(-b * t').
     If curve_fit fails to converge, fall back to a linear fit.
     Both fallback count and total fits are logged at the end.
  3. Extrapolate the fitted curve forward until it crosses HI_FAIL.
     The time-to-crossing is the estimated RUL at point t.
  4. If the window is near-flat (std < FLAT_STD_THR) or the extrapolated
     crossing never arrives, clamp RUL to MAX_LIFE (total observed run).
  5. Uncertainty: ±1 std-dev of fit residuals -> propagated to crossing-time
     uncertainty via a ±1σ perturbation of the fit parameters.

Thresholds confirmed against proxy RMS health index:
  HI_ONSET = 0.7  -- degradation first detected
  HI_FAIL  = 0.5  -- failure threshold (RUL = 0 when HI crosses this)

Plots saved:
  results/figures/health_index_curve.png
  results/figures/rul_curve.png
"""
from __future__ import annotations

import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Tuneable constants (change here, not inline)
# ---------------------------------------------------------------------------
HI_ONSET    = 0.70   # HI value where degradation is considered to have started
HI_FAIL     = 0.50   # HI value that defines failure (RUL = 0)
FIT_WINDOW  = 30     # number of recent HI points used for each local trend fit
                     # (30 pts = ~5 h at 10-min sampling — keeps ~1/3 healthy
                     # + ~2/3 degrading near failure; avoids diluting the slope)
STRIDE      = 5      # evaluate RUL every STRIDE time steps (for speed)
FLAT_STD_THR = 0.01  # if window std < this, treat as flat -> clamp to MAX_LIFE

_HI_CSV     = _PROJECT_ROOT / 'results' / 'metrics' / 'health_index.csv'
_FIG_DIR    = _PROJECT_ROOT / 'results' / 'figures'


# ---------------------------------------------------------------------------
# Fitting models
# ---------------------------------------------------------------------------

def _exp_model(t, A, b):
    return A * np.exp(-b * t)


def _lin_model(t, m, c):
    return m * t + c


def _fit_window(t_win: np.ndarray, hi_win: np.ndarray):
    """
    Try exponential fit; fall back to linear on failure.
    Returns (pred_fn, residual_std, used_fallback: bool).
    """
    used_fallback = False

    # Attempt exponential
    try:
        # Initial guess: A = hi_win[0], b modest positive slope
        A0 = float(np.clip(hi_win[0], 1e-6, 1.0))
        b0 = max(1e-6, float((hi_win[0] - hi_win[-1]) / (t_win[-1] - t_win[0] + 1e-9)))
        p, _ = curve_fit(
            _exp_model, t_win, hi_win,
            p0=[A0, b0],
            bounds=([0, 0], [np.inf, np.inf]),
            maxfev=2000,
        )
        pred_fn    = lambda t, p=p: _exp_model(t, *p)
        resid_std  = float(np.std(hi_win - pred_fn(t_win)))

    except (RuntimeError, ValueError):
        used_fallback = True
        p, _ = np.polyfit(t_win, hi_win, 1), None
        p    = np.polyfit(t_win, hi_win, 1)
        pred_fn   = lambda t, p=p: np.polyval(p, t)
        resid_std = float(np.std(hi_win - pred_fn(t_win)))

    return pred_fn, resid_std, used_fallback


def _extrapolate_crossing(pred_fn, t_last: float, hi_fail: float,
                           max_life: float, search_steps: int = 50_000):
    """
    Walk forward from t_last until pred_fn crosses hi_fail.
    Returns time-to-crossing (RUL), clamped to max_life if never reached.
    """
    dt = (max_life - t_last) / search_steps if max_life > t_last else 1.0
    t  = t_last
    for _ in range(search_steps):
        t += dt
        if t > max_life * 2:          # safety ceiling
            return max_life
        try:
            val = float(pred_fn(t))
        except Exception:
            return max_life
        if val <= hi_fail:
            return max(0.0, t - t_last)
    return max_life                   # crossed beyond search range -> clamp


# ---------------------------------------------------------------------------
# Main estimation function
# ---------------------------------------------------------------------------

def estimate_rul(hi_csv: Path | str = _HI_CSV,
                 fig_dir: Path | str = _FIG_DIR) -> pd.DataFrame:
    hi_csv  = Path(hi_csv)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not hi_csv.exists():
        raise FileNotFoundError(
            f"Health index file not found: {hi_csv}\n"
            "Run Stage 2 (CNN training) first to generate it."
        )

    df = pd.read_csv(hi_csv)
    if 'health_index' not in df.columns:
        raise ValueError(
            f"Expected column 'health_index' in {hi_csv}. "
            f"Got: {list(df.columns)}"
        )

    hi_vals  = df['health_index'].to_numpy(dtype=float)
    n        = len(hi_vals)
    t_all    = np.arange(n, dtype=float)
    max_life = float(n - 1)

    # ---- plot 1: raw health index ----------------------------------------
    _plot_health_index(t_all, hi_vals, fig_dir)

    # ---- sliding RUL estimation ------------------------------------------
    eval_indices = np.arange(FIT_WINDOW, n, STRIDE)

    rul_est   = np.full(n, np.nan)
    rul_lo    = np.full(n, np.nan)    # -1σ band
    rul_hi    = np.full(n, np.nan)    # +1σ band

    n_fallback   = 0
    n_total_fits = 0
    n_clamped    = 0

    for idx in eval_indices:
        t_win  = t_all[idx - FIT_WINDOW: idx]
        hi_win = hi_vals[idx - FIT_WINDOW: idx]

        # Clamp near-flat windows (long healthy plateau -> RUL = max_life)
        if np.std(hi_win) < FLAT_STD_THR:
            rul_est[idx] = max_life
            rul_lo[idx]  = max_life
            rul_hi[idx]  = max_life
            n_clamped   += 1
            n_total_fits += 1
            continue

        pred_fn, resid_std, used_fb = _fit_window(t_win, hi_win)
        n_total_fits += 1
        if used_fb:
            n_fallback += 1

        t_last  = t_win[-1]
        central = _extrapolate_crossing(pred_fn, t_last, HI_FAIL, max_life)

        # Uncertainty: perturb the predicted HI up/down by ±1σ at t_last,
        # refit crossing from those perturbed starting points
        hi_at_last = float(pred_fn(t_last))
        def _perturbed_crossing(delta_hi: float) -> float:
            hi_perturbed = np.clip(hi_win + delta_hi, 0.0, 1.0)
            pf, _, _ = _fit_window(t_win, hi_perturbed)
            return _extrapolate_crossing(pf, t_last, HI_FAIL, max_life)

        ru_lo = _perturbed_crossing(-resid_std)
        ru_hi = _perturbed_crossing(+resid_std)

        rul_est[idx] = np.clip(central, 0.0, max_life)
        rul_lo[idx]  = np.clip(ru_lo,   0.0, max_life)
        rul_hi[idx]  = np.clip(ru_hi,   0.0, max_life)

        if central >= max_life:
            n_clamped += 1

    print(f"\nRUL fit summary:")
    print(f"  Total fit windows   : {n_total_fits}")
    print(f"  Linear fallbacks    : {n_fallback}  ({100*n_fallback/max(1,n_total_fits):.1f}%)")
    print(f"  Clamped to max_life : {n_clamped}   ({100*n_clamped/max(1,n_total_fits):.1f}%)")

    # ---- fill NaN gaps (before first eval point) with max_life --------
    rul_est[:FIT_WINDOW] = max_life
    rul_lo[:FIT_WINDOW]  = max_life
    rul_hi[:FIT_WINDOW]  = max_life

    # Forward-fill any remaining NaN from stride gaps
    for arr in (rul_est, rul_lo, rul_hi):
        mask = np.isnan(arr)
        for i in range(1, len(arr)):
            if mask[i]:
                arr[i] = arr[i - 1]

    # ---- plot 2: RUL curve with uncertainty band -------------------------
    _plot_rul(t_all, rul_est, rul_lo, rul_hi, hi_vals, fig_dir)

    # ---- output dataframe -----------------------------------------------
    result = df.copy()
    result['rul_estimate'] = rul_est
    result['rul_lower_1sigma'] = rul_lo
    result['rul_upper_1sigma'] = rul_hi
    out_csv = hi_csv.parent / 'rul_estimates.csv'
    result.to_csv(out_csv, index=False)
    print(f"Saved RUL estimates -> {out_csv}")

    return result


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_health_index(t, hi, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, hi, linewidth=0.9, color='steelblue', label='Health Index')
    ax.axhline(HI_ONSET, color='orange', linestyle='--', linewidth=1.0,
               label=f'Onset threshold = {HI_ONSET}')
    ax.axhline(HI_FAIL,  color='red',    linestyle='--', linewidth=1.0,
               label=f'Failure threshold = {HI_FAIL}')
    ax.set_xlabel('Time step (file index)')
    ax.set_ylabel('Health Index')
    ax.set_title('CNN Health Index over Time')
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    plt.tight_layout()
    out = fig_dir / 'health_index_curve.png'
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"Saved -> {out}")


def _plot_rul(t, rul, rul_lo, rul_hi, hi, fig_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: health index
    ax1.plot(t, hi, linewidth=0.8, color='steelblue')
    ax1.axhline(HI_ONSET, color='orange', linestyle='--', linewidth=0.9,
                label=f'Onset={HI_ONSET}')
    ax1.axhline(HI_FAIL, color='red', linestyle='--', linewidth=0.9,
                label=f'Failure={HI_FAIL}')
    ax1.set_ylabel('Health Index')
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=8)
    ax1.set_title('Health Index and RUL Estimate with ±1σ Uncertainty')

    # Bottom: RUL
    ax2.plot(t, rul, linewidth=1.0, color='darkorange', label='RUL estimate')
    ax2.fill_between(t, rul_lo, rul_hi, alpha=0.25, color='darkorange',
                     label='±1σ uncertainty band')
    ax2.set_xlabel('Time step (file index)')
    ax2.set_ylabel('Remaining Useful Life [time steps]')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out = fig_dir / 'rul_curve.png'
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"Saved -> {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Estimate RUL from CNN health index curve.')
    parser.add_argument('--hi_csv',  default=str(_HI_CSV),
                        help='Path to health_index.csv')
    parser.add_argument('--fig_dir', default=str(_FIG_DIR),
                        help='Output directory for figures')
    args = parser.parse_args()
    estimate_rul(hi_csv=args.hi_csv, fig_dir=args.fig_dir)
