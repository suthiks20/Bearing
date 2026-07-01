# NASA IMS Bearing RUL Prediction Pipeline

Predicts the Remaining Useful Life (RUL) of rolling-element bearings using the
NASA IMS Bearing Dataset.  The pipeline combines Fractional Synchrosqueezed
Transform (FrSST) time-frequency analysis, physics-informed fault-frequency
matching, and a 1-D temporal CNN health-index model.

---

## Dataset

**NASA IMS Bearing Dataset — 2nd test run**
- Source: NASA Prognostics Center of Excellence (PCoE)
- Duration: 7 days (Feb 12–19 2004), accelerated run-to-failure
- Files: 984 binary-timestamped vibration records
- Format: whitespace-separated floats, 20 480 samples × 4 channels (one
  accelerometer per bearing), 20 kHz sampling rate
- Bearing model: Rexnord ZA-2115, double-row; outer race failure observed in
  Bearing 1 (channel 0) and inner race in Bearing 3 (channel 2)

Place the raw files flat inside `data/raw/2nd_test/` (no subdirectories).

---

## Pipeline Stages

### Stage 1 — FrSST Feature Extraction

**Script**: `src/features/extract_features.py`
**Output**: `results/metrics/peak_features.csv`

Each file is split into adaptive-width peak windows (512–4 096 samples) using
envelope detection and `scipy.signal.find_peaks`.  For each window the
Fractional Synchrosqueezed Transform (`ssqueezepy 0.6.6`) produces a
high-resolution TF map from which the top-3 dominant frequencies and their
relative magnitudes are extracted.

If `ssqueezepy` is absent the module raises `ImportWarning` at import and
`RuntimeError` at runtime — the CWT fallback is **never silent**.

**Physics features** (`src/features/physics_features.py`)
Appended to the same CSV via `add_physics_features()`:

| Feature | Description |
|---|---|
| `kurtosis` | Excess kurtosis (Fisher) — sensitive to impact impulsiveness |
| `entropy` | Shannon entropy of the amplitude histogram |
| `bpfo_match_count` | FrSST dominant frequencies matching BPFO harmonics |
| `bpfi_match_count` | Same for Ball Pass Frequency Inner race |
| `bsf_match_count` | Same for Ball Spin Frequency |
| `ftf_match_count` | Same for Fundamental Train Frequency |

Rexnord ZA-2115 fault frequencies (exact values, 2000 RPM, 20 480 Hz):

| Frequency | Value (Hz) |
|---|---|
| FTF | 14.775 |
| BPFO | 236.40 |
| BPFI | 296.93 |
| BSF | 139.92 |

Match tolerance: `max(20 Hz, 3% × harmonic_freq)` — scales with harmonic
number to match FrSST log-spaced frequency resolution.

---

### Stage 2 — CNN Health Index Model

**Script**: `src/models/train_cnn.py`
**Outputs**: `results/metrics/health_index.csv`, `results/metrics/health_index_proxy.csv`,
`results/metrics/hi_cnn.pth`, `results/figures/training_loss.png`,
`results/figures/cnn_vs_proxy_hi.png`

A 1-D temporal CNN is trained to regress a proxy health index from 12
physics-informed features (excluding rms and energy — see leakage note below).

**Input**: (batch=32, T=30, features=12)
30 consecutive per-file feature vectors form one sample; the CNN slides over
the temporal axis.

**Architecture**:
```
Conv1D(12->32, k=5) + BN + ReLU
Conv1D(32->32, k=3) + BN + ReLU
GlobalAvgPool
Dense(32) + ReLU + Dropout(0.3)
Dense(1)  + Sigmoid  ->  HI in [0, 1]
```

**Label leakage prevention**
The proxy health index (training label) is derived from RMS:
`HI = 1 - (rms - baseline) / (rms_max - baseline)`.
Including `rms` or `energy` as features would give the model trivial access to
its own label.  Both are excluded and a module-level `assert` in
`cnn_health_index.py` raises `AssertionError` if either is re-added.

**Train / val split**
Files 0-786 (train) / Files 787-983 (val).  `build_windows()` is called
*separately* on each partition — no sliding window ever spans the 786/787
file boundary, which would leak future degradation data into training.

**Loss**: `MSE + 0.1 * sum(relu(HI[t+1] - HI[t])^2)`
Monotonicity penalty discourages predicted HI from rising within a batch
(batches fed chronologically; shuffle=False).

**health_index.csv source**
`health_index.csv` (read by Stage 3) is the Gaussian-smoothed proxy RMS HI.
The CNN's per-file predictions are available via `src/models/predict_hi.py`
and are saved to `health_index_proxy.csv` alongside the proxy for comparison.
See the *Model Validation* section below for why CNN predictions are not yet
used to drive Stage 3.

---

### Stage 3 — RUL Estimation

**Script**: `src/models/rul_estimation.py`
**Outputs**: `results/figures/health_index_curve.png`,
`results/figures/rul_curve.png`, `results/metrics/rul_estimates.csv`

For each evaluation point (every 5 files):
1. Fit `HI(t) = A × exp(−b × t)` on the last 30 HI values.
2. If curve_fit fails, fall back to linear (count logged to console).
3. Extrapolate to failure threshold `HI_FAIL = 0.50`.
4. Clamp to `MAX_LIFE` when the window is flat (`std < 0.01`).
5. Uncertainty: ±1σ from fit residuals propagated via HI perturbation.

Expected RUL curve: flat near `MAX_LIFE` for the first ~93% of the run
(healthy plateau), then a sharp nonlinear drop as degradation accelerates.

---

## Model Validation: CNN vs Proxy HI

`src/models/predict_hi.py` runs the trained CNN (`hi_cnn.pth`) over all 984
files and compares its output against the proxy RMS HI side by side
(`results/figures/cnn_vs_proxy_hi.png`, `health_index_proxy.csv` for the
proxy, `health_index.csv` for whichever curve is currently authoritative).

**Finding: the CNN does not detect degradation.**

| Region | CNN HI | Proxy HI |
|---|---|---|
| File 965 (onset) | 0.9951 | 0.6795 |
| File 975 (failure point) | 0.9983 | 0.3863 |
| File 983 (end of run) | 0.9983 | 0.6595 |

CNN predictions stay within `[0.756, 0.998]` across all 984 files and never
cross the onset threshold (0.70), let alone the failure threshold (0.50).
MAE overall is 0.082, but MAE in the validation region (files 787–983, which
contains every degraded file) is 0.139 — the model is uniformly wrong exactly
where it matters. Feeding CNN predictions into `rul_estimation.py` produces a
RUL curve that is flat at `MAX_LIFE` for all 984 files: no hockey-stick, no
usable prognosis.

**Root cause**: the train/val split is strictly chronological (files 0–786
train, 787–983 val) to avoid label leakage across the partition boundary.
Every file with proxy HI < 0.70 falls in the validation set, so the CNN was
trained exclusively on healthy examples. It learned to map physics-feature
patterns to ~1.0 because that was the only label it ever saw — the
degradation feature patterns at files 965–983 are out-of-distribution for the
trained weights, not a calibration error that smoothing or thresholding can
fix.

**Decision**: `health_index.csv` is populated from the proxy RMS HI, not CNN
output (`run_pipeline.py` Stage 2 copies `health_index_proxy.csv` ->
`health_index.csv`). The CNN and its inference path (`predict_hi.py`) remain
in the codebase as a diagnostic / research component — `predict_hi.py` will
raise `RuntimeError` if val-region MAE exceeds 0.25 (currently 0.139, so it
completes but the divergence is still visible in the comparison plot). Fixing
this would require either relaxing the chronological split to include
degraded examples in training, or reformulating the problem as anomaly
detection trained only on healthy data — both are outside the scope of the
current strict no-leakage constraint and are left as future work.

---

## How to Run

### Prerequisites

```bash
# Create and activate venv (Python 3.10)
python -m venv .venv
.venv\Scripts\activate           # Windows
source .venv/bin/activate        # Linux/macOS

# Core dependencies
pip install -r requirements.txt

# PyTorch (CPU-only — install separately due to custom index)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Full pipeline

```bash
python run_pipeline.py
```

### Skip feature extraction (reuse existing CSVs)

```bash
python run_pipeline.py --skip-extraction
```

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--skip-extraction` | off | Reuse existing `peak_features_augmented.csv` |
| `--skip-training` | off | Reuse existing `health_index.csv` |
| `--method` | `frsst` | TF method: `frsst`, `sst`, or `cwt` |
| `--channel` | `0` | Bearing channel (0-indexed; 0 = Bearing 1) |

---

## Project Structure

```
ts_frsst_project/
├── data/
│   └── raw/2nd_test/            # NASA IMS raw files (not in repo)
├── src/
│   ├── features/
│   │   ├── extract_features.py  # Stage 1 pipeline orchestration
│   │   ├── physics_features.py  # Fault-frequency matching + time-domain stats
│   │   ├── peak_detection.py    # Adaptive window / envelope detection
│   │   └── tf_extraction.py     # FrSST (ssqueezepy) + CWT (pywavelets)
│   ├── models/
│   │   ├── cnn_health_index.py  # CNN architecture + feature specification
│   │   ├── train_cnn.py         # Training loop + health_index.csv output
│   │   └── rul_estimation.py    # Sliding-window RUL from HI curve
│   └── utils/
│       └── io_utils.py          # NASA IMS file loader
├── results/
│   ├── figures/                 # All plots (PNG)
│   └── metrics/                 # All CSVs + model weights
├── run_pipeline.py              # Single entrypoint for all three stages
├── requirements.txt
└── README.md
```

---

## Results

After a complete run, the following files are produced:

| Path | Contents |
|---|---|
| `results/figures/training_loss.png` | CNN train / val loss (log scale) |
| `results/figures/health_index_curve.png` | HI over 984 files with thresholds |
| `results/figures/rul_curve.png` | RUL estimate with ±1σ uncertainty band |
| `results/figures/bpfo_match_trend.png` | Per-file fault match rates vs RMS |
| `results/metrics/peak_features_augmented.csv` | 12 features per window per file |
| `results/metrics/health_index.csv` | Smoothed proxy HI, one row per file |
| `results/metrics/rul_estimates.csv` | RUL, lower and upper 1σ bands |
| `results/metrics/hi_cnn.pth` | Best CNN weights (lowest val loss) |

---

## Key Design Decisions

- **FrSST over CWT**: sharper frequency ridges (synchrosqueezed), faster in
  batch mode once numba JIT has warmed up (~2 s/file vs ~3 s/file cold).
- **No rms/energy in CNN features**: both are monotonically related to the HI
  label; including them reduces the problem to trivial identity regression.
- **max-aggregation for fault counts and kurtosis**: a single impulsive window
  per file is the degradation signal; mean aggregation dilutes sparse
  early-stage indicators.
- **FIT_WINDOW = 30** in RUL estimation: the 2nd test bearing degrades over
  only 6–16 files from onset to failure; a wider fit window dilutes the slope.
- **Chronological split, no shuffle**: random splitting would mix degraded and
  healthy files into both partitions, giving optimistically inflated val loss.
