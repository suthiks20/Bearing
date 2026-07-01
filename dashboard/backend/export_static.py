"""
export_static.py — one-time script that materialises every backend API
response as static JSON (and copies scalogram PNGs) so the frontend can
be deployed without a live backend.

Run from the project root or from dashboard/backend/:
    python dashboard/backend/export_static.py

Output → dashboard/frontend/public/data/
    health-index.json
    rul.json
    bpfo-trend.json
    file-details.json     (array of all 984 FileDetailResponse objects)
    scalogram-index.json  ({available_indices: [...]})
    scalograms/           (copied from dashboard/backend/cache/scalograms/)
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent          # dashboard/backend/
PROJECT_ROOT = SCRIPT_DIR.parent.parent                  # project root
OUT_DIR      = SCRIPT_DIR.parent / "frontend" / "public" / "data"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# ── Imports after path setup ──────────────────────────────────────────────────
from inference_cache import cache
from src.features.physics_features import compute_fault_frequencies

_FAULT_FREQS_HZ = compute_fault_frequencies()
_FAULT_LABELS = {
    "bpfo": "BPFO (outer race)",
    "bpfi": "BPFI (inner race)",
    "bsf": "BSF (ball spin)",
    "ftf": "FTF (cage)",
}


def _write(name: str, data: object) -> None:
    path = OUT_DIR / name
    path.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print(f"  wrote  {name}  ({path.stat().st_size:,} bytes)")


def main() -> None:
    print("[export] Loading pipeline cache …")
    cache.load()
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. health-index.json ─────────────────────────────────────────────────────
    df = cache.health_index_df
    _write("health-index.json", {
        "files":            df["file"].tolist(),
        "hi_proxy":         df["hi_proxy"].round(5).tolist(),
        "hi_cnn":           df["hi_cnn"].round(5).tolist(),
        "train_n":          787,
        "mae_train":        round(float(cache.cnn_mae_train), 5),
        "mae_val":          round(float(cache.cnn_mae_val), 5),
        "onset_threshold":  0.70,
        "fail_threshold":   0.50,
    })

    # 2. rul.json ──────────────────────────────────────────────────────────────
    rdf = cache.rul_df
    _write("rul.json", {
        "files":              rdf["file"].tolist(),
        "health_index":       rdf["health_index"].round(5).tolist(),
        "rul_estimate":       rdf["rul_estimate"].round(2).tolist(),
        "rul_lower_1sigma":   rdf["rul_lower_1sigma"].round(2).tolist(),
        "rul_upper_1sigma":   rdf["rul_upper_1sigma"].round(2).tolist(),
        "max_life":           983,
    })

    # 3. bpfo-trend.json ───────────────────────────────────────────────────────
    bdf = cache.bpfo_trend_df
    _write("bpfo-trend.json", {
        "file_idx":   [int(x) for x in bdf["file_idx"].tolist()],
        "bpfo_rate":  bdf["bpfo_rate"].round(4).tolist(),
        "bpfi_rate":  bdf["bpfi_rate"].round(4).tolist(),
        "bsf_rate":   bdf["bsf_rate"].round(4).tolist(),
        "ftf_rate":   bdf["ftf_rate"].round(4).tolist(),
        "mean_rms":   bdf["mean_rms"].round(6).tolist(),
    })

    # 4. file-details.json ─────────────────────────────────────────────────────
    n_files = len(cache.file_names)
    all_details = []
    for idx in range(n_files):
        feat = cache.per_file_features_df.iloc[idx]
        hi   = cache.health_index_df.iloc[idx]
        rul  = cache.rul_df.iloc[idx]

        match_counts: dict[str, float] = {
            "bpfo": float(feat["bpfo_match_count"]),
            "bpfi": float(feat["bpfi_match_count"]),
            "bsf":  float(feat["bsf_match_count"]),
            "ftf":  float(feat["ftf_match_count"]),
        }
        dominant = max(match_counts, key=match_counts.get)

        all_details.append({
            "idx":                    idx,
            "file":                   cache.file_names[idx],
            "hi_proxy":               round(float(hi["hi_proxy"]), 5),
            "hi_cnn":                 round(float(hi["hi_cnn"]), 5),
            "rul_estimate":           round(float(rul["rul_estimate"]), 2),
            "rul_lower_1sigma":       round(float(rul["rul_lower_1sigma"]), 2),
            "rul_upper_1sigma":       round(float(rul["rul_upper_1sigma"]), 2),
            "kurtosis":               round(float(feat["kurtosis"]), 4),
            "entropy":                round(float(feat["entropy"]), 4),
            "match_counts":           match_counts,
            "dominant_fault":         dominant,
            "dominant_fault_label":   _FAULT_LABELS[dominant],
            "dominant_fault_strength": match_counts[dominant],
            "dominant_fault_hz":      round(_FAULT_FREQS_HZ[dominant], 2),
            "fault_freqs_hz":         {k: round(v, 2) for k, v in _FAULT_FREQS_HZ.items()},
        })

    _write("file-details.json", all_details)

    # 5. scalogram-index.json ──────────────────────────────────────────────────
    scalogram_src = SCRIPT_DIR / "cache" / "scalograms"
    available_indices = sorted(
        int(p.stem.split("_")[1])
        for p in scalogram_src.glob("file_*.png")
    )
    _write("scalogram-index.json", {"available_indices": available_indices})

    # 6. Copy scalogram PNGs ───────────────────────────────────────────────────
    scalogram_dst = OUT_DIR / "scalograms"
    if scalogram_dst.exists():
        shutil.rmtree(scalogram_dst)
    shutil.copytree(scalogram_src, scalogram_dst)
    n_pngs = len(list(scalogram_dst.glob("*.png")))
    print(f"  copied {n_pngs} scalogram PNGs -> {scalogram_dst.relative_to(SCRIPT_DIR.parent.parent)}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_bytes = sum(f.stat().st_size for f in OUT_DIR.rglob("*") if f.is_file())
    print(f"\n[export] Done.  Total: {total_bytes:,} bytes  ({total_bytes / 1_048_576:.1f} MB)")
    print(f"         Output: {OUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
