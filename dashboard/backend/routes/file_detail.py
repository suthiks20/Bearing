"""GET /api/file/{idx} -> per-file aggregated features, dominant fault type, RUL."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from inference_cache import cache
from src.features.physics_features import compute_fault_frequencies

router = APIRouter()

_FAULT_FREQS_HZ = compute_fault_frequencies()
_FAULT_LABELS = {
    "bpfo": "BPFO (outer race)",
    "bpfi": "BPFI (inner race)",
    "bsf": "BSF (ball spin)",
    "ftf": "FTF (cage)",
}


@router.get("/file/{idx}")
def get_file_detail(idx: int):
    n_files = len(cache.file_names)
    if idx < 0 or idx >= n_files:
        raise HTTPException(status_code=404, detail=f"file index out of range [0, {n_files - 1}]")

    feat_row = cache.per_file_features_df.iloc[idx]
    hi_row = cache.health_index_df.iloc[idx]
    rul_row = cache.rul_df.iloc[idx]

    match_counts = {
        "bpfo": float(feat_row["bpfo_match_count"]),
        "bpfi": float(feat_row["bpfi_match_count"]),
        "bsf": float(feat_row["bsf_match_count"]),
        "ftf": float(feat_row["ftf_match_count"]),
    }
    dominant_fault = max(match_counts, key=match_counts.get)
    dominant_strength = match_counts[dominant_fault]

    return {
        "idx": idx,
        "file": cache.file_names[idx],
        "hi_proxy": round(float(hi_row["hi_proxy"]), 5),
        "hi_cnn": round(float(hi_row["hi_cnn"]), 5),
        "rul_estimate": round(float(rul_row["rul_estimate"]), 2),
        "rul_lower_1sigma": round(float(rul_row["rul_lower_1sigma"]), 2),
        "rul_upper_1sigma": round(float(rul_row["rul_upper_1sigma"]), 2),
        "kurtosis": round(float(feat_row["kurtosis"]), 4),
        "entropy": round(float(feat_row["entropy"]), 4),
        "match_counts": match_counts,
        "dominant_fault": dominant_fault,
        "dominant_fault_label": _FAULT_LABELS[dominant_fault],
        "dominant_fault_strength": dominant_strength,
        "dominant_fault_hz": round(_FAULT_FREQS_HZ[dominant_fault], 2),
        "fault_freqs_hz": {k: round(v, 2) for k, v in _FAULT_FREQS_HZ.items()},
    }
