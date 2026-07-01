"""GET /api/bpfo-trend -> bpfo_trend_stats.csv as JSON (984 rows)."""
from __future__ import annotations

from fastapi import APIRouter

from inference_cache import cache

router = APIRouter()


@router.get("/bpfo-trend")
def get_bpfo_trend():
    df = cache.bpfo_trend_df
    return {
        "file_idx": df["file_idx"].tolist(),
        "bpfo_rate": df["bpfo_rate"].round(4).tolist(),
        "bpfi_rate": df["bpfi_rate"].round(4).tolist(),
        "bsf_rate": df["bsf_rate"].round(4).tolist(),
        "ftf_rate": df["ftf_rate"].round(4).tolist(),
        "mean_rms": df["mean_rms"].round(6).tolist(),
    }
