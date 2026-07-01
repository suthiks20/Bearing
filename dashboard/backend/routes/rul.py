"""GET /api/rul -> rul_estimates.csv as JSON (984 rows)."""
from __future__ import annotations

from fastapi import APIRouter

from inference_cache import cache

router = APIRouter()


@router.get("/rul")
def get_rul():
    df = cache.rul_df
    return {
        "files": df["file"].tolist(),
        "health_index": df["health_index"].round(5).tolist(),
        "rul_estimate": df["rul_estimate"].round(2).tolist(),
        "rul_lower_1sigma": df["rul_lower_1sigma"].round(2).tolist(),
        "rul_upper_1sigma": df["rul_upper_1sigma"].round(2).tolist(),
        "max_life": 983,
    }
