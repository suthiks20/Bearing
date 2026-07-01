"""GET /api/health-index -> CNN + proxy HI per file, plus MAE summary."""
from __future__ import annotations

from fastapi import APIRouter

from inference_cache import cache

router = APIRouter()


@router.get("/health-index")
def get_health_index():
    df = cache.health_index_df
    return {
        "files": df["file"].tolist(),
        "hi_proxy": df["hi_proxy"].round(5).tolist(),
        "hi_cnn": df["hi_cnn"].round(5).tolist(),
        "train_n": 787,
        "mae_train": round(cache.cnn_mae_train, 5),
        "mae_val": round(cache.cnn_mae_val, 5),
        "onset_threshold": 0.70,
        "fail_threshold": 0.50,
    }
