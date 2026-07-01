"""GET /api/scalogram/{idx} -> nearest precomputed scalogram PNG."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()

_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "scalograms"


def _available_indices() -> list[int]:
    return sorted(
        int(p.stem.split("_")[1])
        for p in _CACHE_DIR.glob("file_*.png")
    )


@router.get("/scalogram/{idx}")
def get_scalogram(idx: int):
    indices = _available_indices()
    if not indices:
        raise HTTPException(status_code=503, detail="Scalogram cache is empty. Run precompute_scalograms.py first.")

    nearest = min(indices, key=lambda i: abs(i - idx))
    path = _CACHE_DIR / f"file_{nearest:04d}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Cached scalogram not found")

    return FileResponse(path, media_type="image/png", headers={"X-Nearest-Index": str(nearest)})


@router.get("/scalogram-index")
def get_scalogram_index():
    return {"available_indices": _available_indices()}
