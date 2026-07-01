"""
FastAPI entrypoint for the bearing RUL dashboard backend.

Read-only with respect to src/, results/, data/ -- this process only reads
existing CSVs/PNGs/model weights and serves them as JSON. Run from this
directory:

    uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))             # dashboard/backend
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # project root

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from inference_cache import cache
from routes import health_index, rul, bpfo_trend, file_detail, scalogram

app = FastAPI(title="Bearing RUL Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _load_cache() -> None:
    cache.load()


app.include_router(health_index.router, prefix="/api")
app.include_router(rul.router, prefix="/api")
app.include_router(bpfo_trend.router, prefix="/api")
app.include_router(file_detail.router, prefix="/api")
app.include_router(scalogram.router, prefix="/api")


@app.get("/api/health")
def health_check():
    return {"status": "ok", "n_files": len(cache.file_names)}
