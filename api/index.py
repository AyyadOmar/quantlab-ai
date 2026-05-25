from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "docs" / "demo"


def read_json(filename: str) -> dict | list:
    return json.loads((DEMO_DIR / filename).read_text())


app = FastAPI(title="QuantLab AI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "quantlab-ai-api"}


@app.get("/api/demo/summary")
def demo_summary() -> dict:
    return read_json("aapl_xgboost_summary.json")


@app.get("/api/demo/benchmarks")
def demo_benchmarks() -> dict:
    return read_json("aapl_xgboost_benchmarks.json")


@app.get("/api/demo/live-signals")
def demo_live_signals() -> list[dict]:
    return read_json("aapl_latest_signals.json")


@app.get("/api/demo/cross-validation")
def demo_cross_validation() -> dict:
    return read_json("aapl_xgboost_cross_validation.json")
