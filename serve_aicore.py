"""
serve_aicore.py - AquaVerde quantile demand forecast + replenishment API

Supports both bundle formats:
  1) Preferred Databricks export bundle:
       {"model": <unwrapped MLflow PythonModel>, "features": [...], "horizon": 7}
  2) Older two-model bundle:
       {"models": {0.5: model_p50, 0.9: model_p90}, "features": [...], "horizon": 7}

Endpoints:
  GET  /v2/healthz      lightweight health check, always avoids model load
  GET  /v2/readyz       loads model + feature snapshot and validates readiness
  GET  /v2/skus         lists available SKU / DC pairs
  POST /v2/forecast     {"sku":"SKU1014", "dc_id":"DC_SOF"}
  POST /v2/replenish    {"sku":"SKU1014", "dc_id":"DC_SOF"}
"""
from pathlib import Path
import math
import os
from typing import Any, Tuple

import cloudpickle
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

APP = Path(os.getenv("APP_DIR", "/app"))
REVIEW_DAYS = 1
SERVICE_Z = 1.645  # Approx. 95% service level

app = FastAPI(title="AquaVerde Demand Forecast and Replenishment", version="3.3")
_STATE: dict[str, Any] = {}


def _load_bundle(path: Path) -> dict[str, Any]:
    """Load either a cloudpickle or joblib model bundle."""
    try:
        with open(path, "rb") as f:
            bundle = cloudpickle.load(f)
    except Exception:
        bundle = joblib.load(path)

    if not isinstance(bundle, dict):
        # Allow a raw pyfunc-like model, but require features to be supplied separately in future.
        raise ValueError(
            "model.joblib must be a dict bundle. Expected keys: model/features/horizon "
            "or models/features/horizon."
        )

    if "features" not in bundle:
        raise ValueError(f"model.joblib missing key 'features'. Found keys: {list(bundle.keys())}")

    if "model" not in bundle and "models" not in bundle:
        raise ValueError(
            "model.joblib must contain either key 'model' or key 'models'. "
            f"Found keys: {list(bundle.keys())}"
        )

    bundle.setdefault("horizon", 7)
    return bundle


def state() -> dict[str, Any]:
    if not _STATE:
        model_path = APP / "model.joblib"
        csv_path = APP / "latest_features.csv"

        if not model_path.exists():
            raise FileNotFoundError(f"Missing model bundle: {model_path}")
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing feature snapshot: {csv_path}")

        bundle = _load_bundle(model_path)
        snap = pd.read_csv(csv_path)

        required_base = {"sku", "dc_id"}
        missing_base = sorted(required_base - set(snap.columns))
        if missing_base:
            raise ValueError(f"latest_features.csv missing required columns: {missing_base}")

        features = list(bundle["features"])
        missing_features = [c for c in features if c not in snap.columns]
        if missing_features:
            raise ValueError(
                "latest_features.csv is missing model feature columns: "
                f"{missing_features[:30]}"
            )

        snap["sku"] = snap["sku"].astype(str)
        snap["dc_id"] = snap["dc_id"].astype(str)
        snap["key"] = snap["sku"] + "|" + snap["dc_id"]

        _STATE.update(
            bundle=bundle,
            features=features,
            horizon=int(bundle.get("horizon", 7)),
            rows={r["key"]: r for r in snap.to_dict("records")},
        )
    return _STATE


class Req(BaseModel):
    sku: str
    dc_id: str


def _extract_quantiles(pred: Any) -> Tuple[float, float]:
    """Parse prediction result into p50 and p90 values."""
    if isinstance(pred, pd.DataFrame):
        row = pred.iloc[0]
        candidates_p50 = ["forecast_p50_7d", "p50", "q50", "median", "0.5"]
        candidates_p90 = ["forecast_p90_7d", "p90", "q90", "0.9"]
        p50 = next((row[c] for c in candidates_p50 if c in pred.columns), None)
        p90 = next((row[c] for c in candidates_p90 if c in pred.columns), None)
        if p50 is not None and p90 is not None:
            return float(p50), float(p90)
        if pred.shape[1] >= 2:
            return float(row.iloc[0]), float(row.iloc[1])

    if isinstance(pred, dict):
        candidates_p50 = ["forecast_p50_7d", "p50", "q50", "median", 0.5, "0.5"]
        candidates_p90 = ["forecast_p90_7d", "p90", "q90", 0.9, "0.9"]
        p50 = next((pred[k] for k in candidates_p50 if k in pred), None)
        p90 = next((pred[k] for k in candidates_p90 if k in pred), None)
        if isinstance(p50, (list, tuple, np.ndarray, pd.Series)):
            p50 = p50[0]
        if isinstance(p90, (list, tuple, np.ndarray, pd.Series)):
            p90 = p90[0]
        if p50 is not None and p90 is not None:
            return float(p50), float(p90)

    arr = np.asarray(pred)
    if arr.ndim == 0:
        raise ValueError("Model returned only one scalar; expected P50 and P90.")
    if arr.ndim == 1:
        if len(arr) >= 2:
            return float(arr[0]), float(arr[1])
    if arr.ndim >= 2 and arr.shape[1] >= 2:
        return float(arr[0, 0]), float(arr[0, 1])

    raise ValueError(f"Cannot extract P50/P90 from prediction output type {type(pred)}: {pred}")


def _predict_single_model(model: Any, X: pd.DataFrame) -> Tuple[float, float]:
    """Call an unwrapped MLflow PythonModel or a normal estimator and return P50/P90."""
    errors = []
    for call in (
        lambda: model.predict(None, X.copy()),        # unwrapped mlflow.pyfunc.PythonModel
        lambda: model.predict(None, X.copy(), None),  # pyfunc with params slot
        lambda: model.predict(X.copy()),              # sklearn/lightgbm-like wrapper
    ):
        try:
            return _extract_quantiles(call())
        except TypeError as exc:
            errors.append(str(exc))
            continue
    raise ValueError("Unable to call bundle['model'].predict. Errors: " + " | ".join(errors))


def _quantiles(sku: str, dc_id: str) -> Tuple[dict[str, Any], float, float] | None:
    s = state()
    key = f"{sku}|{dc_id}"
    row = s["rows"].get(key)
    if row is None:
        return None

    X = pd.DataFrame([[row[f] for f in s["features"]]], columns=s["features"])
    bundle = s["bundle"]

    if "models" in bundle:
        models = bundle["models"]
        p50_model = models.get(0.5) or models.get("0.5") or models.get("p50") or models.get("q50")
        p90_model = models.get(0.9) or models.get("0.9") or models.get("p90") or models.get("q90")
        if p50_model is None or p90_model is None:
            raise ValueError(f"bundle['models'] must contain P50 and P90 models. Found: {list(models.keys())}")
        p50 = float(np.clip(p50_model.predict(X)[0], 0, None))
        p90 = float(np.clip(p90_model.predict(X)[0], 0, None))
    else:
        p50, p90 = _predict_single_model(bundle["model"], X)
        p50 = float(np.clip(p50, 0, None))
        p90 = float(np.clip(p90, 0, None))

    if p90 < p50:
        p50, p90 = p90, p50
    return row, p50, p90


@app.get("/v2/healthz")
def healthz():
    # Keep this lightweight for Docker, KServe, and SAP AI Core health checks.
    return {"status": "ok"}


@app.get("/v2/readyz")
def readyz():
    try:
        s = state()
        return {
            "status": "ready",
            "skus_loaded": len(s["rows"]),
            "horizon_days": s["horizon"],
            "bundle_keys": list(s["bundle"].keys()),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/v2/skus")
def skus():
    s = state()
    return {"pairs": [k.replace("|", " / ") for k in s["rows"].keys()]}


@app.post("/v2/forecast")
def forecast(req: Req):
    result = _quantiles(req.sku, req.dc_id)
    if result is None:
        raise HTTPException(status_code=404, detail="unknown sku/dc")
    row, p50, p90 = result
    return {
        "sku": req.sku,
        "dc_id": req.dc_id,
        "forecast_p50_7d": round(p50, 1),
        "forecast_p90_7d": round(p90, 1),
        "current_on_hand": round(float(row.get("on_hand_units", 0))),
    }


@app.post("/v2/replenish")
def replenish(req: Req):
    result = _quantiles(req.sku, req.dc_id)
    if result is None:
        raise HTTPException(status_code=404, detail="unknown sku/dc")

    row, p50, p90 = result
    horizon = state()["horizon"]
    on_hand = float(row.get("on_hand_units", 0) or 0)
    lead = float(row.get("lead_time_days", horizon) or horizon)
    moq = float(row.get("moq_units", 0) or 0)
    mult = float(row.get("order_multiple", 1) or 1)
    cap = float(row.get("max_weekly_capacity", math.inf) or math.inf)

    protection_days = lead + REVIEW_DAYS
    daily_demand = p50 / horizon if horizon else 0
    sigma_7 = max(0.0, (p90 - p50)) / 1.2816
    sigma_protection = sigma_7 * math.sqrt(protection_days / horizon) if horizon else 0
    safety_stock = SERVICE_Z * sigma_protection
    order_up_to = daily_demand * protection_days + safety_stock
    raw_qty = max(0.0, order_up_to - on_hand)

    qty = max(raw_qty, moq) if raw_qty > 0 else 0.0
    qty = math.ceil(qty / mult) * mult if mult else qty
    capped = qty > cap
    qty = min(qty, cap)

    days_of_cover = on_hand / daily_demand if daily_demand > 0 else 99
    risk = "HIGH" if days_of_cover < lead else "MEDIUM" if days_of_cover < lead * 1.5 else "LOW"

    return {
        "sku": req.sku,
        "dc_id": req.dc_id,
        "supplier": row.get("supplier_name"),
        "forecast_p50_7d": round(p50, 1),
        "forecast_p90_7d": round(p90, 1),
        "current_on_hand": round(on_hand),
        "lead_time_days": round(lead),
        "days_of_cover": round(days_of_cover, 1),
        "safety_stock": round(safety_stock),
        "recommended_order_qty": int(round(qty)) if math.isfinite(qty) else 0,
        "capacity_constrained": bool(capped),
        "oos_risk": risk,
    }
