"""FastAPI service: loads the serialized sklearn pipeline once at startup."""
import json
import os
import sys
import threading
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))                      # so the pickle can import src.features

MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "model.joblib"))
INFO_PATH = Path(os.getenv("MODEL_INFO_PATH", ROOT / "models" / "model_info.json"))

STATE = {"model": None, "info": {}, "error": None, "started": time.time(), "loaded_at": None}


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.requests = Counter()
        self.status = Counter()
        self.predictions = Counter()
        self.errors = 0
        self.latency_sum = 0.0
        self.latency_n = 0
        self.latency_max = 0.0
        self.prob_sum = 0.0

    def record(self, path, status, seconds):
        with self.lock:
            self.requests[path] += 1
            self.status[str(status)] += 1
            self.latency_sum += seconds
            self.latency_n += 1
            self.latency_max = max(self.latency_max, seconds)
            if status >= 400:
                self.errors += 1

    def record_prediction(self, label, prob):
        with self.lock:
            self.predictions[label] += 1
            self.prob_sum += prob

    def snapshot(self):
        with self.lock:
            n = sum(self.predictions.values())
            return {"uptime_seconds": round(time.time() - STATE["started"], 1),
                    "requests_total": sum(self.requests.values()),
                    "requests_by_endpoint": dict(self.requests),
                    "responses_by_status": dict(self.status),
                    "errors_total": self.errors,
                    "predictions_total": n,
                    "predictions_by_class": dict(self.predictions),
                    "avg_survival_probability": round(self.prob_sum / n, 4) if n else None,
                    "avg_latency_ms": round(1000 * self.latency_sum / self.latency_n, 3) if self.latency_n else 0.0,
                    "max_latency_ms": round(1000 * self.latency_max, 3),
                    "model_version": STATE["info"].get("version")}


METRICS = Metrics()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        STATE["model"] = joblib.load(MODEL_PATH)             # load once at startup
        STATE["info"] = json.loads(INFO_PATH.read_text(encoding="utf-8")) if INFO_PATH.exists() else {}
        STATE["loaded_at"] = time.time()
        STATE["error"] = None
    except Exception as exc:                                  # keep serving /health with 503 instead of crashing
        STATE["model"], STATE["error"] = None, f"{type(exc).__name__}: {exc}"
    yield


app = FastAPI(title="Titanic Survival API", version="1.0.0", lifespan=lifespan,
              description="Decision Tree pipeline (clean -> engineer -> One-Hot/MinMax -> Chi-Square -> DecisionTree)")


@app.middleware("http")
async def track_requests(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        METRICS.record(request.url.path, 500, time.perf_counter() - start)
        raise
    METRICS.record(request.url.path, response.status_code, time.perf_counter() - start)
    return response


class Passenger(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "Pclass": 1, "Name": "Futrelle, Mrs. Jacques Heath", "Sex": "female", "Age": 35,
        "SibSp": 1, "Parch": 0, "Fare": 53.1, "Embarked": "S"}})
    Pclass: Literal[1, 2, 3]
    Sex: Literal["male", "female"]
    Age: Optional[float] = Field(None, ge=0, le=120, description="missing -> training median")
    SibSp: int = Field(0, ge=0, le=20)
    Parch: int = Field(0, ge=0, le=20)
    Fare: Optional[float] = Field(None, ge=0, description="missing -> training median")
    Embarked: Optional[Literal["C", "Q", "S"]] = Field(None, description="missing -> training mode")
    Name: Optional[str] = Field(None, description="used only to extract the Title (Mr/Mrs/Miss/Master/Rare)")


class Prediction(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    prediction: int
    label: str
    survival_probability: float
    model_version: Optional[str] = None
    latency_ms: float


def _require_model():
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {STATE['error']}")
    return STATE["model"]


@app.get("/health", tags=["monitoring"])
def health():
    ok = STATE["model"] is not None
    body = {"status": "ok" if ok else "unavailable", "model_loaded": ok,
            "uptime_seconds": round(time.time() - STATE["started"], 1), "error": STATE["error"]}
    if not ok:
        raise HTTPException(status_code=503, detail=body)
    return body


@app.get("/model-info", tags=["monitoring"])
def model_info():
    model = _require_model()
    return {**STATE["info"],
            "pipeline_steps": [name for name, _ in model.steps],
            "model_path": str(MODEL_PATH.name),
            "loaded_at": pd.Timestamp(STATE["loaded_at"], unit="s", tz="UTC").isoformat()}


@app.post("/predict", response_model=Prediction, tags=["prediction"])
def predict(p: Passenger):
    model = _require_model()
    t0 = time.perf_counter()
    row = p.model_dump()
    if not row.get("Name"):                                   # placeholder so the Title step still works
        row["Name"] = "Doe, Mrs. Jane" if row["Sex"] == "female" else "Doe, Mr. John"
    X = pd.DataFrame([row])
    try:
        proba = float(model.predict_proba(X)[0, 1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")
    pred = int(proba >= 0.5)
    label = "Survived" if pred else "Did not survive"
    METRICS.record_prediction(label, proba)
    return Prediction(prediction=pred, label=label, survival_probability=round(proba, 4),
                      model_version=STATE["info"].get("version"),
                      latency_ms=round((time.perf_counter() - t0) * 1000, 3))


@app.get("/metrics", tags=["monitoring"])
def metrics(format: str = Query("json", pattern="^(json|prometheus)$")):
    snap = METRICS.snapshot()
    if format == "json":
        return snap
    lines = [f"titanic_uptime_seconds {snap['uptime_seconds']}",
             f"titanic_requests_total {snap['requests_total']}",
             f"titanic_errors_total {snap['errors_total']}",
             f"titanic_predictions_total {snap['predictions_total']}",
             f"titanic_avg_latency_ms {snap['avg_latency_ms']}",
             f"titanic_max_latency_ms {snap['max_latency_ms']}"]
    lines += [f'titanic_requests_by_endpoint{{path="{k}"}} {v}' for k, v in snap["requests_by_endpoint"].items()]
    lines += [f'titanic_predictions_by_class{{label="{k}"}} {v}' for k, v in snap["predictions_by_class"].items()]
    return PlainTextResponse("\n".join(lines) + "\n")
