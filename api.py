"""HTTP API for workload estimation and hardware recommendations."""
from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from estimate import HARDWARE
from backends import AnalyticalBackend, Workload

app = FastAPI(title="ML Workload Forecaster API", version="0.1.0")
backend = AnalyticalBackend()

class WorkloadRequest(BaseModel):
    model_params_m: float = Field(gt=0, description="Model parameters in millions")
    batch_size: int = Field(default=1, gt=0)
    input_size: int = Field(default=128, gt=0)
    precision_bytes: int = Field(default=4, ge=1, le=4)
    target_latency_ms: float = Field(default=1000, gt=0)

@app.get("/health")
def health():
    return {"status": "ok", "backend": backend.name}

@app.post("/v1/predict")
def predict(request: WorkloadRequest):
    workload = Workload(**request.model_dump(exclude={"target_latency_ms"}))
    predictions = []
    for hardware in HARDWARE:
        row = backend.predict(workload, hardware)
        row["meets_target"] = row["predicted_latency_ms"] <= request.target_latency_ms
        predictions.append(row)
    return {"backend": backend.name, "workload": workload.__dict__, "predictions": predictions}

@app.post("/v1/recommend")
def recommend(request: WorkloadRequest):
    result = predict(request)
    candidates = [x for x in result["predictions"] if x["fits_memory"] and x["meets_target"]]
    if not candidates: raise HTTPException(422, "No listed hardware meets the constraints.")
    return {"recommendation": min(candidates, key=lambda x: x["estimated_hourly_cost"]),
            "candidates": sorted(candidates, key=lambda x: x["estimated_hourly_cost"])}
