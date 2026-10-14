"""End-to-end analytical estimator used before measured data is available."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any
from analytical import estimate_transformer_memory_mb

@dataclass
class Hardware:
    name: str
    memory_gb: float
    hourly_cost: float
    relative_speed: float

HARDWARE = [
    Hardware("Apple Silicon MPS", 16.0, 0.20, 0.55),
    Hardware("NVIDIA T4", 16.0, 0.35, 0.75),
    Hardware("NVIDIA L4", 24.0, 0.80, 1.00),
    Hardware("NVIDIA A100", 40.0, 2.50, 2.50),
]

def estimate_workload(model_params_m: float, batch_size: int, input_size: int,
                      hardware: Hardware, precision_bytes: int = 4) -> dict[str, Any]:
    memory_mb = estimate_transformer_memory_mb(model_params_m, batch_size, input_size, precision_bytes)
    # Transparent rough baseline: compute grows with parameters, batch, and sequence length.
    latency_ms = max(0.1, (model_params_m * batch_size * input_size / 1000) / hardware.relative_speed)
    throughput = batch_size / (latency_ms / 1000)
    return {**asdict(hardware), "predicted_memory_mb": round(memory_mb, 2),
            "predicted_latency_ms": round(latency_ms, 2),
            "predicted_throughput": round(throughput, 2),
            "fits_memory": memory_mb <= hardware.memory_gb * 1024,
            "estimated_hourly_cost": hardware.hourly_cost}

def recommend(model_params_m: float, batch_size: int, input_size: int,
              target_latency_ms: float = 1000.0) -> list[dict[str, Any]]:
    rows = [estimate_workload(model_params_m, batch_size, input_size, h) for h in HARDWARE]
    return [r for r in sorted(rows, key=lambda row: row["estimated_hourly_cost"])
            if r["fits_memory"] and r["predicted_latency_ms"] <= target_latency_ms]
