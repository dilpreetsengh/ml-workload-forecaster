"""Shared schema and feature preparation for benchmark records."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

FEATURES = ["batch_size", "input_size", "model_params_m", "device_memory_gb"]
TARGETS = ["mean_latency_ms", "throughput_samples_per_second", "peak_memory_mb"]

@dataclass
class WorkloadSpec:
    model: str
    device: str
    batch_size: int
    input_size: int
    precision: str = "float32"
    mode: str = "inference"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def numeric_features(record: dict[str, Any]) -> dict[str, float]:
    return {
        "batch_size": float(record.get("batch_size", 1)),
        "input_size": float(record.get("input_size", 128)),
        "model_params_m": float(record.get("model_params_m", 0)),
        "device_memory_gb": float(record.get("device_memory_gb", 0)),
    }
