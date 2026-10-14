"""Choose feasible configurations from predicted performance and cost."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class HardwareOption:
    name: str
    memory_gb: float
    hourly_cost: float

def recommend(options: list[HardwareOption], predicted_memory_mb: float, predicted_latency_ms: float, target_latency_ms: float) -> list[dict]:
    candidates = [o for o in options if predicted_memory_mb <= o.memory_gb * 1024 and predicted_latency_ms <= target_latency_ms]
    return [dict(name=o.name, memory_gb=o.memory_gb, hourly_cost=o.hourly_cost) for o in sorted(candidates, key=lambda x: x.hourly_cost)]
