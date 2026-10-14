"""Cost and constraint utilities for workload planning."""
from __future__ import annotations

def gpu_hours(runtime_seconds: float, replicas: int = 1) -> float:
    if runtime_seconds < 0 or replicas < 1:
        raise ValueError("runtime_seconds must be non-negative and replicas must be positive")
    return runtime_seconds * replicas / 3600.0

def estimated_cost(runtime_seconds: float, hourly_rate: float, replicas: int = 1) -> float:
    if hourly_rate < 0:
        raise ValueError("hourly_rate must be non-negative")
    return gpu_hours(runtime_seconds, replicas) * hourly_rate

def meets_constraints(latency_ms: float, memory_mb: float, target_latency_ms: float,
                     available_memory_mb: float) -> bool:
    return latency_ms <= target_latency_ms and memory_mb <= available_memory_mb
