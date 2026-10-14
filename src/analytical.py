"""Simple explainable estimates used as a baseline, not production guarantees."""
from __future__ import annotations

def estimate_transformer_memory_mb(params_m: float, batch_size: int, input_size: int, bytes_per_value: int = 4) -> float:
    # Approximate weights plus activations; measured models must replace this baseline.
    weights = params_m * 1_000_000 * bytes_per_value
    activations = batch_size * input_size * max(1.0, params_m / 10.0) * bytes_per_value
    return (weights + activations) / (1024 ** 2)

def estimate_throughput(samples: int, latency_ms: float) -> float:
    return samples / (latency_ms / 1000.0) if latency_ms > 0 else 0.0
