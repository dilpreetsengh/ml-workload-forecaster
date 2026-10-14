"""Generate clearly labeled synthetic records for testing the software pipeline."""
from __future__ import annotations
import json
from pathlib import Path
import random

def generate(path: Path, rows: int = 100, seed: int = 42) -> None:
    random.seed(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for _ in range(rows):
            batch = random.choice([1, 2, 4, 8])
            length = random.choice([32, 64, 128, 256])
            params = random.choice([4.4, 14.5, 67.0, 110.0])
            latency = params * batch * length / 400 + random.uniform(-2, 2)
            row = {"model":"synthetic-demo-not-measured", "device":"synthetic-demo",
                   "dtype":"float32", "batch_size":batch, "input_size":length,
                   "model_params_m":params, "device_memory_gb":16.0,
                   "warmup_runs":0, "measured_runs":0,
                   "mean_latency_ms":max(.1, latency), "p50_latency_ms":max(.1, latency*.98),
                   "p95_latency_ms":max(.1, latency*1.08),
                   "throughput_samples_per_second":batch/(max(.1, latency)/1000),
                   "peak_memory_mb":params*4+batch*length*.02, "status":"success",
                   "data_source":"synthetic_demo_only"}
            handle.write(json.dumps(row) + "\n")

if __name__ == "__main__":
    generate(Path("data/raw/demo_results.jsonl"))
    print("Wrote synthetic demo data. Do not report it as measured performance.")
