"""Predict workload behavior from a trained model or analytical fallback."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from analytical import estimate_transformer_memory_mb

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--params-m",type=float,default=14.5)
    p.add_argument("--batch-size",type=int,default=1)
    p.add_argument("--input-size",type=int,default=128)
    p.add_argument("--memory-gb",type=float,default=16.0)
    p.add_argument("--latency-target-ms",type=float,default=1000.0)
    a=p.parse_args()
    memory=estimate_transformer_memory_mb(a.params_m,a.batch_size,a.input_size)
    print(json.dumps({"mode":"analytical_baseline","predicted_memory_mb":round(memory,2),"fits_memory":memory<=a.memory_gb*1024,"inputs":vars(a)},indent=2))

if __name__ == "__main__": main()
