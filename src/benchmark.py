"""Run reproducible PyTorch inference benchmarks on CPU, Apple MPS, or CUDA."""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class BenchmarkResult:
    model: str
    device: str
    dtype: str
    batch_size: int
    input_size: int
    warmup_runs: int
    measured_runs: int
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    throughput_samples_per_second: float
    peak_memory_mb: float | None
    status: str
    error: str | None = None


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def reset_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def peak_memory_mb(device: torch.device) -> float | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return torch.mps.current_allocated_memory() / (1024 ** 2)
    return None


def benchmark(model_name: str, device: torch.device, batch_size: int,
              input_size: int, warmup_runs: int, measured_runs: int) -> BenchmarkResult:
    result_args = dict(model=model_name, device=str(device), dtype="float32",
                       batch_size=batch_size, input_size=input_size,
                       warmup_runs=warmup_runs, measured_runs=measured_runs)
    try:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).eval().to(device)
        text = ["benchmark workload input"] * batch_size
        inputs = tokenizer(text, return_tensors="pt", padding=True,
                           truncation=True, max_length=input_size)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            for _ in range(warmup_runs):
                model(**inputs)
        synchronize(device)
        reset_memory(device)

        timings = []
        with torch.inference_mode():
            for _ in range(measured_runs):
                start = time.perf_counter()
                model(**inputs)
                synchronize(device)
                timings.append((time.perf_counter() - start) * 1000)

        values = np.asarray(timings, dtype=np.float64)
        mean_ms = float(values.mean())
        return BenchmarkResult(**result_args,
            mean_latency_ms=mean_ms,
            p50_latency_ms=float(np.percentile(values, 50)),
            p95_latency_ms=float(np.percentile(values, 95)),
            throughput_samples_per_second=float(batch_size / (mean_ms / 1000)),
            peak_memory_mb=peak_memory_mb(device), status="success")
    except Exception as exc:
        return BenchmarkResult(**result_args, mean_latency_ms=0.0,
            p50_latency_ms=0.0, p95_latency_ms=0.0,
            throughput_samples_per_second=0.0, peak_memory_mb=None,
            status="error", error=f"{type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="prajjwal1/bert-tiny")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--measured-runs", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("data/raw/benchmark_results.jsonl"))
    args = parser.parse_args()

    device = select_device(args.device)
    result = benchmark(args.model, device, args.batch_size, args.input_size,
                       args.warmup_runs, args.measured_runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({**asdict(result), "platform": platform.platform()}) + "\n")
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
