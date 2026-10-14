"""Backend interface for analytical, measured, and future research predictors."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
from estimate import Hardware, estimate_workload

@dataclass
class Workload:
    model_params_m: float
    batch_size: int
    input_size: int
    precision_bytes: int = 4

class PredictorBackend(Protocol):
    name: str
    def predict(self, workload: Workload, hardware: Hardware) -> dict: ...

class AnalyticalBackend:
    name = "analytical-baseline"
    def predict(self, workload: Workload, hardware: Hardware) -> dict:
        return estimate_workload(workload.model_params_m, workload.batch_size,
                                 workload.input_size, hardware, workload.precision_bytes)

class MeasuredBackend:
    name = "measured-results"
    def __init__(self, records: list[dict]): self.records = records
    def predict(self, workload: Workload, hardware: Hardware) -> dict:
        matches = [r for r in self.records if r.get("device") == hardware.name
                   and r.get("batch_size") == workload.batch_size
                   and r.get("input_size") == workload.input_size]
        if not matches: raise LookupError("No measured record matches this workload.")
        return {**matches[-1], "backend": self.name}
