"""Evaluation metrics for measured versus predicted workload behavior."""
from __future__ import annotations
import numpy as np

def regression_metrics(actual, predicted) -> dict[str, float]:
    y=np.asarray(actual,dtype=float); p=np.asarray(predicted,dtype=float)
    if y.shape != p.shape or y.size == 0: raise ValueError("actual and predicted must have equal nonzero shape")
    error=np.abs(y-p)
    return {"mae":float(error.mean()),"rmse":float(np.sqrt(np.mean((y-p)**2))),
            "mape_percent":float(np.mean(error/np.maximum(np.abs(y),1e-8))*100),
            "max_absolute_error":float(error.max())}
