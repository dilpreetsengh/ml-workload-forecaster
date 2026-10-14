"""Train small baseline predictors from benchmark JSONL records."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import joblib
from schema import FEATURES, TARGETS

def load_records(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    frame = pd.DataFrame(rows)
    if frame.empty: raise ValueError("No benchmark records found.")
    return frame[frame.status == "success"].copy()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw/benchmark_results.jsonl"))
    parser.add_argument("--model-out", type=Path, default=Path("models/performance.joblib"))
    args = parser.parse_args()
    frame = load_records(args.input)
    missing = [c for c in FEATURES + TARGETS if c not in frame]
    if missing: raise ValueError(f"Missing columns: {missing}. Add metadata before training.")
    if len(frame) < 10: raise ValueError("Collect at least 10 successful benchmark records first.")
    x_train, x_test, y_train, y_test = train_test_split(frame[FEATURES], frame[TARGETS], test_size=.2, random_state=42)
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=2))
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    metrics = {target: {"mae": float(mean_absolute_error(y_test[target], pred[:, i])), "rmse": float(mean_squared_error(y_test[target], pred[:, i]) ** .5)} for i, target in enumerate(TARGETS)}
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "targets": TARGETS, "metrics": metrics}, args.model_out)
    print(json.dumps({"records": len(frame), "metrics": metrics}, indent=2))

if __name__ == "__main__": main()
