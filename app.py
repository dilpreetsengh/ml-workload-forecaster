"""Minimal dashboard for inspecting collected benchmark records."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ML Workload Forecaster", layout="wide")
st.title("ML Workload Forecaster")
st.caption("Measured benchmark results — predictions will be added after enough data is collected.")
path = Path("data/raw/benchmark_results.jsonl")
if not path.exists():
    st.info("No benchmark data yet. Run: python src/benchmark.py --device mps")
else:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    frame = pd.DataFrame(rows)
    st.metric("Successful runs", int((frame.status == "success").sum()))
    st.dataframe(frame, use_container_width=True)
    if "mean_latency_ms" in frame:
        st.line_chart(frame[["mean_latency_ms", "p95_latency_ms"]])
