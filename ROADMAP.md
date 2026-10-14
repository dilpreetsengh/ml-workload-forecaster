# Roadmap

## Phase 1 — Reproducible benchmark MVP

- [ ] Define a versioned workload schema
- [ ] Add environment and hardware metadata collection
- [ ] Implement PyTorch inference benchmark runner
- [ ] Measure warm-up, latency, throughput, and peak memory
- [ ] Save results as CSV and JSON
- [ ] Add one transformer and one vision workload
- [ ] Add basic plots and summary tables

## Phase 2 — Prediction

- [ ] Implement analytical FLOPs and memory baseline
- [ ] Create train/validation/test split by workload configuration
- [ ] Train a tree-based latency predictor
- [ ] Train a memory predictor
- [ ] Report MAE, RMSE, MAPE, and worst-case error
- [ ] Compare predictions with measured results

## Phase 3 — Recommendations

- [ ] Add feasibility and out-of-memory checks
- [ ] Add target-latency constraints
- [ ] Add estimated cost calculations
- [ ] Recommend the lowest-cost configuration meeting constraints

## Phase 4 — Product layer

- [ ] Add FastAPI endpoints
- [ ] Add Streamlit dashboard
- [ ] Add experiment reproducibility instructions
- [ ] Add tests and continuous integration
- [ ] Document limitations and research inspirations
