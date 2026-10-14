# ML Workload Forecaster

A hardware-aware system for measuring and predicting machine-learning workload performance.

The project aims to estimate latency, throughput, peak accelerator memory, workload feasibility, and estimated execution cost before a workload is deployed.

## Current scope

- PyTorch workloads
- Transformer and vision models
- Batch-size and input-size sweeps
- Measured latency, throughput, and peak memory
- Analytical baselines
- Machine-learning performance predictors
- Reproducible experiments

## Planned architecture

1. Workload specification
2. Benchmark runner
3. Measurement dataset
4. Analytical baseline
5. ML predictor
6. Evaluation and error analysis
7. Recommendation API/dashboard

## Project status

Early setup. No performance claims have been made yet. All reported results will come from reproducible experiments in this repository.

## Research foundations

This project studies ideas from NeuSight, GPUMemNet, DeepView.Predict, PipeWeave, and llm-analysis. Their licenses and citations will be preserved where code or research ideas are used.

## License

To be decided after the initial implementation and dependency review.
