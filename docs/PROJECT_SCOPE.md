# Initial Project Scope

## Problem

Before running a machine-learning workload, engineers often need to estimate whether a hardware configuration will fit in memory, how fast it will run, and how much it may cost.

## Initial question

Given a PyTorch model, hardware description, batch size, input size, precision, and inference mode, can we predict latency, throughput, and peak memory from benchmark data?

## First supported experiment

The first experiment will benchmark inference workloads while varying:

- model
- batch size
- input size or sequence length
- precision
- warm-up count
- measured hardware

## Metrics

- mean latency
- p50 latency
- p95 latency
- throughput
- peak allocated memory
- peak reserved memory
- out-of-memory status

## Evaluation rule

The test set must contain configurations that were not used to train the predictor. Results must report both average and worst-case error. Predictions will not be presented as production guarantees.

## Non-goals for the MVP

- training large language models
- distributed multi-node training
- custom CUDA kernel optimization
- live cloud-price scraping
- claiming generalization to every accelerator
