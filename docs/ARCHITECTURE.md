# Architecture

## Runtime flow

1. A client submits a workload specification.
2. The API validates the request.
3. A prediction backend estimates latency, throughput, and memory.
4. The feasibility layer removes configurations that do not fit memory or latency constraints.
5. The recommendation layer ranks remaining options by cost.
6. A future measured-data backend can replace or calibrate the analytical backend without changing the API.

## Backend contract

All predictors receive a normalized `Workload` and a `Hardware` record and return a common prediction dictionary. This makes research implementations such as NeuSight or PipeWeave optional adapters instead of hard-coded dependencies.

## Current limitations

The analytical backend is a transparent baseline. Hardware costs and relative-speed values are illustrative configuration data, not live provider prices. Production deployment requires measured calibration and verified pricing sources.
