# Optional research backends

This project keeps external research systems optional because they target different hardware and environments.

## NeuSight

Repository: https://github.com/scai-tech/NeuSight

License: MIT

Use case: learned deep-learning performance prediction across GPUs. Integrate through a backend adapter when running a compatible CUDA/Linux environment. Preserve the MIT copyright and license if source files are incorporated.

## PipeWeave

Repository: https://github.com/zksainx/pipeweave

License: Apache-2.0

Use case: analytical plus learned transformer latency prediction. Use its operator-level feature design as an optional NVIDIA backend. Preserve Apache license and NOTICE requirements for any copied files.

## GPUMemNet

Repository: https://github.com/itu-rad/GPUMemNet

Use case: GPU memory workload generation and monitoring methodology. Its dataset conventions inform our benchmark schema; no source code is copied into the core package.

## DeepView.Predict

Repository: https://github.com/CentML/DeepView.Predict

License: Apache-2.0

Use case: runtime prediction for PyTorch training iterations. It is a candidate future backend for training workloads.

## Integration rule

External code must be isolated behind an adapter, retain its original license and copyright notices, and include a note describing local modifications. Research inspiration alone is documented in `docs/ATTRIBUTIONS.md`.
