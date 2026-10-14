# Usage guide

## Analytical estimator

```bash
python src/predict.py --params-m 14.5 --batch-size 4 --input-size 128
```

## Synthetic software demo

The demo validates the pipeline without pretending to be measured hardware data:

```bash
python src/demo_data.py
python -m streamlit run app.py
```

Synthetic records are labeled `synthetic_demo_only` and must never be used as resume performance claims.

## Real benchmark

```bash
python src/benchmark.py --device mps --batch-size 1 --input-size 128
```

Real benchmark records can later train the predictor:

```bash
pip install -e '.[modeling]'
python src/train_predictor.py
```
