# Experiment Protocol

1. Use a fixed model and tokenizer.
2. Run warm-up iterations before recording measurements.
3. Synchronize the selected device before stopping each timer.
4. Repeat each configuration with multiple measured iterations.
5. Sweep batch size and input length rather than recording one cherry-picked result.
6. Store raw results before creating plots or training predictors.
7. Keep hardware, software, backend, and precision metadata with every record.
8. Split evaluation data by configuration and report MAE, RMSE, and worst-case error.

Results must be reproducible and must not be inflated or manually edited for presentation.
