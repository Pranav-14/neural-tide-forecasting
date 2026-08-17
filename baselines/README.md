# Student Baselines

This folder contains simple forecasting baselines you can run before building your own model.

The required output format is:

```csv
series_id,timestamp,prediction
```

Run validation baselines:

```bash
DATA_DIR=/path/to/downloaded/hf/dataset
python student/baseline/run_baselines.py \
  --train "$DATA_DIR/train.csv" \
  --forecast-index "$DATA_DIR/forecast_index_validation.csv" \
  --output-dir /tmp/student_baselines
```

Available baselines:

- `naive_last_value`: repeat the final observed target per series.
- `lag24_repeat`: repeat the last 24 observed values.
- `lag168_repeat`: repeat the last 168 observed values.
- `seasonal_mean`: average historical targets by series, day-of-week, and hour-of-day.

These are deliberately simple. Your project model should improve on the provided seasonal-mean baseline.
