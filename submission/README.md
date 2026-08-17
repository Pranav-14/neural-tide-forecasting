# Final Submission Template

Use this folder as the starting point for `final_submission.zip`.

For simple runnable baseline forecasts, see `student/baseline/`.

## Required Files

- `predict.py`: inference entrypoint.
- `requirements.txt`: Python dependencies needed for inference.
- `checkpoint.pt`: trained model weights or checkpoint.
- `src/`: optional package code.

## Required Command

Your submission must support:

```bash
python predict.py --input_dir /data/input --output_file /output/predictions.csv --checkpoint /submission/checkpoint.pt
```

## Output Format

Required schema:

```csv
series_id,timestamp,prediction
```

Your output must cover every row in the provided forecast index. Public validation uses `forecast_index_validation.csv`; private evaluation uses `forecast_index_test.csv` in the input directory. In the current benchmark design, the 24-hour forecast horizon is a rollout block length. The validation and private test forecast indices each contain 336 hourly timestamps per series.

The leaderboard reports MAE, MSE, RMSE, MAPE, sMAPE, and WAPE. Lower is better for all metrics.

The model name is entered in the leaderboard Space when uploading validation predictions or the final archive. It is not part of `predictions.csv`. Use the same model name for the final archive as for the validation row that represents that checkpoint.

## Packaging

From inside this directory, create the archive with:

```bash
zip -r final_submission.zip predict.py requirements.txt checkpoint.pt src
```

Do not include training data, private data, virtual environments, caches, or large unused artifacts.
