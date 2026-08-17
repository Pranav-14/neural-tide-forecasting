"""Simple student-facing forecasting baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def naive_last_value_forecast(
    train_frame: pd.DataFrame,
    forecast_index: pd.DataFrame,
    *,
    series_col: str = "series_id",
    time_col: str = "timestamp",
    target_col: str = "target",
    prediction_col: str = "prediction",
) -> pd.DataFrame:
    """Repeat each series' last observed target value across the forecast index."""
    last_values = (
        train_frame.sort_values(time_col)
        .groupby(series_col, as_index=False)[target_col]
        .last()
        .rename(columns={target_col: prediction_col})
    )
    return forecast_index[[series_col, time_col]].merge(last_values, on=series_col, how="left")


def repeat_last_period_forecast(
    train_frame: pd.DataFrame,
    forecast_index: pd.DataFrame,
    period: int,
    *,
    series_col: str = "series_id",
    time_col: str = "timestamp",
    target_col: str = "target",
    prediction_col: str = "prediction",
) -> pd.DataFrame:
    """Repeat the last observed period for each series across the forecast index."""
    rows = []
    for series_id, index_part in forecast_index.groupby(series_col, sort=False):
        history = (
            train_frame.loc[train_frame[series_col].eq(series_id)]
            .sort_values(time_col)
            .tail(period)[target_col]
            .to_numpy()
        )
        if len(history) == 0:
            raise ValueError(f"No history found for series {series_id!r}.")
        repeats = int(np.ceil(len(index_part) / len(history)))
        result = index_part[[series_col, time_col]].copy()
        result[prediction_col] = np.tile(history, repeats)[: len(index_part)]
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def add_seasonal_keys(frame: pd.DataFrame, *, time_col: str = "timestamp") -> pd.DataFrame:
    """Add hour-of-day and day-of-week columns used by seasonal baselines."""
    result = frame.copy()
    timestamps = pd.to_datetime(result[time_col])
    result["_hour"] = timestamps.dt.hour
    result["_dayofweek"] = timestamps.dt.dayofweek
    return result


def seasonal_mean_forecast(
    train_frame: pd.DataFrame,
    forecast_index: pd.DataFrame,
    *,
    series_col: str = "series_id",
    time_col: str = "timestamp",
    target_col: str = "target",
    prediction_col: str = "prediction",
) -> pd.DataFrame:
    """Forecast by per-series day/hour means with series and global fallbacks."""
    train_keyed = add_seasonal_keys(train_frame, time_col=time_col)
    forecast_keyed = add_seasonal_keys(forecast_index, time_col=time_col)

    seasonal = (
        train_keyed.groupby([series_col, "_dayofweek", "_hour"], as_index=False)[target_col]
        .mean()
        .rename(columns={target_col: prediction_col})
    )
    result = forecast_keyed[[series_col, time_col, "_dayofweek", "_hour"]].merge(
        seasonal,
        on=[series_col, "_dayofweek", "_hour"],
        how="left",
    )

    series_means = (
        train_frame.groupby(series_col, as_index=False)[target_col]
        .mean()
        .rename(columns={target_col: "_series_mean"})
    )
    result = result.merge(series_means, on=series_col, how="left")
    global_mean = float(train_frame[target_col].mean())
    result[prediction_col] = result[prediction_col].fillna(result["_series_mean"]).fillna(global_mean)
    return result[[series_col, time_col, prediction_col]]


def make_all_baselines(train_frame: pd.DataFrame, forecast_index: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build every student-facing baseline prediction frame."""
    return {
        "naive_last_value": naive_last_value_forecast(train_frame, forecast_index),
        "lag24_repeat": repeat_last_period_forecast(train_frame, forecast_index, period=24),
        "lag168_repeat": repeat_last_period_forecast(train_frame, forecast_index, period=168),
        "seasonal_mean": seasonal_mean_forecast(train_frame, forecast_index),
    }


def write_baselines(train_path: Path, forecast_index_path: Path, output_dir: Path) -> None:
    """Write all baseline prediction CSVs for a train/index pair."""
    train = pd.read_csv(train_path)
    forecast_index = pd.read_csv(forecast_index_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, predictions in make_all_baselines(train, forecast_index).items():
        predictions.to_csv(output_dir / f"{name}.csv", index=False)
