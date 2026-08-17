"""Evaluation metrics and helper utilities for time series forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> Dict[str, float]:
    """Compute all official course evaluation metrics.
    
    Metrics:
      - WAPE (Weighted Absolute Percentage Error) - Primary metric
      - MAE (Mean Absolute Error)
      - MSE (Mean Squared Error)
      - RMSE (Root Mean Squared Error)
      - MAPE (Mean Absolute Percentage Error)
      - sMAPE (Symmetric Mean Absolute Percentage Error)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Errors
    abs_err = np.abs(y_true - y_pred)
    sq_err = (y_true - y_pred) ** 2

    # WAPE: sum(|y - y_hat|) / sum(|y|)
    wape = float(np.sum(abs_err) / (np.sum(np.abs(y_true)) + eps))

    # MAE & MSE & RMSE
    mae = float(np.mean(abs_err))
    mse = float(np.mean(sq_err))
    rmse = float(np.sqrt(mse))

    # MAPE: mean(|y - y_hat| / (|y| + eps)) * 100
    mape = float(np.mean(abs_err / (np.abs(y_true) + eps)) * 100.0)

    # sMAPE: mean(2 * |y - y_hat| / (|y| + |y_hat| + eps)) * 100
    smape = float(np.mean(2.0 * abs_err / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100.0)

    return {
        "WAPE": wape,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "sMAPE": smape,
    }


def evaluate_dataframe(
    ground_truth_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    series_col: str = "series_id",
    time_col: str = "timestamp",
    target_col: str = "target",
    pred_col: str = "prediction"
) -> Dict[str, float]:
    """Merge ground truth and predictions on series_id and timestamp, then compute metrics."""
    merged = ground_truth_df[[series_col, time_col, target_col]].merge(
        predictions_df[[series_col, time_col, pred_col]],
        on=[series_col, time_col],
        how="inner"
    )
    if len(merged) != len(ground_truth_df):
        raise ValueError(
            f"Prediction mismatch: Expected {len(ground_truth_df)} rows, but matched {len(merged)} rows."
        )
    return compute_metrics(merged[target_col].to_numpy(), merged[pred_col].to_numpy())
