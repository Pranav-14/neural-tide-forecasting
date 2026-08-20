"""24-step iterative rolling inference engine for 336-hour forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.dataset import (
    ALL_COVARIATE_COLUMNS,
    FORECAST_COVARIATES,
    FUTURE_COVARIATE_COLUMNS,
    Preprocessor,
)


def run_iterative_rolling_forecast(
    model: nn.Module,
    train_df: pd.DataFrame,
    future_input_df: pd.DataFrame,
    forecast_index_df: pd.DataFrame,
    preprocessor_means: Optional[Dict[str, float]] = None,
    lookback_len: int = 168,
    horizon: int = 24,
    device: torch.device = torch.device("cpu"),
) -> pd.DataFrame:
    """Generate multi-step forecasts via 24-step recursive rollout across 336 hours.
    
    Args:
        model: Trained TiDE/ForecastModel instance.
        train_df: Historical training dataframe containing target and past features.
        future_input_df: Future covariates dataframe (validation_input.csv or test_input.csv).
        forecast_index_df: Target rows requiring predictions (forecast_index_*.csv).
        preprocessor_means: Column mean dictionary fitted during training.
        lookback_len: Lookback window length (default: 168).
        horizon: Step block length (default: 24).
        device: Torch computation device.
    
    Returns:
        DataFrame matching forecast_index_df schema: [series_id, timestamp, prediction].
    """
    model.eval()
    model.to(device)

    # 1. Feature columns definition
    past_cov_cols = list(ALL_COVARIATE_COLUMNS)
    future_cov_cols = list(FUTURE_COVARIATE_COLUMNS)
    for col in FORECAST_COVARIATES + ["shock_risk"]:
        mask_col = f"{col}_isnan"
        past_cov_cols.append(mask_col)
        if col in FUTURE_COVARIATE_COLUMNS:
            future_cov_cols.append(mask_col)

    # 2. Impute and prepare historical data
    preprocessor = Preprocessor(add_missing_indicator=True)
    if preprocessor_means is not None:
        preprocessor.column_means = preprocessor_means
        preprocessor.is_fitted = True
    else:
        preprocessor.fit(train_df)

    train_proc = preprocessor.transform(train_df)
    future_proc = preprocessor.transform(future_input_df)

    # Sort data
    train_proc["timestamp"] = pd.to_datetime(train_proc["timestamp"])
    train_proc = train_proc.sort_values(["series_id", "timestamp"]).reset_index(drop=True)

    future_proc["timestamp"] = pd.to_datetime(future_proc["timestamp"])
    future_proc = future_proc.sort_values(["series_id", "timestamp"]).reset_index(drop=True)

    # 3. Process each series
    predictions_list: List[pd.DataFrame] = []

    unique_series = forecast_index_df["series_id"].unique()

    with torch.no_grad():
        for series_id in unique_series:
            s_train = train_proc[train_proc["series_id"] == series_id]
            s_future = future_proc[future_proc["series_id"] == series_id]
            s_index = forecast_index_df[forecast_index_df["series_id"] == series_id].copy()

            if len(s_train) < lookback_len:
                raise ValueError(f"Series {series_id} has {len(s_train)} history rows, requires at least {lookback_len}.")

            # Initial lookback buffers
            target_buffer = list(s_train["target"].iloc[-lookback_len:].to_numpy(dtype=np.float32))
            past_cov_buffer = list(s_train[past_cov_cols].iloc[-lookback_len:].to_numpy(dtype=np.float32))

            # Future covariates matrix
            future_cov_matrix = s_future[future_cov_cols].to_numpy(dtype=np.float32)
            future_past_cov_matrix = s_future[past_cov_cols].to_numpy(dtype=np.float32)

            n_future_steps = len(s_future)
            series_preds: List[float] = []

            for start_step in range(0, n_future_steps, horizon):
                end_step = min(start_step + horizon, n_future_steps)
                curr_horizon = end_step - start_step

                # Prepare input tensors
                x_past_target = torch.from_numpy(np.array(target_buffer[-lookback_len:], dtype=np.float32)).unsqueeze(0).to(device)
                x_past_cov = torch.from_numpy(np.array(past_cov_buffer[-lookback_len:], dtype=np.float32)).unsqueeze(0).to(device)

                # Future covariates slice
                if curr_horizon == horizon:
                    x_future_cov = torch.tensor([future_cov_matrix[start_step:end_step]], dtype=torch.float32, device=device)
                else:
                    # Pad if last block is shorter
                    pad_len = horizon - curr_horizon
                    cov_slice = future_cov_matrix[start_step:end_step]
                    padded_cov = np.pad(cov_slice, ((0, pad_len), (0, 0)), mode="edge")
                    x_future_cov = torch.tensor([padded_cov], dtype=torch.float32, device=device)

                # Predict 24 steps
                y_pred_tensor = model(x_past_target, x_past_cov, x_future_cov)
                y_pred_np = y_pred_tensor.cpu().numpy()[0][:curr_horizon]

                # Clamp negative predictions if target is strictly positive operational load
                y_pred_np = np.clip(y_pred_np, a_min=0.0, a_max=None)

                series_preds.extend(y_pred_np.tolist())

                # Update rolling history buffers
                target_buffer.extend(y_pred_np.tolist())
                past_cov_buffer.extend(future_past_cov_matrix[start_step:end_step])

            # Align with forecast index
            s_index["prediction"] = series_preds[: len(s_index)]
            predictions_list.append(s_index[["series_id", "timestamp", "prediction"]])

    predictions_df = pd.concat(predictions_list, ignore_index=True)
    return predictions_df
