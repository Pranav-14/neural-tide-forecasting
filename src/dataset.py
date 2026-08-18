"""Dataset and DataLoader pipeline for multivariate time-series forecasting."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# Define canonical feature categories based on benchmark metadata
TEMPORAL_COVARIATES = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "zone_sin",
    "zone_cos",
]

FORECAST_COVARIATES = [
    "demand_forecast",
    "staffing_forecast",
    "queue_pressure_forecast",
    "network_pressure_forecast",
    "event_load_forecast",
    "service_irregularity_risk_forecast",
    "throughput_disruption_risk_forecast",
    "upstream_quality_forecast",
    "unit_reliability_forecast",
]

OPERATIONAL_COVARIATES = [
    "trend",
    "nominal_capacity",
    "workload_intensity",
    "promotion_intensity",
    "shock_risk",
    "maintenance_known",
]

# Future covariates are features available across both past and future forecast horizon
FUTURE_COVARIATE_COLUMNS = TEMPORAL_COVARIATES + FORECAST_COVARIATES

# Past covariates include all features (temporal, forecast, and operational)
ALL_COVARIATE_COLUMNS = TEMPORAL_COVARIATES + FORECAST_COVARIATES + OPERATIONAL_COVARIATES


class Preprocessor:
    """Handles missing value imputation and feature normalization without data leakage."""

    def __init__(self, add_missing_indicator: bool = True):
        self.add_missing_indicator = add_missing_indicator
        self.column_means: Dict[str, float] = {}
        self.is_fitted = False
        self.impute_columns = FORECAST_COVARIATES + ["shock_risk"]

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        """Compute column statistics from the training set only."""
        for col in ALL_COVARIATE_COLUMNS + ["target"]:
            if col in df.columns:
                self.column_means[col] = float(df[col].mean(skipna=True))
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply forward-fill, fallback mean imputation, and optional missingness masks."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on training data before calling transform.")

        df_out = df.copy()

        # Sort by series_id and timestamp to ensure forward fill respects temporal order
        if "timestamp" in df_out.columns:
            df_out["timestamp"] = pd.to_datetime(df_out["timestamp"])
            df_out = df_out.sort_values(["series_id", "timestamp"]).reset_index(drop=True)

        # Impute missing values per series
        grouped = df_out.groupby("series_id", group_keys=False)
        for col in self.impute_columns:
            if col in df_out.columns:
                if self.add_missing_indicator:
                    df_out[f"{col}_isnan"] = df_out[col].isna().astype(np.float32)
                # Forward-fill within series, then back-fill, then global training mean fallback
                df_out[col] = grouped[col].apply(lambda s: s.ffill().bfill())
                fallback = self.column_means.get(col, 0.0)
                df_out[col] = df_out[col].fillna(fallback)

        return df_out


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset generating sliding-window sequences for multi-horizon forecasting.
    
    Yields:
      - x_past_target: Tensor of shape (lookback_len,)
      - x_past_cov: Tensor of shape (lookback_len, num_past_covariates)
      - x_future_cov: Tensor of shape (horizon, num_future_covariates)
      - y_future_target: Tensor of shape (horizon,)
    """

    def __init__(
        self,
        series_targets: List[np.ndarray],
        series_past_covs: List[np.ndarray],
        series_future_covs: List[np.ndarray],
        lookback_len: int = 168,
        horizon: int = 24,
        stride: int = 1,
    ):
        self.lookback_len = lookback_len
        self.horizon = horizon
        self.total_window_len = lookback_len + horizon

        self.series_targets = series_targets
        self.series_past_covs = series_past_covs
        self.series_future_covs = series_future_covs

        # Build index mapping: (series_idx, start_t)
        self.samples: List[Tuple[int, int]] = []
        for s_idx, target_seq in enumerate(series_targets):
            n_steps = len(target_seq)
            if n_steps >= self.total_window_len:
                for t in range(0, n_steps - self.total_window_len + 1, stride):
                    self.samples.append((s_idx, t))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s_idx, t = self.samples[idx]

        target_seq = self.series_targets[s_idx]
        past_cov_seq = self.series_past_covs[s_idx]
        future_cov_seq = self.series_future_covs[s_idx]

        # Past window [t : t + lookback_len]
        x_past_target = torch.from_numpy(target_seq[t : t + self.lookback_len]).float()
        x_past_cov = torch.from_numpy(past_cov_seq[t : t + self.lookback_len]).float()

        # Future window [t + lookback_len : t + total_window_len]
        x_future_cov = torch.from_numpy(
            future_cov_seq[t + self.lookback_len : t + self.total_window_len]
        ).float()
        y_future_target = torch.from_numpy(
            target_seq[t + self.lookback_len : t + self.total_window_len]
        ).float()

        return x_past_target, x_past_cov, x_future_cov, y_future_target


def prepare_series_arrays(
    df: pd.DataFrame,
    past_cov_cols: List[str],
    future_cov_cols: List[str],
    target_col: str = "target",
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Group DataFrame by series_id and extract contiguous NumPy arrays."""
    series_targets = []
    series_past_covs = []
    series_future_covs = []

    for _, group in df.groupby("series_id", sort=False):
        series_targets.append(group[target_col].to_numpy(dtype=np.float32))
        series_past_covs.append(group[past_cov_cols].to_numpy(dtype=np.float32))
        series_future_covs.append(group[future_cov_cols].to_numpy(dtype=np.float32))

    return series_targets, series_past_covs, series_future_covs


def get_dataloaders(
    train_csv_path: str = "data/benchmark/train.csv",
    lookback_len: int = 168,
    horizon: int = 24,
    val_horizon: int = 336,
    batch_size: int = 128,
    stride_train: int = 1,
    stride_val: int = 24,
    add_missing_indicator: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, Preprocessor, Dict[str, int]]:
    """Load data, split into train and local validation sets, and build PyTorch DataLoaders."""
    raw_df = pd.read_csv(train_csv_path)

    # Determine past and future feature lists
    past_cov_cols = list(ALL_COVARIATE_COLUMNS)
    future_cov_cols = list(FUTURE_COVARIATE_COLUMNS)

    if add_missing_indicator:
        for col in FORECAST_COVARIATES + ["shock_risk"]:
            mask_col = f"{col}_isnan"
            past_cov_cols.append(mask_col)
            if col in FUTURE_COVARIATE_COLUMNS:
                future_cov_cols.append(mask_col)

    # Split train and validation by series timestamps
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
    raw_df = raw_df.sort_values(["series_id", "timestamp"]).reset_index(drop=True)

    train_parts = []
    val_parts = []

    for _, group in raw_df.groupby("series_id", sort=False):
        train_parts.append(group.iloc[:-val_horizon])
        # Include lookback_len from train tail to form first validation window
        val_parts.append(group.iloc[-(val_horizon + lookback_len) :])

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)

    # Fit preprocessor on training data only
    preprocessor = Preprocessor(add_missing_indicator=add_missing_indicator).fit(train_df)
    train_df_proc = preprocessor.transform(train_df)
    val_df_proc = preprocessor.transform(val_df)

    # Convert to contiguous series arrays
    train_targets, train_past_covs, train_future_covs = prepare_series_arrays(
        train_df_proc, past_cov_cols, future_cov_cols
    )
    val_targets, val_past_covs, val_future_covs = prepare_series_arrays(
        val_df_proc, past_cov_cols, future_cov_cols
    )

    # Datasets
    train_dataset = TimeSeriesDataset(
        train_targets,
        train_past_covs,
        train_future_covs,
        lookback_len=lookback_len,
        horizon=horizon,
        stride=stride_train,
    )
    val_dataset = TimeSeriesDataset(
        val_targets,
        val_past_covs,
        val_future_covs,
        lookback_len=lookback_len,
        horizon=horizon,
        stride=stride_val,
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    dims = {
        "lookback_len": lookback_len,
        "horizon": horizon,
        "past_cov_dim": len(past_cov_cols),
        "future_cov_dim": len(future_cov_cols),
        "n_train_samples": len(train_dataset),
        "n_val_samples": len(val_dataset),
    }

    return train_loader, val_loader, preprocessor, dims
