"""Inference entrypoint for final private and validation evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch

from src.model import ForecastModel


# Feature column definitions
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

FUTURE_COVARIATE_COLUMNS = TEMPORAL_COVARIATES + FORECAST_COVARIATES
ALL_COVARIATE_COLUMNS = TEMPORAL_COVARIATES + FORECAST_COVARIATES + OPERATIONAL_COVARIATES


def preprocess_dataframe(
    df: pd.DataFrame,
    column_means: Dict[str, float],
    add_missing_indicator: bool = True,
) -> pd.DataFrame:
    """Impute missing values per series using forward-fill, back-fill, and training mean fallback."""
    df_out = df.copy()
    if "timestamp" in df_out.columns:
        df_out["timestamp"] = pd.to_datetime(df_out["timestamp"])
        df_out = df_out.sort_values(["series_id", "timestamp"]).reset_index(drop=True)

    # Recreate any covariate column missing from the input schema so that downstream
    # feature-matrix lookups cannot raise KeyError on an unexpected private test layout.
    for col in ALL_COVARIATE_COLUMNS:
        if col not in df_out.columns:
            df_out[col] = np.float32(column_means.get(col, 0.0))

    grouped = df_out.groupby("series_id", group_keys=False)
    impute_cols = FORECAST_COVARIATES + ["shock_risk"]

    for col in impute_cols:
        if col in df_out.columns:
            if add_missing_indicator:
                df_out[f"{col}_isnan"] = df_out[col].isna().astype(np.float32)
            df_out[col] = grouped[col].apply(lambda s: s.ffill().bfill())
            fallback = column_means.get(col, 0.0)
            df_out[col] = df_out[col].fillna(fallback)

    return df_out


def load_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load forecast index, future input features, and historical train data."""
    # 1. Forecast Index
    index_candidates = [
        input_dir / "forecast_index_test.csv",
        input_dir / "forecast_index_validation.csv",
    ]
    forecast_index = None
    for p in index_candidates:
        if p.exists():
            forecast_index = pd.read_csv(p)
            break
    if forecast_index is None:
        raise FileNotFoundError(f"Missing forecast index file in {input_dir}.")

    # 2. Future input features
    input_candidates = [
        input_dir / "test_input.csv",
        input_dir / "validation_input.csv",
    ]
    future_input = None
    future_input_path = None
    for p in input_candidates:
        if p.exists():
            future_input = pd.read_csv(p)
            future_input_path = p
            break
    if future_input is None:
        raise FileNotFoundError(f"Missing test_input.csv or validation_input.csv in {input_dir}.")

    # 3. Target history. Named candidates first, then any CSV in the input tree that
    #    actually carries a target column, so an unexpected file layout cannot zero the run.
    named_candidates = [
        input_dir / "train.csv",
        input_dir / "history.csv",
        input_dir / "train_input.csv",
        input_dir.parent / "benchmark" / "train.csv",
    ]
    history_frames: List[pd.DataFrame] = []
    for p in named_candidates:
        if p.exists():
            df = pd.read_csv(p)
            if "target" in df.columns and "series_id" in df.columns:
                history_frames.append(df)
                break

    if not history_frames:
        # Fall back to scanning the input tree for any file holding observed targets.
        for p in sorted(input_dir.rglob("*.csv")):
            if p.name in {"forecast_index_test.csv", "forecast_index_validation.csv"}:
                continue
            if future_input_path is not None and p.resolve() == future_input_path.resolve():
                continue  # handled separately below
            try:
                head = pd.read_csv(p, nrows=1)
            except Exception:
                continue
            if "target" in head.columns and "series_id" in head.columns:
                df = pd.read_csv(p).dropna(subset=["target"])
                if len(df) > 0:
                    history_frames.append(df)

    # The future-input file itself may embed observed history rows alongside the
    # rows to forecast; harvest any labelled rows from it as additional history.
    if "target" in future_input.columns:
        labelled = future_input[future_input["target"].notna()]
        if len(labelled) > 0:
            history_frames.append(labelled.copy())
        future_input = future_input.drop(columns=["target"])
        history_frames = [f.dropna(subset=["target"]) for f in history_frames]
        history_frames = [f for f in history_frames if len(f) > 0]

    if not history_frames:
        raise FileNotFoundError(
            f"Missing historical target data in {input_dir}. "
            f"Found CSV files: {[p.name for p in input_dir.rglob('*.csv')]}"
        )

    train_df = max(history_frames, key=len) if len(history_frames) == 1 else (
        pd.concat(history_frames, ignore_index=True)
        .drop_duplicates(subset=["series_id", "timestamp"], keep="last")
    )

    return forecast_index, future_input, train_df


def predict_rolling(
    model: ForecastModel,
    train_df: pd.DataFrame,
    future_input_df: pd.DataFrame,
    forecast_index_df: pd.DataFrame,
    column_means: Dict[str, float],
    lookback_len: int = 168,
    horizon: int = 24,
    device: torch.device = torch.device("cpu"),
) -> pd.DataFrame:
    """Execute 24-step iterative rolling predictions for all series across 336 steps."""
    model.eval()
    model.to(device)

    past_cov_cols = list(ALL_COVARIATE_COLUMNS)
    future_cov_cols = list(FUTURE_COVARIATE_COLUMNS)
    for col in FORECAST_COVARIATES + ["shock_risk"]:
        mask_col = f"{col}_isnan"
        past_cov_cols.append(mask_col)
        if col in FUTURE_COVARIATE_COLUMNS:
            future_cov_cols.append(mask_col)

    train_proc = preprocess_dataframe(train_df, column_means)
    future_proc = preprocess_dataframe(future_input_df, column_means)

    # Parse the forecast-index timestamps for ordering/joining while preserving the
    # original string column verbatim for the output file.
    fidx = forecast_index_df.copy()
    fidx["_ts"] = pd.to_datetime(fidx["timestamp"])

    predictions_list: List[pd.DataFrame] = []
    unique_series = fidx["series_id"].unique()

    with torch.no_grad():
        for series_id in unique_series:
            s_train = train_proc[train_proc["series_id"] == series_id]
            # Chronological order is what makes the recursive rollout meaningful; never
            # rely on the row order the index file happens to ship with.
            s_index = fidx[fidx["series_id"] == series_id].sort_values("_ts").copy()

            # Restrict the covariate frame to exactly the timestamps being forecast, so
            # embedded history rows or extra rows cannot shift the rollout off-target.
            s_future_all = future_proc[future_proc["series_id"] == series_id]
            s_future = s_future_all[s_future_all["timestamp"].isin(set(s_index["_ts"]))].sort_values("timestamp")

            if len(s_future) != len(s_index):
                raise ValueError(
                    f"Series {series_id}: {len(s_index)} forecast rows but {len(s_future)} "
                    f"matching covariate rows; cannot align the rollout."
                )
            if len(s_train) < lookback_len:
                raise ValueError(
                    f"Series {series_id} has {len(s_train)} history rows, requires at least {lookback_len}."
                )

            target_buffer = list(s_train["target"].iloc[-lookback_len:].to_numpy(dtype=np.float32))
            past_cov_buffer = list(s_train[past_cov_cols].iloc[-lookback_len:].to_numpy(dtype=np.float32))

            future_cov_matrix = s_future[future_cov_cols].to_numpy(dtype=np.float32)
            future_past_cov_matrix = s_future[past_cov_cols].to_numpy(dtype=np.float32)

            n_future_steps = len(s_future)
            series_preds: List[float] = []

            for start_step in range(0, n_future_steps, horizon):
                end_step = min(start_step + horizon, n_future_steps)
                curr_horizon = end_step - start_step

                x_past_target = torch.from_numpy(np.array(target_buffer[-lookback_len:], dtype=np.float32)).unsqueeze(0).to(device)
                x_past_cov = torch.from_numpy(np.array(past_cov_buffer[-lookback_len:], dtype=np.float32)).unsqueeze(0).to(device)

                if curr_horizon == horizon:
                    x_future_cov = torch.from_numpy(np.array(future_cov_matrix[start_step:end_step], dtype=np.float32)).unsqueeze(0).to(device)
                else:
                    pad_len = horizon - curr_horizon
                    cov_slice = future_cov_matrix[start_step:end_step]
                    padded_cov = np.pad(cov_slice, ((0, pad_len), (0, 0)), mode="edge")
                    x_future_cov = torch.from_numpy(np.array(padded_cov, dtype=np.float32)).unsqueeze(0).to(device)

                y_pred_tensor = model(x_past_target, x_past_cov, x_future_cov)
                y_pred_np = y_pred_tensor.cpu().numpy()[0][:curr_horizon]
                y_pred_np = np.clip(y_pred_np, a_min=0.0, a_max=None)

                series_preds.extend(y_pred_np.tolist())

                target_buffer.extend(y_pred_np.tolist())
                past_cov_buffer.extend(future_past_cov_matrix[start_step:end_step])

            s_index["prediction"] = series_preds[: len(s_index)]
            predictions_list.append(s_index[["series_id", "timestamp", "prediction"]])

    # Restore the original row order of the forecast index file.
    out = pd.concat(predictions_list, ignore_index=True)
    key = forecast_index_df[["series_id", "timestamp"]].copy()
    key["_row"] = np.arange(len(key))
    out = out.merge(key, on=["series_id", "timestamp"], how="right").sort_values("_row")
    if out["prediction"].isna().any():
        raise ValueError("Failed to produce a prediction for every forecast-index row.")
    return out[["series_id", "timestamp", "prediction"]].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate private test or validation predictions.")
    parser.add_argument("--input_dir", required=True, type=Path, help="Path containing input files.")
    parser.add_argument("--output_file", required=True, type=Path, help="Path to write output predictions CSV.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to model checkpoint.pt.")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    forecast_index, future_input, train_df = load_inputs(args.input_dir)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config", {})
    state_dict = checkpoint.get("state_dict", checkpoint)
    preprocessor_means = checkpoint.get("preprocessor_means", {})

    model = ForecastModel(
        lookback_len=config.get("lookback_len", 168),
        horizon=config.get("horizon", 24),
        past_cov_dim=config.get("past_cov_dim", 32),
        future_cov_dim=config.get("future_cov_dim", 25),
        feature_dim=config.get("feature_dim", 16),
        hidden_dim=config.get("hidden_dim", 256),
        decoder_dim=config.get("decoder_dim", 128),
        num_encoder_layers=config.get("num_encoder_layers", 2),
        num_decoder_layers=config.get("num_decoder_layers", 2),
        dropout=config.get("dropout", 0.1),
        use_revin=config.get("use_revin", True),
        use_past_covariates=config.get("use_past_covariates", True),
        use_future_covariates=config.get("use_future_covariates", True),
    )

    model.load_state_dict(state_dict)
    model.eval()

    print(f"[INFO] Generating predictions for {len(forecast_index):,} target rows...")
    predictions = predict_rolling(
        model=model,
        train_df=train_df,
        future_input_df=future_input,
        forecast_index_df=forecast_index,
        column_means=preprocessor_means,
        lookback_len=config.get("lookback_len", 168),
        horizon=config.get("horizon", 24),
    )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_file, index=False)
    print(f"[SUCCESS] Wrote predictions to {args.output_file}")


if __name__ == "__main__":
    main()
