"""Evaluate baselines on a local validation split (last 336 steps of train.csv)."""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd


# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from baselines.baselines import make_all_baselines
from src.utils import compute_metrics


def main():
    print("Loading data for local validation backtest...")
    df = pd.read_csv("data/benchmark/train.csv")
    
    val_horizon = 336
    
    # Sort by series_id and timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["series_id", "timestamp"]).reset_index(drop=True)
    
    # Split each series: first (N - 336) steps -> local_train, last 336 steps -> local_val
    train_parts = []
    val_parts = []
    
    for series_id, group in df.groupby("series_id", sort=False):
        train_parts.append(group.iloc[:-val_horizon])
        val_parts.append(group.iloc[-val_horizon:])
        
    local_train = pd.concat(train_parts, ignore_index=True)
    local_val = pd.concat(val_parts, ignore_index=True)
    
    # Format timestamps back to string format
    local_train["timestamp"] = local_train["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    local_val["timestamp"] = local_val["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    forecast_index = local_val[["series_id", "timestamp"]].copy()
    
    print(f"Local Train shape: {local_train.shape}")
    print(f"Local Val shape: {local_val.shape}")
    
    print("Running baseline forecasts...")
    baselines_preds = make_all_baselines(local_train, forecast_index)
    
    results = []
    y_true = local_val["target"].to_numpy()
    
    for name, pred_df in baselines_preds.items():
        y_pred = pred_df["prediction"].to_numpy()
        metrics = compute_metrics(y_true, y_pred)
        row = {"Baseline": name}
        row.update({k: f"{v:.4f}" if k not in ["MAPE", "sMAPE"] else f"{v:.2f}%" for k, v in metrics.items()})
        results.append(row)
        
    results_df = pd.DataFrame(results)
    print("\n=== LOCAL VALIDATION (336-step) BASELINE BENCHMARKS ===")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
