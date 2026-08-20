"""Recursive 336-step backtest of the trained model against all heuristic baselines.

This is the evaluation protocol the leaderboard uses: the model sees the true history
once, then rolls its own 24-step predictions forward 14 times to cover the full
336-hour horizon. Every model in the table is scored on the identical held-out
window (the last 336 hours of train.csv), which is excluded from training by
`get_dataloaders`, so the comparison is apples-to-apples.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from baselines.baselines import make_all_baselines
from src.inference import run_iterative_rolling_forecast
from src.model import TiDE
from src.utils import compute_metrics

VAL_HORIZON = 336


def run_backtest(
    train_csv_path: str = "data/benchmark/train.csv",
    checkpoint_path: str = "submission/checkpoint.pt",
    output_csv: str = "experiments/benchmark_results.csv",
) -> pd.DataFrame:
    df = pd.read_csv(train_csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["series_id", "timestamp"]).reset_index(drop=True)

    train_parts, val_parts = [], []
    for _, group in df.groupby("series_id", sort=False):
        train_parts.append(group.iloc[:-VAL_HORIZON])
        val_parts.append(group.iloc[-VAL_HORIZON:])
    local_train = pd.concat(train_parts, ignore_index=True)
    local_val = pd.concat(val_parts, ignore_index=True)
    for frame in (local_train, local_val):
        frame["timestamp"] = frame["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    forecast_index = local_val[["series_id", "timestamp"]].copy()
    future_input = local_val.drop(columns=["target"])
    ground_truth = local_val[["series_id", "timestamp", "target"]]

    print(f"[INFO] Local holdout: {len(local_val):,} rows "
          f"({local_val['series_id'].nunique()} series x {VAL_HORIZON} hours)")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    model = TiDE(**config)
    model.load_state_dict(checkpoint["state_dict"])

    print("[INFO] Running recursive 24x14 rollout...")
    preds = run_iterative_rolling_forecast(
        model=model,
        train_df=local_train,
        future_input_df=future_input,
        forecast_index_df=forecast_index,
        preprocessor_means=checkpoint.get("preprocessor_means"),
        lookback_len=config["lookback_len"],
        horizon=config["horizon"],
    )
    merged = ground_truth.merge(preds, on=["series_id", "timestamp"], how="left")
    assert merged["prediction"].notna().all(), "Missing predictions after merge."

    # Persist the rollout so figures can plot predictions against the matching ground truth.
    preds_path = Path("experiments/predictions/tide_revin_backtest.csv")
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(preds_path, index=False)
    print(f"[INFO] Backtest predictions saved to {preds_path}")

    rows = [{"Model": "TiDE + RevIN (Ours)",
             **compute_metrics(merged["target"].to_numpy(), merged["prediction"].to_numpy())}]

    print("[INFO] Scoring heuristic baselines on the identical window...")
    for name, pred_df in make_all_baselines(local_train, forecast_index).items():
        rows.append({"Model": name,
                     **compute_metrics(local_val["target"].to_numpy(),
                                       pred_df["prediction"].to_numpy())})

    results = pd.DataFrame(rows).sort_values("WAPE", ascending=False).reset_index(drop=True)
    naive_wape = results.loc[results["Model"] == "naive_last_value", "WAPE"].iloc[0]
    results["ErrorReductionVsNaive"] = (1.0 - results["WAPE"] / naive_wape) * 100.0

    out = results[["Model", "WAPE", "MAE", "RMSE", "sMAPE", "ErrorReductionVsNaive"]]
    print("\n" + "=" * 88)
    print("=== RECURSIVE 336-STEP BACKTEST (identical protocol for every model) ===")
    print("=" * 88)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("=" * 88)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"[SUCCESS] Results saved to {output_csv}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", default="data/benchmark/train.csv")
    parser.add_argument("--checkpoint", default="submission/checkpoint.pt")
    parser.add_argument("--output", default="experiments/benchmark_results.csv")
    args = parser.parse_args()
    run_backtest(args.train_csv, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
