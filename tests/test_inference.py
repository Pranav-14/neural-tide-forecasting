"""Unit test for 24-step iterative rolling inference engine."""

import sys
from pathlib import Path
import pandas as pd
import torch

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.model import TiDE
from src.inference import run_iterative_rolling_forecast


def test_rolling_inference():
    print("Testing iterative rolling inference...")
    train_df = pd.read_csv("data/benchmark/train.csv")
    val_in = pd.read_csv("data/benchmark/validation_input.csv")
    val_idx = pd.read_csv("data/benchmark/forecast_index_validation.csv")

    # Take first 2 series for quick verification
    test_units = ["unit_000", "unit_001"]
    train_sub = train_df[train_df["series_id"].isin(test_units)]
    val_in_sub = val_in[val_in["series_id"].isin(test_units)]
    val_idx_sub = val_idx[val_idx["series_id"].isin(test_units)]

    model = TiDE(
        lookback_len=168,
        horizon=24,
        past_cov_dim=32,
        future_cov_dim=25,
        feature_dim=16,
        hidden_dim=64,
        decoder_dim=32,
        use_revin=True,
    )

    preds_df = run_iterative_rolling_forecast(
        model=model,
        train_df=train_sub,
        future_input_df=val_in_sub,
        forecast_index_df=val_idx_sub,
        lookback_len=168,
        horizon=24,
    )

    print(f"Prediction output shape: {preds_df.shape} (Expected: [{len(val_idx_sub)}, 3])")
    assert len(preds_df) == len(val_idx_sub)
    assert list(preds_df.columns) == ["series_id", "timestamp", "prediction"]
    assert not preds_df["prediction"].isna().any(), "NaN found in predictions"
    assert (preds_df["prediction"] >= 0.0).all(), "Negative prediction found"

    print("[SUCCESS] Rolling inference test passed successfully!")


if __name__ == "__main__":
    test_rolling_inference()
