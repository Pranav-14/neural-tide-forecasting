"""Unit tests and verification for src/dataset.py."""

import sys
from pathlib import Path

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from src.dataset import get_dataloaders


def test_dataset_pipeline():
    print("Testing get_dataloaders pipeline...")
    train_loader, val_loader, preprocessor, dims = get_dataloaders(
        train_csv_path="data/benchmark/train.csv",
        lookback_len=168,
        horizon=24,
        val_horizon=336,
        batch_size=64,
        add_missing_indicator=True,
    )

    print(f"Dataset Dims: {dims}")
    assert dims["lookback_len"] == 168
    assert dims["horizon"] == 24
    assert dims["past_cov_dim"] > 20
    assert dims["future_cov_dim"] > 10
    assert dims["n_train_samples"] > 100000

    # Fetch first batch from train_loader
    for x_past_target, x_past_cov, x_future_cov, y_future_target in train_loader:
        print("\n--- Train Batch Tensor Shapes ---")
        print(f"x_past_target shape: {x_past_target.shape} (Expected: [64, 168])")
        print(f"x_past_cov shape:    {x_past_cov.shape} (Expected: [64, 168, {dims['past_cov_dim']}])")
        print(f"x_future_cov shape:  {x_future_cov.shape} (Expected: [64, 24, {dims['future_cov_dim']}])")
        print(f"y_future_target shape: {y_future_target.shape} (Expected: [64, 24])")

        assert x_past_target.shape == (64, 168)
        assert x_past_cov.shape == (64, 168, dims["past_cov_dim"])
        assert x_future_cov.shape == (64, 24, dims["future_cov_dim"])
        assert y_future_target.shape == (64, 24)

        assert not torch.isnan(x_past_target).any(), "NaN found in x_past_target"
        assert not torch.isnan(x_past_cov).any(), "NaN found in x_past_cov"
        assert not torch.isnan(x_future_cov).any(), "NaN found in x_future_cov"
        assert not torch.isnan(y_future_target).any(), "NaN found in y_future_target"
        break

    # Fetch first batch from val_loader
    for x_past_target, x_past_cov, x_future_cov, y_future_target in val_loader:
        print("\n--- Val Batch Tensor Shapes ---")
        print(f"Val batch size: {x_past_target.shape[0]}")
        assert not torch.isnan(x_past_target).any(), "NaN found in val x_past_target"
        break

    print("\n[SUCCESS] All dataset tests passed successfully!")


if __name__ == "__main__":
    test_dataset_pipeline()
