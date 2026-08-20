"""Unit test for BATADAL dataset pipeline and covariate-free multi-target TiDE model."""

import sys
from pathlib import Path
import torch
import pandas as pd

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from experiments.run_batadal import (
    load_and_preprocess_batadal,
    BatadalDataset,
    BatadalTiDE,
    PRIMARY_TARGETS,
    PRESSURE_COVARIATES,
    STATUS_COVARIATES,
)


def test_batadal_pipeline():
    print("Testing BATADAL data preprocessing...")
    df = load_and_preprocess_batadal("data/batadal/training_dataset_1.csv")

    assert len(df) > 0, "Empty dataframe"
    assert "hour_sin" in df.columns, "Missing cyclical temporal feature"
    assert "dow_cos" in df.columns, "Missing cyclical temporal feature"

    target_cols = PRIMARY_TARGETS  # 19 targets
    temporal_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]
    past_cov_cols = PRESSURE_COVARIATES + STATUS_COVARIATES + temporal_cols
    future_cov_cols = temporal_cols

    print(f"Creating BatadalDataset (Targets: {len(target_cols)}, Past Covs: {len(past_cov_cols)})...")
    dataset = BatadalDataset(
        targets_matrix=df[target_cols].iloc[:500].to_numpy(),
        past_covs_matrix=df[past_cov_cols].iloc[:500].to_numpy(),
        future_covs_matrix=df[future_cov_cols].iloc[:500].to_numpy(),
        lookback_len=168,
        horizon=24,
        stride=1,
    )

    assert len(dataset) > 0, "Dataset is empty"
    x_past_target, x_past_cov, x_future_cov, y_future_target = dataset[0]

    assert x_past_target.shape == (168, len(target_cols))
    assert x_past_cov.shape == (168, len(past_cov_cols))
    assert x_future_cov.shape == (24, len(future_cov_cols))
    assert y_future_target.shape == (24, len(target_cols))

    print("Testing BatadalTiDE forward pass...")
    model = BatadalTiDE(
        num_targets=len(target_cols),
        lookback_len=168,
        horizon=24,
        past_cov_dim=len(past_cov_cols),
        future_cov_dim=len(future_cov_cols),
        feature_dim=16,
        hidden_dim=64,
        decoder_dim=32,
        use_revin=True,
    )

    batch_past_target = x_past_target.unsqueeze(0)
    batch_past_cov = x_past_cov.unsqueeze(0)
    batch_future_cov = x_future_cov.unsqueeze(0)

    y_pred = model(batch_past_target, batch_past_cov, batch_future_cov)
    assert y_pred.shape == (1, 24, len(target_cols)), f"Unexpected shape {y_pred.shape}"
    assert not torch.isnan(y_pred).any(), "NaN found in predictions"

    print("[SUCCESS] BATADAL pipeline and model forward test passed successfully!")


if __name__ == "__main__":
    test_batadal_pipeline()
