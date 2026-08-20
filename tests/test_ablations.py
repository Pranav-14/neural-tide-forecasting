"""Unit test for LSTM neural baseline and ablation configurations."""

import sys
from pathlib import Path
import torch

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from experiments.run_ablations import LSTMBaseline
from src.model import TiDE


def test_lstm_baseline():
    print("Testing LSTM Neural Baseline forward pass...")
    B, L, H = 4, 168, 24
    past_cov_dim = 32
    future_cov_dim = 25

    model = LSTMBaseline(
        lookback_len=L,
        horizon=H,
        past_cov_dim=past_cov_dim,
        future_cov_dim=future_cov_dim,
        hidden_dim=32,
        num_layers=2,
        use_revin=True,
    )

    x_past_target = torch.randn(B, L)
    x_past_cov = torch.randn(B, L, past_cov_dim)
    x_future_cov = torch.randn(B, H, future_cov_dim)

    y_pred = model(x_past_target, x_past_cov, x_future_cov)
    assert y_pred.shape == (B, H), f"Expected shape ({B}, {H}), got {y_pred.shape}"
    assert not torch.isnan(y_pred).any(), "NaN in LSTM output"

    print("[SUCCESS] LSTM baseline test passed successfully!")


def test_tide_ablations():
    print("Testing TiDE ablation configurations...")
    B, L, H = 4, 168, 24
    past_cov_dim = 32
    future_cov_dim = 25

    # 1. No Covariates
    model_no_cov = TiDE(
        lookback_len=L, horizon=H, past_cov_dim=past_cov_dim, future_cov_dim=future_cov_dim,
        feature_dim=16, hidden_dim=64, decoder_dim=32, use_revin=True,
        use_past_covariates=False, use_future_covariates=False,
    )
    x_target = torch.randn(B, L)
    x_pcov = torch.randn(B, L, past_cov_dim)
    x_fcov = torch.randn(B, H, future_cov_dim)
    y1 = model_no_cov(x_target, x_pcov, x_fcov)
    assert y1.shape == (B, H)

    # 2. No RevIN
    model_no_revin = TiDE(
        lookback_len=L, horizon=H, past_cov_dim=past_cov_dim, future_cov_dim=future_cov_dim,
        feature_dim=16, hidden_dim=64, decoder_dim=32, use_revin=False,
    )
    y2 = model_no_revin(x_target, x_pcov, x_fcov)
    assert y2.shape == (B, H)

    print("[SUCCESS] TiDE ablations test passed successfully!")


if __name__ == "__main__":
    test_lstm_baseline()
    test_tide_ablations()
