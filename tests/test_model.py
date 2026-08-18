"""Unit tests for TiDE and RevIN in src/model.py."""

import sys
from pathlib import Path

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from src.model import RevIN, TiDE


def test_revin():
    print("Testing RevIN...")
    revin = RevIN(num_features=1, affine=True)
    x = torch.randn(32, 168) * 10.0 + 50.0  # arbitrary non-zero mean and scale
    
    x_norm = revin.normalize(x)
    assert x_norm.shape == (32, 168)
    
    # Check normalized statistics
    assert torch.allclose(x_norm.mean(dim=1), torch.zeros(32), atol=1e-4)
    assert torch.allclose(x_norm.std(dim=1, unbiased=False), torch.ones(32), atol=1e-4)
    
    # Check inverse recovery
    x_rec = revin.denormalize(x_norm)
    assert torch.allclose(x, x_rec, atol=1e-4)
    print("RevIN normalization and denormalization test passed.")


def test_tide_forward():
    print("\nTesting TiDE forward pass...")
    B, L, H = 16, 168, 24
    past_dim, future_dim = 32, 25

    model = TiDE(
        lookback_len=L,
        horizon=H,
        past_cov_dim=past_dim,
        future_cov_dim=future_dim,
        feature_dim=16,
        hidden_dim=128,
        decoder_dim=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        use_revin=True,
    )

    x_past_target = torch.randn(B, L) * 5.0 + 10.0
    x_past_cov = torch.randn(B, L, past_dim)
    x_future_cov = torch.randn(B, H, future_dim)

    y_pred = model(x_past_target, x_past_cov, x_future_cov)
    print(f"Output prediction shape: {y_pred.shape} (Expected: [{B}, {H}])")
    assert y_pred.shape == (B, H)
    assert not torch.isnan(y_pred).any(), "NaN in TiDE predictions"

    # Backward pass check
    loss = y_pred.sum()
    loss.backward()
    print("Backward gradient pass succeeded.")


if __name__ == "__main__":
    test_revin()
    test_tide_forward()
    print("\n[SUCCESS] All model tests passed successfully!")
