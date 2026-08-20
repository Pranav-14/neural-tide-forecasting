"""Systematic Ablation Studies and LSTM Neural Baseline for Operational Load Forecasting."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset import get_dataloaders
from src.model import TiDE, RevIN
from src.train import set_seed
from src.utils import compute_metrics


# ==============================================================================
# 1. Standard Neural Baseline: LSTM Sequence Model
# ==============================================================================
class LSTMBaseline(nn.Module):
    """Standard 2-layer LSTM baseline for multi-horizon forecasting."""

    def __init__(
        self,
        lookback_len: int = 168,
        horizon: int = 24,
        past_cov_dim: int = 32,
        future_cov_dim: int = 25,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_revin: bool = True,
    ):
        super().__init__()
        self.lookback_len = lookback_len
        self.horizon = horizon
        self.use_revin = use_revin

        if self.use_revin:
            self.revin = RevIN(num_features=1, affine=True)

        input_dim = 1 + past_cov_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Future covariate encoder
        self.future_proj = nn.Linear(future_cov_dim, 32)

        # Projection head to multi-step forecast horizon
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 32 * horizon, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(
        self,
        x_past_target: torch.Tensor,       # (B, L)
        x_past_cov: torch.Tensor,          # (B, L, past_cov_dim)
        x_future_cov: torch.Tensor,        # (B, H, future_cov_dim)
    ) -> torch.Tensor:
        B = x_past_target.size(0)

        # Step 1: RevIN Normalize target
        if self.use_revin:
            norm_target = self.revin.normalize(x_past_target.unsqueeze(-1)).squeeze(-1)
        else:
            norm_target = x_past_target

        # Step 2: Concat target history and past covariates
        lstm_input = torch.cat([norm_target.unsqueeze(-1), x_past_cov], dim=-1)  # (B, L, 1 + D_p)

        # Step 3: LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)
        last_hidden = h_n[-1]  # (B, hidden_dim)

        # Step 4: Future covariates projection
        proj_future = self.future_proj(x_future_cov).reshape(B, -1)  # (B, H * 32)

        # Step 5: Head projection
        combined = torch.cat([last_hidden, proj_future], dim=-1)
        y_norm_pred = self.head(combined)  # (B, H)

        # Step 6: RevIN Denormalize
        if self.use_revin:
            y_pred = self.revin.denormalize(y_norm_pred.unsqueeze(-1)).squeeze(-1)
        else:
            y_pred = y_norm_pred

        return y_pred


# ==============================================================================
# 2. Ablation Runner and Evaluator
# ==============================================================================
def train_and_eval_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 3,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Train a model variant and compute validation metrics."""
    print(f"\n---> Training: {model_name} ({epochs} epochs)...")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.HuberLoss(delta=1.0)

    best_wape = float("inf")
    best_metrics = {}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_train = 0

        for x_past_target, x_past_cov, x_future_cov, y_future_target in train_loader:
            x_past_target = x_past_target.to(device)
            x_past_cov = x_past_cov.to(device)
            x_future_cov = x_future_cov.to(device)
            y_future_target = y_future_target.to(device)

            optimizer.zero_grad()
            y_pred = model(x_past_target, x_past_cov, x_future_cov)
            loss = criterion(y_pred, y_future_target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * x_past_target.size(0)
            n_train += x_past_target.size(0)

        scheduler.step()

        # Validation pass
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for x_past_target, x_past_cov, x_future_cov, y_future_target in val_loader:
                x_past_target = x_past_target.to(device)
                x_past_cov = x_past_cov.to(device)
                x_future_cov = x_future_cov.to(device)

                y_pred = model(x_past_target, x_past_cov, x_future_cov)
                y_pred = torch.clamp(y_pred, min=0.0)

                all_preds.append(y_pred.cpu().numpy())
                all_targets.append(y_future_target.numpy())

        y_pred_flat = np.concatenate(all_preds, axis=0).flatten()
        y_true_flat = np.concatenate(all_targets, axis=0).flatten()

        m = compute_metrics(y_true_flat, y_pred_flat)
        if m["WAPE"] < best_wape:
            best_wape = m["WAPE"]
            best_metrics = m.copy()

        dur = time.time() - t0
        print(f"  Epoch {epoch:2d}/{epochs} | Val WAPE: {m['WAPE']:.4f} | Val MAE: {m['MAE']:.4f} | Val RMSE: {m['RMSE']:.4f} ({dur:.1f}s)")

    return best_metrics


def run_all_ablations(
    train_csv_path: str = "data/benchmark/train.csv",
    epochs: int = 3,
    batch_size: int = 256,
    stride_train: int = 6,  # Stride 6 gives 60k representative training windows for fast CPU ablations
    seed: int = 42,
):
    """Execute systematic ablation experiments across all modeling components."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running Ablation Suite on: {device}")

    # Standard Lookback (L=168, H=24)
    train_loader_168, val_loader_168, _, dims_168 = get_dataloaders(
        train_csv_path=train_csv_path,
        lookback_len=168,
        horizon=24,
        val_horizon=336,
        batch_size=batch_size,
        stride_train=stride_train,
        stride_val=24,
        num_workers=0,
    )

    # Short Lookback (L=72, H=24)
    train_loader_72, val_loader_72, _, dims_72 = get_dataloaders(
        train_csv_path=train_csv_path,
        lookback_len=72,
        horizon=24,
        val_horizon=336,
        batch_size=batch_size,
        stride_train=stride_train,
        stride_val=24,
        num_workers=0,
    )

    past_cov_dim = dims_168["past_cov_dim"]
    future_cov_dim = dims_168["future_cov_dim"]

    ablation_results = []

    # --------------------------------------------------------------------------
    # Experiment 1: TiDE (No Covariates)
    # --------------------------------------------------------------------------
    model_no_cov = TiDE(
        lookback_len=168, horizon=24, past_cov_dim=past_cov_dim, future_cov_dim=future_cov_dim,
        feature_dim=16, hidden_dim=256, decoder_dim=128, use_revin=True,
        use_past_covariates=False, use_future_covariates=False,
    )
    m1 = train_and_eval_model("1. TiDE (No Covariates)", model_no_cov, train_loader_168, val_loader_168, epochs=epochs, device=device)
    ablation_results.append({"Configuration": "TiDE (No Covariates)", **m1})

    # --------------------------------------------------------------------------
    # Experiment 2: TiDE (No RevIN)
    # --------------------------------------------------------------------------
    model_no_revin = TiDE(
        lookback_len=168, horizon=24, past_cov_dim=past_cov_dim, future_cov_dim=future_cov_dim,
        feature_dim=16, hidden_dim=256, decoder_dim=128, use_revin=False,
        use_past_covariates=True, use_future_covariates=True,
    )
    m2 = train_and_eval_model("2. TiDE (No RevIN)", model_no_revin, train_loader_168, val_loader_168, epochs=epochs, device=device)
    ablation_results.append({"Configuration": "TiDE (No RevIN)", **m2})

    # --------------------------------------------------------------------------
    # Experiment 3: TiDE (Short Lookback L=72)
    # --------------------------------------------------------------------------
    model_l72 = TiDE(
        lookback_len=72, horizon=24, past_cov_dim=past_cov_dim, future_cov_dim=future_cov_dim,
        feature_dim=16, hidden_dim=256, decoder_dim=128, use_revin=True,
        use_past_covariates=True, use_future_covariates=True,
    )
    m3 = train_and_eval_model("3. TiDE (Lookback L=72)", model_l72, train_loader_72, val_loader_72, epochs=epochs, device=device)
    ablation_results.append({"Configuration": "TiDE (Lookback L=72)", **m3})

    # --------------------------------------------------------------------------
    # Experiment 4: LSTM Neural Baseline
    # --------------------------------------------------------------------------
    lstm_model = LSTMBaseline(
        lookback_len=168, horizon=24, past_cov_dim=past_cov_dim, future_cov_dim=future_cov_dim,
        hidden_dim=128, num_layers=2, use_revin=True,
    )
    m4 = train_and_eval_model("4. LSTM Neural Baseline", lstm_model, train_loader_168, val_loader_168, epochs=epochs, device=device)
    ablation_results.append({"Configuration": "LSTM Neural Baseline", **m4})

    # --------------------------------------------------------------------------
    # Experiment 5: TiDE + RevIN (Full Proposed Model)
    # --------------------------------------------------------------------------
    model_full = TiDE(
        lookback_len=168, horizon=24, past_cov_dim=past_cov_dim, future_cov_dim=future_cov_dim,
        feature_dim=16, hidden_dim=256, decoder_dim=128, use_revin=True,
        use_past_covariates=True, use_future_covariates=True,
    )
    m5 = train_and_eval_model("5. TiDE + RevIN (Full Proposed)", model_full, train_loader_168, val_loader_168, epochs=epochs, device=device)
    ablation_results.append({"Configuration": "TiDE + RevIN (Full Proposed)", **m5})

    # Summary Table
    print("\n" + "=" * 85)
    print("=== SYSTEMATIC ABLATION STUDIES SUMMARY (336-Hour Horizon) ===")
    print("=" * 85)

    summary_rows = []
    for r in ablation_results:
        summary_rows.append({
            "Configuration": r["Configuration"],
            "WAPE": f"{r['WAPE']:.4f}",
            "MAE": f"{r['MAE']:.4f}",
            "RMSE": f"{r['RMSE']:.4f}",
            "sMAPE": f"{r['sMAPE']:.2f}%",
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    print("=" * 85)

    summary_df.to_csv("experiments/ablation_results.csv", index=False)
    print("[SUCCESS] Ablation results saved to experiments/ablation_results.csv")


if __name__ == "__main__":
    run_all_ablations()
