"""Training pipeline for TiDE with RevIN."""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataset import get_dataloaders
from src.model import TiDE
from src.utils import compute_metrics


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_loss_fn(loss_name: str) -> nn.Module:
    """Return configured loss function."""
    if loss_name.lower() == "huber":
        return nn.HuberLoss(delta=1.0)
    elif loss_name.lower() == "l1":
        return nn.L1Loss()
    elif loss_name.lower() == "mse":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}. Choose from 'huber', 'l1', 'mse'.")


def evaluate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model on validation split and compute official competition metrics."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_past_target, x_past_cov, x_future_cov, y_future_target in val_loader:
            x_past_target = x_past_target.to(device)
            x_past_cov = x_past_cov.to(device)
            x_future_cov = x_future_cov.to(device)
            y_future_target = y_future_target.to(device)

            y_pred = model(x_past_target, x_past_cov, x_future_cov)
            loss = criterion(y_pred, y_future_target)
            total_loss += loss.item() * x_past_target.size(0)

            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_future_target.cpu().numpy())

    n_samples = len(val_loader.dataset)
    avg_loss = total_loss / max(n_samples, 1)

    y_pred_arr = np.concatenate(all_preds, axis=0).flatten()
    y_true_arr = np.concatenate(all_targets, axis=0).flatten()

    metrics = compute_metrics(y_true_arr, y_pred_arr)
    return avg_loss, metrics


def train_model(
    train_csv_path: str = "data/benchmark/train.csv",
    checkpoint_dir: str = "experiments/checkpoints",
    lookback_len: int = 168,
    horizon: int = 24,
    val_horizon: int = 336,
    batch_size: int = 256,
    hidden_dim: int = 256,
    decoder_dim: int = 128,
    feature_dim: int = 16,
    num_encoder_layers: int = 2,
    num_decoder_layers: int = 2,
    dropout: float = 0.1,
    use_revin: bool = True,
    use_past_covariates: bool = True,
    use_future_covariates: bool = True,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 25,
    patience: int = 8,
    loss_name: str = "huber",
    seed: int = 42,
    num_workers: int = 0,
    device_name: Optional[str] = None,
) -> Dict[str, float]:
    """Execute complete TiDE model training loop with validation monitoring and early stopping."""
    set_seed(seed)

    if device_name is not None:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Using device: {device}")

    # Build DataLoaders
    print("[INFO] Building DataLoaders from benchmark dataset...")
    train_loader, val_loader, preprocessor, dims = get_dataloaders(
        train_csv_path=train_csv_path,
        lookback_len=lookback_len,
        horizon=horizon,
        val_horizon=val_horizon,
        batch_size=batch_size,
        stride_train=1,
        stride_val=24,
        add_missing_indicator=True,
        num_workers=num_workers,
    )

    print(f"[INFO] Train samples: {dims['n_train_samples']:,} | Val samples: {dims['n_val_samples']:,}")
    print(f"[INFO] Past cov dim: {dims['past_cov_dim']} | Future cov dim: {dims['future_cov_dim']}")

    # Instantiate Model
    model = TiDE(
        lookback_len=lookback_len,
        horizon=horizon,
        past_cov_dim=dims["past_cov_dim"],
        future_cov_dim=dims["future_cov_dim"],
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        decoder_dim=decoder_dim,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dropout=dropout,
        use_revin=use_revin,
        use_past_covariates=use_past_covariates,
        use_future_covariates=use_future_covariates,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] TiDE trainable parameters: {total_params:,}")

    # Loss, Optimizer, Scheduler
    criterion = get_loss_fn(loss_name)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_tide_model.pt")

    best_val_wape = float("inf")
    best_metrics = {}
    epochs_no_improve = 0

    print("\n" + "=" * 80)
    print(f"{'Epoch':^7} | {'Train Loss':^10} | {'Val Loss':^10} | {'Val WAPE':^10} | {'Val MAE':^10} | {'Val RMSE':^10} | {'LR':^9}")
    print("=" * 80)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
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

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * x_past_target.size(0)
            n_train += x_past_target.size(0)

        scheduler.step()
        avg_train_loss = train_loss / max(n_train, 1)

        # Validation
        avg_val_loss, metrics = evaluate(model, val_loader, criterion, device)
        val_wape = metrics["WAPE"]
        current_lr = optimizer.param_groups[0]["lr"]

        is_best = val_wape < best_val_wape
        if is_best:
            best_val_wape = val_wape
            best_metrics = metrics.copy()
            epochs_no_improve = 0

            # Save best checkpoint
            checkpoint_data = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_wape": val_wape,
                "metrics": metrics,
                "config": {
                    "lookback_len": lookback_len,
                    "horizon": horizon,
                    "past_cov_dim": dims["past_cov_dim"],
                    "future_cov_dim": dims["future_cov_dim"],
                    "feature_dim": feature_dim,
                    "hidden_dim": hidden_dim,
                    "decoder_dim": decoder_dim,
                    "num_encoder_layers": num_encoder_layers,
                    "num_decoder_layers": num_decoder_layers,
                    "dropout": dropout,
                    "use_revin": use_revin,
                    "use_past_covariates": use_past_covariates,
                    "use_future_covariates": use_future_covariates,
                },
                "preprocessor_means": preprocessor.column_means,
            }
            torch.save(checkpoint_data, best_checkpoint_path)
            # Also save directly to submission/checkpoint.pt
            torch.save(checkpoint_data, "submission/checkpoint.pt")
        else:
            epochs_no_improve += 1

        best_marker = " *" if is_best else ""
        epoch_dur = time.time() - epoch_start
        print(
            f"{epoch:^7} | {avg_train_loss:^10.4f} | {avg_val_loss:^10.4f} | {val_wape:^10.4f} | {metrics['MAE']:^10.4f} | {metrics['RMSE']:^10.4f} | {current_lr:^9.2e}{best_marker} ({epoch_dur:.1f}s)"
        )

        if epochs_no_improve >= patience:
            print(f"\n[INFO] Early stopping triggered after {epoch} epochs (no improvement for {patience} epochs).")
            break

    print("=" * 80)
    print(f"\n[SUCCESS] Best Model Validation WAPE: {best_val_wape:.4f}")
    for k, v in best_metrics.items():
        if k != "WAPE":
            print(f"  - {k}: {v:.4f}")
    print(f"[INFO] Checkpoint saved to: {best_checkpoint_path}")

    return best_metrics


def main():
    parser = argparse.ArgumentParser(description="Train TiDE with RevIN on operational load forecasting.")
    parser.add_argument("--train-csv", type=str, default="data/benchmark/train.csv")
    parser.add_argument("--checkpoint-dir", type=str, default="experiments/checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--decoder-dim", type=int, default=128)
    parser.add_argument("--feature-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--loss", type=str, default="huber", choices=["huber", "l1", "mse"])
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_model(
        train_csv_path=args.train_csv,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        decoder_dim=args.decoder_dim,
        feature_dim=args.feature_dim,
        dropout=args.dropout,
        loss_name=args.loss,
        patience=args.patience,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
