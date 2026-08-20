"""BATADAL Water Distribution Network SCADA Cross-Domain Forecasting Pipeline."""

from __future__ import annotations

import argparse
import os
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
from torch.utils.data import DataLoader, Dataset

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.model import RevIN, ResidualBlock
from src.train import set_seed
from src.utils import compute_metrics


# Define BATADAL sensor targets and features
TANK_TARGETS = ["L_T1", "L_T2", "L_T3", "L_T4", "L_T5", "L_T6", "L_T7"]
FLOW_TARGETS = [
    "F_PU1", "F_PU2", "F_PU3", "F_PU4", "F_PU5",
    "F_PU6", "F_PU7", "F_PU8", "F_PU9", "F_PU10", "F_PU11", "F_V2"
]
PRIMARY_TARGETS = TANK_TARGETS + FLOW_TARGETS

PRESSURE_COVARIATES = [
    "P_J280", "P_J269", "P_J300", "P_J256", "P_J289",
    "P_J415", "P_J302", "P_J306", "P_J307", "P_J317", "P_J14", "P_J422"
]
STATUS_COVARIATES = [
    "S_PU1", "S_PU2", "S_PU3", "S_PU4", "S_PU5",
    "S_PU6", "S_PU7", "S_PU8", "S_PU9", "S_PU10", "S_PU11", "S_V2"
]


def load_and_preprocess_batadal(
    csv_path: str = "data/batadal/training_dataset_1.csv"
) -> pd.DataFrame:
    """Load BATADAL dataset, clean attack periods, and generate temporal encodings."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Filter out attack periods if flag exists
    if "ATT_FLAG" in df.columns:
        df = df[df["ATT_FLAG"] == 0].copy()

    # Parse datetime: format DD/MM/YY HH
    df["dt"] = pd.to_datetime(df["DATETIME"], format="%d/%m/%y %H", errors="coerce")
    df = df.sort_values("dt").reset_index(drop=True)

    # Cyclical calendar features
    hours = df["dt"].dt.hour.to_numpy()
    dows = df["dt"].dt.dayofweek.to_numpy()

    df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * dows / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dows / 7.0)
    df["is_weekend"] = (dows >= 5).astype(np.float32)

    return df


class BatadalDataset(Dataset):
    """Sliding-window PyTorch Dataset for multivariate multi-target BATADAL forecasting."""

    def __init__(
        self,
        targets_matrix: np.ndarray,       # (T, num_targets)
        past_covs_matrix: np.ndarray,     # (T, num_past_covs)
        future_covs_matrix: np.ndarray,   # (T, num_future_covs)
        lookback_len: int = 168,
        horizon: int = 24,
        stride: int = 1,
    ):
        self.lookback_len = lookback_len
        self.horizon = horizon
        self.total_window_len = lookback_len + horizon

        self.targets = targets_matrix.astype(np.float32)
        self.past_covs = past_covs_matrix.astype(np.float32)
        self.future_covs = future_covs_matrix.astype(np.float32)

        n_steps = len(self.targets)
        self.valid_starts = list(range(0, n_steps - self.total_window_len + 1, stride))

    def __len__(self) -> int:
        return len(self.valid_starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t = self.valid_starts[idx]

        # Past window [t : t + lookback_len]
        x_past_target = torch.from_numpy(self.targets[t : t + self.lookback_len])          # (L, num_targets)
        x_past_cov = torch.from_numpy(self.past_covs[t : t + self.lookback_len])            # (L, D_cov)

        # Future window [t + lookback_len : t + total_window_len]
        x_future_cov = torch.from_numpy(self.future_covs[t + self.lookback_len : t + self.total_window_len])  # (H, D_future)
        y_future_target = torch.from_numpy(self.targets[t + self.lookback_len : t + self.total_window_len])  # (H, num_targets)

        return x_past_target, x_past_cov, x_future_cov, y_future_target


class BatadalTiDE(nn.Module):
    """Covariate-Free / Dynamic Temporal TiDE model for multi-target forecasting."""

    def __init__(
        self,
        num_targets: int = 19,
        lookback_len: int = 168,
        horizon: int = 24,
        past_cov_dim: int = 29,
        future_cov_dim: int = 5,
        feature_dim: int = 16,
        hidden_dim: int = 256,
        decoder_dim: int = 128,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dropout: float = 0.1,
        use_revin: bool = True,
    ):
        super().__init__()
        self.num_targets = num_targets
        self.lookback_len = lookback_len
        self.horizon = horizon
        self.use_revin = use_revin

        # 1. Multi-target RevIN Normalization
        if self.use_revin:
            self.revin = RevIN(num_features=num_targets, affine=True)

        # 2. Covariate Projections
        self.past_cov_proj = nn.Linear(past_cov_dim, feature_dim)
        self.future_cov_proj = nn.Linear(future_cov_dim, feature_dim)

        # 3. Dense Encoder
        encoder_input_dim = lookback_len * num_targets + lookback_len * feature_dim + horizon * feature_dim
        encoder_layers = [ResidualBlock(encoder_input_dim, hidden_dim, dropout=dropout)]
        for _ in range(num_encoder_layers - 1):
            encoder_layers.append(ResidualBlock(hidden_dim, hidden_dim, dropout=dropout))
        self.encoder = nn.Sequential(*encoder_layers)

        # 4. Dense Decoder
        self.decoder_dim = decoder_dim
        decoder_layers = [ResidualBlock(hidden_dim, decoder_dim * horizon, dropout=dropout)]
        for _ in range(num_decoder_layers - 1):
            decoder_layers.append(ResidualBlock(decoder_dim * horizon, decoder_dim * horizon, dropout=dropout))
        self.decoder = nn.Sequential(*decoder_layers)

        # 5. Temporal Head (projects to all targets per step)
        self.temporal_head = nn.Sequential(
            nn.Linear(decoder_dim + feature_dim, decoder_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_dim // 2, num_targets),
        )

        # 6. Direct Linear Residual Skip (L -> H per target)
        self.residual_skip = nn.Linear(lookback_len, horizon)

    def forward(
        self,
        x_past_target: torch.Tensor,       # (B, L, num_targets)
        x_past_cov: torch.Tensor,          # (B, L, past_cov_dim)
        x_future_cov: torch.Tensor,        # (B, H, future_cov_dim)
    ) -> torch.Tensor:
        B = x_past_target.size(0)

        # Step 1: Multi-target RevIN Normalization
        if self.use_revin:
            norm_target = self.revin.normalize(x_past_target)  # (B, L, num_targets)
        else:
            norm_target = x_past_target

        # Step 2: Feature Projections
        proj_past = self.past_cov_proj(x_past_cov)        # (B, L, feature_dim)
        proj_future = self.future_cov_proj(x_future_cov)  # (B, H, feature_dim)

        # Step 3: Dense Encoder
        enc_in = torch.cat([
            norm_target.reshape(B, -1),
            proj_past.reshape(B, -1),
            proj_future.reshape(B, -1),
        ], dim=-1)
        global_embed = self.encoder(enc_in)  # (B, hidden_dim)

        # Step 4: Dense Decoder
        dec_out = self.decoder(global_embed).reshape(B, self.horizon, self.decoder_dim)

        # Step 5: Temporal Head per step
        step_inputs = torch.cat([dec_out, proj_future], dim=-1)
        head_out = self.temporal_head(step_inputs)  # (B, H, num_targets)

        # Step 6: Direct Linear Skip across time dimension per target
        # Transpose to (B, num_targets, L) -> linear(L->H) -> transpose to (B, H, num_targets)
        res_out = self.residual_skip(norm_target.transpose(1, 2)).transpose(1, 2)
        y_norm_pred = head_out + res_out

        # Step 7: RevIN Denormalize
        if self.use_revin:
            y_pred = self.revin.denormalize(y_norm_pred)
        else:
            y_pred = y_norm_pred

        return y_pred


def evaluate_batadal_baselines(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    targets: List[str],
    lookback_len: int = 168,
    horizon: int = 24,
) -> Dict[str, Dict[str, float]]:
    """Compute heuristic baseline metrics on BATADAL validation split."""
    val_targets = val_df[targets].to_numpy()
    train_targets = train_df[targets].to_numpy()

    # Align sliding windows for validation evaluation
    n_val_steps = len(val_targets)
    all_y_true = []
    all_naive_pred = []
    all_lag24_pred = []
    all_lag168_pred = []
    all_mean_pred = []

    full_history = list(train_targets[-lookback_len:])

    # Global and hourly means from train
    train_df_copy = train_df.copy()
    train_df_copy["_hour"] = train_df_copy["dt"].dt.hour
    hourly_means = train_df_copy.groupby("_hour")[targets].mean().to_dict(orient="index")

    for t in range(0, n_val_steps - horizon + 1, horizon):
        y_true = val_targets[t : t + horizon]
        all_y_true.append(y_true)

        # 1. Naive last-value: repeat last observed target vector across horizon
        last_val = full_history[-1]
        all_naive_pred.append(np.tile(last_val, (horizon, 1)))

        # 2. Lag-24: repeat last 24 hours
        lag24 = np.array(full_history[-24:])
        all_lag24_pred.append(lag24)

        # 3. Lag-168: repeat last 168 hours
        lag168 = np.array(full_history[-lookback_len:])[:horizon]
        all_lag168_pred.append(lag168)

        # 4. Seasonal mean by hour
        val_hours = val_df["dt"].iloc[t : t + horizon].dt.hour.to_numpy()
        seasonal_pred = np.array([[hourly_means[h][col] for col in targets] for h in val_hours])
        all_mean_pred.append(seasonal_pred)

        # Append true observations to rolling buffer
        full_history.extend(y_true)

    y_true_arr = np.concatenate(all_y_true, axis=0).flatten()

    results = {}
    preds_map = {
        "naive_last_value": np.concatenate(all_naive_pred, axis=0).flatten(),
        "lag24_repeat": np.concatenate(all_lag24_pred, axis=0).flatten(),
        "lag168_repeat": np.concatenate(all_lag168_pred, axis=0).flatten(),
        "seasonal_mean": np.concatenate(all_mean_pred, axis=0).flatten(),
    }

    for name, p_arr in preds_map.items():
        results[name] = compute_metrics(y_true_arr, p_arr)

    return results


def run_batadal_experiments(
    csv_path: str = "data/batadal/training_dataset_1.csv",
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    lookback_len: int = 168,
    horizon: int = 24,
    train_ratio: float = 0.80,
    seed: int = 42,
):
    """Run full BATADAL cross-domain benchmark: Baselines vs TiDE + RevIN."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1. Load data
    print("[INFO] Loading and preprocessing BATADAL sensor dataset...")
    df = load_and_preprocess_batadal(csv_path)
    print(f"[INFO] Total clean normal-operation rows: {len(df):,} hours (~{len(df)/24:.1f} days)")

    # 2. Features and Targets
    target_cols = PRIMARY_TARGETS  # 7 tank levels + 12 flow rates = 19 targets
    temporal_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]
    past_cov_cols = PRESSURE_COVARIATES + STATUS_COVARIATES + temporal_cols  # 12 + 12 + 5 = 29
    future_cov_cols = temporal_cols  # 5 temporal features

    print(f"[INFO] Multi-target count: {len(target_cols)} (7 Tanks + 12 Flow Rates)")
    print(f"[INFO] Past covariate count: {len(past_cov_cols)} | Future covariate count: {len(future_cov_cols)}")

    # 3. Train/Val Split (80% / 20%)
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)

    print(f"[INFO] Training rows: {len(train_df):,} | Validation rows: {len(val_df):,}")

    # 4. Evaluate Baselines
    print("\n[INFO] Evaluating heuristic baselines on BATADAL validation split...")
    baseline_results = evaluate_batadal_baselines(
        train_df, val_df, target_cols, lookback_len=lookback_len, horizon=horizon
    )

    # 5. Build PyTorch Datasets
    train_dataset = BatadalDataset(
        targets_matrix=train_df[target_cols].to_numpy(),
        past_covs_matrix=train_df[past_cov_cols].to_numpy(),
        future_covs_matrix=train_df[future_cov_cols].to_numpy(),
        lookback_len=lookback_len,
        horizon=horizon,
        stride=1,
    )
    val_dataset = BatadalDataset(
        targets_matrix=val_df[target_cols].to_numpy(),
        past_covs_matrix=val_df[past_cov_cols].to_numpy(),
        future_covs_matrix=val_df[future_cov_cols].to_numpy(),
        lookback_len=lookback_len,
        horizon=horizon,
        stride=horizon,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"[INFO] Train sliding windows: {len(train_dataset):,} | Val sliding windows: {len(val_dataset):,}")

    # 6. Instantiate TiDE Model
    model = BatadalTiDE(
        num_targets=len(target_cols),
        lookback_len=lookback_len,
        horizon=horizon,
        past_cov_dim=len(past_cov_cols),
        future_cov_dim=len(future_cov_cols),
        feature_dim=16,
        hidden_dim=256,
        decoder_dim=128,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.1,
        use_revin=True,
    ).to(device)

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    print("\n" + "=" * 80)
    print(f"{'Epoch':^7} | {'Train Loss':^10} | {'Val Loss':^10} | {'Val WAPE':^10} | {'Val MAE':^10} | {'Val RMSE':^10}")
    print("=" * 80)

    best_val_wape = float("inf")
    best_metrics = {}

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

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * x_past_target.size(0)
            n_train += x_past_target.size(0)

        scheduler.step()
        avg_train_loss = train_loss / max(n_train, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x_past_target, x_past_cov, x_future_cov, y_future_target in val_loader:
                x_past_target = x_past_target.to(device)
                x_past_cov = x_past_cov.to(device)
                x_future_cov = x_future_cov.to(device)
                y_future_target = y_future_target.to(device)

                y_pred = model(x_past_target, x_past_cov, x_future_cov)
                # Clamp non-negative
                y_pred = torch.clamp(y_pred, min=0.0)
                loss = criterion(y_pred, y_future_target)

                val_loss += loss.item() * x_past_target.size(0)
                n_val += x_past_target.size(0)

                all_preds.append(y_pred.cpu().numpy())
                all_targets.append(y_future_target.cpu().numpy())

        avg_val_loss = val_loss / max(n_val, 1)
        y_pred_flat = np.concatenate(all_preds, axis=0).flatten()
        y_true_flat = np.concatenate(all_targets, axis=0).flatten()

        metrics = compute_metrics(y_true_flat, y_pred_flat)
        val_wape = metrics["WAPE"]

        is_best = val_wape < best_val_wape
        if is_best:
            best_val_wape = val_wape
            best_metrics = metrics.copy()
            # Save checkpoint
            os.makedirs("experiments/checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "experiments/checkpoints/batadal_tide_model.pt")

        best_marker = " *" if is_best else ""
        epoch_dur = time.time() - epoch_start
        print(
            f"{epoch:^7} | {avg_train_loss:^10.4f} | {avg_val_loss:^10.4f} | {val_wape:^10.4f} | {metrics['MAE']:^10.4f} | {metrics['RMSE']:^10.4f}{best_marker} ({epoch_dur:.1f}s)"
        )

    # Summary Table
    print("\n" + "=" * 80)
    print("=== BATADAL CROSS-DOMAIN VALIDATION BENCHMARKS ===")
    print("=" * 80)

    results_table = []
    for name, m in baseline_results.items():
        results_table.append({
            "Model": name,
            "WAPE": f"{m['WAPE']:.4f}",
            "MAE": f"{m['MAE']:.4f}",
            "RMSE": f"{m['RMSE']:.4f}",
            "sMAPE": f"{m['sMAPE']:.2f}%",
        })

    results_table.append({
        "Model": "TiDE + RevIN (Ours)",
        "WAPE": f"{best_metrics['WAPE']:.4f}",
        "MAE": f"{best_metrics['MAE']:.4f}",
        "RMSE": f"{best_metrics['RMSE']:.4f}",
        "sMAPE": f"{best_metrics['sMAPE']:.2f}%",
    })

    res_df = pd.DataFrame(results_table)
    print(res_df.to_string(index=False))
    print("=" * 80)

    # Save results to CSV for reporting
    res_df.to_csv("experiments/batadal_results.csv", index=False)
    print("[SUCCESS] BATADAL results saved to experiments/batadal_results.csv")


if __name__ == "__main__":
    run_batadal_experiments()
