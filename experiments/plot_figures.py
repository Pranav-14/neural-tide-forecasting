"""Generate publication figures for the research report.

Every curve plotted here is real model output, scored on the same window it is drawn
against. Figures 1 and 2 come from `experiments/backtest_recursive.py`; Figure 3 runs the
trained BATADAL checkpoint over its held-out split.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))

from experiments.run_batadal import (
    PRESSURE_COVARIATES,
    PRIMARY_TARGETS,
    STATUS_COVARIATES,
    BatadalDataset,
    BatadalTiDE,
    load_and_preprocess_batadal,
)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["figure.dpi"] = 300

OUTPUT_DIR = Path("report/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VAL_HORIZON = 336
BACKTEST_PREDS = Path("experiments/predictions/tide_revin_backtest.csv")
BENCHMARK_RESULTS = Path("experiments/benchmark_results.csv")


def _save(fig, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}.png and .pdf")


def plot_benchmark_forecasts(series_ids=("unit_000", "unit_005")) -> None:
    """Figure 1: recursive 336-hour rollout against ground truth for the identical window."""
    print("Generating Figure 1: Benchmark Forecast Curves...")
    if not BACKTEST_PREDS.exists():
        raise FileNotFoundError(
            f"{BACKTEST_PREDS} not found. Run `python experiments/backtest_recursive.py` first."
        )

    train_df = pd.read_csv("data/benchmark/train.csv")
    train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
    preds_df = pd.read_csv(BACKTEST_PREDS)
    preds_df["timestamp"] = pd.to_datetime(preds_df["timestamp"])

    fig, axes = plt.subplots(len(series_ids), 1, figsize=(10, 5.5), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, s_id in zip(axes, series_ids):
        truth = (
            train_df[train_df["series_id"] == s_id]
            .sort_values("timestamp")
            .iloc[-VAL_HORIZON:][["timestamp", "target"]]
        )
        pred = preds_df[preds_df["series_id"] == s_id].sort_values("timestamp")

        # Both curves must describe identical timestamps, or the figure is meaningless.
        merged = truth.merge(pred, on="timestamp", how="inner")
        assert len(merged) == VAL_HORIZON, (
            f"{s_id}: {len(merged)} overlapping timestamps, expected {VAL_HORIZON}."
        )

        hours = np.arange(1, VAL_HORIZON + 1)
        wape = np.abs(merged["target"] - merged["prediction"]).sum() / np.abs(merged["target"]).sum()
        ax.plot(hours, merged["target"], label="Ground Truth", color="#1f77b4", linewidth=1.5, alpha=0.85)
        ax.plot(
            hours, merged["prediction"], label="Neural-TiDE (recursive)", color="#d62728",
            linewidth=1.5, linestyle="--", alpha=0.9,
        )
        ax.set_ylabel("Operational Load")
        ax.set_title(f"336-Hour Recursive Rollout: {s_id}  (WAPE = {wape:.4f})", fontweight="bold", pad=6)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", frameon=True)

    axes[-1].set_xlabel("Forecast Horizon (Hours)")
    plt.tight_layout()
    _save(fig, "fig1_forecast_curves")


def plot_model_comparison_bars() -> None:
    """Figure 2: metric comparison read directly from the backtest results file."""
    print("Generating Figure 2: Metric Comparison Bar Chart...")
    if not BENCHMARK_RESULTS.exists():
        raise FileNotFoundError(
            f"{BENCHMARK_RESULTS} not found. Run `python experiments/backtest_recursive.py` first."
        )

    pretty = {
        "naive_last_value": "Naive Last-Value",
        "lag168_repeat": "Lag-168 (Weekly)",
        "lag24_repeat": "Lag-24 (Daily)",
        "seasonal_mean": "Seasonal Mean",
        "TiDE + RevIN (Ours)": "Neural-TiDE (Ours)",
    }
    res = pd.read_csv(BENCHMARK_RESULTS)
    res = res[res["Model"].isin(pretty)].copy()
    res["label"] = res["Model"].map(pretty)
    res = res.sort_values("WAPE", ascending=False)

    x = np.arange(len(res))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.5))
    groups = [
        ax.bar(x - width, res["WAPE"], width, label="WAPE (Primary)", color="#2ca02c", alpha=0.9),
        ax.bar(x, res["MAE"], width, label="MAE", color="#1f77b4", alpha=0.9),
        ax.bar(x + width, res["RMSE"], width, label="RMSE", color="#ff7f0e", alpha=0.9),
    ]
    ax.set_ylabel("Error Metric Value (lower is better)")
    ax.set_title("Operational Load Benchmark: Recursive 336-Hour Rollout", fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(res["label"], rotation=15, ha="right")
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.7)
    for group in groups:
        group[-1].set_edgecolor("black")
        group[-1].set_linewidth(1.5)

    plt.tight_layout()
    _save(fig, "fig2_model_comparison")


def plot_batadal_forecasts(train_ratio: float = 0.80, lookback_len: int = 168, horizon: int = 24) -> None:
    """Figure 3: real BATADAL model predictions against held-out sensor telemetry."""
    print("Generating Figure 3: BATADAL Water SCADA Forecasts...")
    ckpt_path = Path("experiments/checkpoints/batadal_tide_model.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found. Run `python experiments/run_batadal.py` first.")

    df = load_and_preprocess_batadal()
    temporal = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]
    past_cols = PRESSURE_COVARIATES + STATUS_COVARIATES + temporal
    val_df = df.iloc[int(len(df) * train_ratio):].reset_index(drop=True)

    dataset = BatadalDataset(
        targets_matrix=val_df[PRIMARY_TARGETS].to_numpy(),
        past_covs_matrix=val_df[past_cols].to_numpy(),
        future_covs_matrix=val_df[temporal].to_numpy(),
        lookback_len=lookback_len,
        horizon=horizon,
        stride=horizon,
    )
    model = BatadalTiDE(
        num_targets=len(PRIMARY_TARGETS),
        lookback_len=lookback_len,
        horizon=horizon,
        past_cov_dim=len(past_cols),
        future_cov_dim=len(temporal),
    )
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    preds, truths = [], []
    with torch.no_grad():
        for x_target, x_past, x_future, y in DataLoader(dataset, batch_size=64):
            preds.append(torch.clamp(model(x_target, x_past, x_future), min=0.0).numpy())
            truths.append(y.numpy())
    y_pred = np.concatenate(preds).reshape(-1, len(PRIMARY_TARGETS))
    y_true = np.concatenate(truths).reshape(-1, len(PRIMARY_TARGETS))

    n_show = 168  # one week of the held-out period
    hours = np.arange(1, n_show + 1)
    panels = [
        ("L_T1", "Water Level (m)", "Tank Level Forecast (L_T1)", "#1f77b4", "#d62728"),
        ("F_PU1", "Flow Rate (LPS)", "Pumping Flow Rate Forecast (F_PU1)", "#2ca02c", "#9467bd"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    for ax, (col, ylabel, title, c_true, c_pred) in zip(axes, panels):
        k = PRIMARY_TARGETS.index(col)
        truth, pred = y_true[:n_show, k], y_pred[:n_show, k]
        wape = np.abs(truth - pred).sum() / (np.abs(truth).sum() + 1e-8)
        ax.plot(hours, truth, color=c_true, label=f"Observed {col}", linewidth=1.5)
        ax.plot(hours, pred, color=c_pred, linestyle="--", label="Neural-TiDE Prediction", linewidth=1.5)
        ax.set_ylabel(ylabel)
        ax.set_title(f"BATADAL Water SCADA: {title}  (WAPE = {wape:.4f})", fontweight="bold")
        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, linestyle=":", alpha=0.6)

    axes[1].set_xlabel("Held-Out Time Index (Hours)")
    plt.tight_layout()
    _save(fig, "fig3_batadal_forecast")


if __name__ == "__main__":
    plot_benchmark_forecasts()
    plot_model_comparison_bars()
    plot_batadal_forecasts()
    print("[SUCCESS] All report figures generated from real model output.")
