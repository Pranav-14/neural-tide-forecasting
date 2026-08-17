"""Inference entrypoint for final private evaluation.

Students should replace the placeholder logic with their trained model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.model import ForecastModel


def load_forecast_index(input_dir: Path) -> pd.DataFrame:
    """Load the rows that need predictions."""
    candidates = [
        input_dir / "forecast_index_test.csv",
        input_dir / "forecast_index_validation.csv",
    ]
    for forecast_index in candidates:
        if forecast_index.exists():
            return pd.read_csv(forecast_index)
    expected = ", ".join(path.name for path in candidates)
    raise FileNotFoundError(f"Expected one of {expected} in input_dir.")


def main() -> None:
    """Load a checkpoint and write placeholder private-test predictions."""
    parser = argparse.ArgumentParser(description="Generate private test predictions.")
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_file", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    forecast_index = load_forecast_index(args.input_dir)
    model = ForecastModel()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        raise ValueError("Checkpoint must be a state_dict or a dict containing `state_dict`.")

    model.eval()

    # Replace this with feature loading, preprocessing, and model inference.
    # The placeholder writes zeros so the file contract is explicit.
    predictions = forecast_index[["series_id", "timestamp"]].copy()
    predictions["prediction"] = 0.0

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()
