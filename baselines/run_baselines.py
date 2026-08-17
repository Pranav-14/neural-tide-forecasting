"""Run simple baseline forecasts."""

from __future__ import annotations

import argparse
from pathlib import Path

from baselines import write_baselines


def main() -> None:
    """CLI entrypoint for writing baseline prediction CSVs."""
    parser = argparse.ArgumentParser(description="Generate simple forecasting baseline CSVs.")
    parser.add_argument("--train", required=True, type=Path, help="Training CSV.")
    parser.add_argument("--forecast-index", required=True, type=Path, help="Forecast index CSV.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for baseline prediction CSVs.")
    args = parser.parse_args()

    write_baselines(args.train, args.forecast_index, args.output_dir)
    print(f"Wrote baseline predictions to {args.output_dir}")


if __name__ == "__main__":
    main()
