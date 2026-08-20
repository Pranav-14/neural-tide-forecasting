"""Regression tests for input-layout handling in the submission inference script.

The private test directory layout is not known in advance, so `predict.py` must not
depend on a particular filename, on a covariate column being present, or on the
forecast index arriving in chronological order. Each test below corresponds to a
failure mode that previously either crashed the run or silently misaligned the
rollout, which would have produced a zero score.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "benchmark"


def _load_predict_module():
    """Import submission/predict.py in isolation.

    The repository root also contains a `src` package, so it must be shadowed by
    `submission/src` while the module executes; otherwise `src.model` resolves to the
    training-side module, which exports `TiDE` rather than `ForecastModel`.
    """
    sub_dir = ROOT / "submission"
    saved_path = list(sys.path)
    saved_src = {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}
    for key in saved_src:
        del sys.modules[key]
    sys.path.insert(0, str(sub_dir))
    try:
        spec = importlib.util.spec_from_file_location("submission_predict", sub_dir / "predict.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for key in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
            del sys.modules[key]
        sys.modules.update(saved_src)
        sys.path[:] = saved_path
    return module


predict = _load_predict_module()

pytestmark = pytest.mark.skipif(
    not (DATA / "train.csv").exists(), reason="benchmark data not present"
)


@pytest.fixture(scope="module")
def frames():
    return (
        pd.read_csv(DATA / "train.csv"),
        pd.read_csv(DATA / "validation_input.csv"),
        pd.read_csv(DATA / "forecast_index_validation.csv"),
    )


def test_history_found_under_unexpected_filename(tmp_path, frames):
    """History must be discovered even when it is not called train.csv."""
    train, future, index = frames
    train.to_csv(tmp_path / "context_window.csv", index=False)
    future.to_csv(tmp_path / "test_input.csv", index=False)
    index.to_csv(tmp_path / "forecast_index_test.csv", index=False)

    fi, fut, hist = predict.load_inputs(tmp_path)
    assert len(fi) == len(index)
    assert "target" in hist.columns and hist["target"].notna().all()
    assert len(hist) == len(train)


def test_history_embedded_in_future_input(tmp_path, frames):
    """History rows shipped inside the future-input file must be harvested, not forecast."""
    train, future, index = frames
    tail = train.groupby("series_id", group_keys=False).tail(200)
    unlabelled = future.copy()
    unlabelled["target"] = np.nan
    pd.concat([tail, unlabelled], ignore_index=True).to_csv(tmp_path / "test_input.csv", index=False)
    index.to_csv(tmp_path / "forecast_index_test.csv", index=False)

    fi, fut, hist = predict.load_inputs(tmp_path)
    # Only labelled rows become history; the unlabelled rows stay in the future frame.
    assert hist["target"].notna().all()
    assert len(hist) == len(tail)
    assert "target" not in fut.columns


def test_missing_history_raises_informative_error(tmp_path, frames):
    """With no target anywhere, fail loudly and name the files that were inspected."""
    _, future, index = frames
    future.to_csv(tmp_path / "test_input.csv", index=False)
    index.to_csv(tmp_path / "forecast_index_test.csv", index=False)

    with pytest.raises(FileNotFoundError, match="Missing historical target data"):
        predict.load_inputs(tmp_path)


def test_absent_covariate_column_is_backfilled(frames):
    """A covariate missing from the input schema must not raise a KeyError."""
    _, future, _ = frames
    stripped = future.drop(columns=["shock_risk"])
    means = {"shock_risk": 0.25}

    out = predict.preprocess_dataframe(stripped, means)
    assert "shock_risk" in out.columns
    assert np.isclose(out["shock_risk"].iloc[0], 0.25)
    for col in predict.ALL_COVARIATE_COLUMNS:
        assert col in out.columns


def test_rollout_is_chronological_regardless_of_index_order(frames):
    """A shuffled forecast index must yield the same prediction for each timestamp."""
    train, future, index = frames
    series = list(index["series_id"].unique()[:2])
    train_s = train[train["series_id"].isin(series)]
    future_s = future[future["series_id"].isin(series)]
    index_s = index[index["series_id"].isin(series)].reset_index(drop=True)

    import torch

    from src.model import TiDE

    torch.manual_seed(0)
    model = TiDE(lookback_len=168, horizon=24, past_cov_dim=32, future_cov_dim=25)
    means = {c: 0.0 for c in predict.ALL_COVARIATE_COLUMNS}

    ordered = predict.predict_rolling(model, train_s, future_s, index_s, means)
    shuffled_index = index_s.sample(frac=1, random_state=0).reset_index(drop=True)
    shuffled = predict.predict_rolling(model, train_s, future_s, shuffled_index, means)

    # Row order must follow the supplied index, but each key must carry the same value.
    assert shuffled[["series_id", "timestamp"]].equals(shuffled_index[["series_id", "timestamp"]])
    merged = ordered.merge(shuffled, on=["series_id", "timestamp"], suffixes=("_a", "_b"))
    assert len(merged) == len(index_s)
    np.testing.assert_allclose(merged["prediction_a"], merged["prediction_b"], rtol=0, atol=0)
