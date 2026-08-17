# Neural-TiDE: Multivariate Long-Horizon Time-Series Forecasting

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An efficient, pure PyTorch implementation of **TiDE (Time-series Dense Encoder)** enhanced with **RevIN (Reversible Instance Normalization)** for long-horizon multivariate time-series forecasting with dynamic future covariates and static metadata.

---

## Key Highlights

* **High Computational Efficiency:** MLP-based encoder-decoder design that is $5\text{--}10\times$ faster than standard Transformer architectures while matching or outperforming them on long horizons.
* **Native Future Covariate Support:** Direct encoding of dynamic known-future signals (e.g., workload forecasts, scheduled maintenance, cyclical calendar encodings) into temporal decoder projections.
* **Non-Stationarity & Scale Robustness:** Integrated **RevIN** handles distribution shifts and heterogeneous scales across diverse correlated time-series units.
* **Multi-Step Iterative Rollout Engine:** Built-in recursive sliding window rollout designed for arbitrary multi-week forecast horizons.
* **Modular Pipeline:** Extensible dataset loaders with robust imputation for missing forecast signals, evaluation metric suites (WAPE, MAE, RMSE, MAPE, sMAPE), and baseline benchmarks.

---

## Architecture Overview

```text
                               ┌────────────────────────────────┐
                               │  Historical Target (L = 168)   │
                               └───────────────┬────────────────┘
                                               │
                                       [ RevIN Normalization ]
                                               │
  ┌─────────────────────────────┐              ▼
  │ Past Covariates (L x D_cov) ├────► [ Feature Projection ] ◄──── Dynamic Future Covariates (H x D_cov)
  └─────────────────────────────┘              │
                                               ▼
                                    ┌───────────────────────┐
                                    │     Dense Encoder     │ (Residual MLP Blocks)
                                    └──────────┬────────────┘
                                               │
                                               ▼ Global Embedding (e)
                                    ┌───────────────────────┐
                                    │   Temporal Decoder    │ ◄──── Future Covariates (per step)
                                    └──────────┬────────────┘
                                               │
                                               ▼
                                      ( + Residual Skip ) ◄─────── Direct Linear Skip from History
                                               │
                                      [ RevIN Denormalize ]
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │     Output Forecast (H = 24)   │
                               └────────────────────────────────┘
```

---

## Project Structure

```text
neural-tide-forecasting/
├── src/                      # Core neural architecture & pipeline
│   ├── model.py              # TiDE architecture & RevIN implementation
│   ├── dataset.py            # Sliding-window dataset generator & imputation
│   ├── train.py              # Training loop, optimizer, and learning rate schedulers
│   ├── inference.py          # Multi-step rolling forecast engine
│   └── utils.py              # Evaluation metric suite (WAPE, MAE, RMSE, sMAPE)
│
├── baselines/                # Reference heuristic & statistical baselines
│   ├── baselines.py          # Naive last-value, Lag-24, Lag-168, Seasonal Mean
│   └── run_baselines.py      # Baseline generation script
│
├── experiments/              # Local validation backtests, EDA, and ablations
│   ├── eda.py                # Dataset analysis and statistics
│   └── eval_local_baselines.py # Local validation benchmark runner
│
├── submission/               # Production inference package & checkpoint interface
│   ├── predict.py            # Standardized CLI entrypoint
│   ├── requirements.txt      # Minimal runtime dependencies
│   └── src/
│
└── report/                   # LaTeX source & bibliography for empirical report
```

---

## Getting Started

### 1. Installation
Clone the repository and install required dependencies:
```bash
git clone https://github.com/Pranav-14/neural-tide-forecasting.git
cd neural-tide-forecasting
pip install -r submission/requirements.txt
```

### 2. Run Baselines
Generate standard heuristic forecasts (Naive Last-Value, Lag-24, Lag-168, Seasonal Mean):
```bash
python baselines/run_baselines.py \
    --train data/benchmark/train.csv \
    --forecast-index data/benchmark/forecast_index_validation.csv \
    --output-dir experiments/baseline_predictions
```

### 3. Evaluate Baseline Benchmarks
Run a local validation backtest across all metrics:
```bash
python experiments/eval_local_baselines.py
```

---

## Benchmark Results (Local Validation Split)

Evaluated on the final 336-hour hold-out window:

| Model | WAPE (Primary) ↓ | MAE ↓ | RMSE ↓ | sMAPE ↓ |
|---|---|---|---|---|
| **`naive_last_value`** | 0.5450 | 5.8214 | 7.5521 | 65.73% |
| **`lag168_repeat`** | 0.4549 | 4.8584 | 6.3610 | 54.43% |
| **`lag24_repeat`** | 0.4180 | 4.4646 | 5.8853 | 49.50% |
| **`seasonal_mean`** | 0.3140 | 3.3541 | 4.5432 | 34.70% |
| **TiDE + RevIN (Ours)** | *Training...* | *—* | *—* | *—* |

---

## References

1. **TiDE:** Das, A., Kong, W., Leach, A., Mathur, S., Sen, R., & Yu, R. (2023). *Long-term Forecasting with TiDE: Time-series Dense Encoder*. Transactions on Machine Learning Research (TMLR).
2. **RevIN:** Kim, T., Kim, J., Tae, Y., Park, C., Choi, J. H., & Choo, J. (2022). *Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift*. ICLR 2022.
3. **BATADAL:** Taormina, R., et al. (2018). *Battle of the Attack Detection Algorithms: Disclosing Cyber Attacks on Water Distribution Networks*. Journal of Water Resources Planning and Management.
