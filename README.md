# Neural-TiDE: Multivariate Long-Horizon Time-Series Forecasting

An efficient, pure PyTorch implementation of **TiDE (Time-series Dense Encoder)** enhanced with **RevIN (Reversible Instance Normalization)** for long-horizon multivariate time-series forecasting across covariate-rich and covariate-free domains.

---

## Key Highlights

* **High Computational Efficiency:** MLP-based encoder-decoder design that is $5\text{--}10\times$ faster than standard Transformer architectures while matching or outperforming them on long horizons.
* **Native Future Covariate Support:** Direct encoding of dynamic known-future signals (e.g., workload forecasts, scheduled maintenance, cyclical calendar encodings) into temporal decoder projections.
* **Non-Stationarity & Scale Robustness:** Integrated **RevIN** handles distribution shifts and heterogeneous scales across diverse correlated time-series units.
* **Cross-Domain Generalization:** Seamless operation across both **covariate-rich** (operational load forecasting) and **covariate-free** (BATADAL water distribution SCADA) regimes.
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
│   ├── eda.py                # Benchmark dataset analysis
│   ├── eda_batadal.py        # BATADAL SCADA dataset analysis
│   ├── eval_local_baselines.py # Benchmark baseline evaluation
│   └── run_batadal.py        # BATADAL cross-domain experiment runner
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

### 3. Train TiDE Model
Train the primary TiDE + RevIN model on the benchmark operational dataset:
```bash
python src/train.py --epochs 15 --batch-size 256 --lr 0.001 --loss huber
```

### 4. Generate Predictions (CLI Interface)
Generate 336-step iterative rolling predictions matching the submission specification:
```bash
python submission/predict.py \
    --input_dir data/benchmark \
    --output_file experiments/predictions/tide_revin_val.csv \
    --checkpoint submission/checkpoint.pt
```

### 5. Run BATADAL Cross-Domain Benchmark
Evaluate covariate-free multi-target forecasting on the BATADAL water distribution network:
```bash
python experiments/run_batadal.py
```

---

## Empirical Benchmark Results

### 1. Operational Load Benchmark (Covariate-Rich, 96 Series, 336-Hour Horizon)

| Model | WAPE (Primary) ↓ | MAE ↓ | RMSE ↓ | sMAPE ↓ | Error Reduction vs Naive |
|---|---|---|---|---|---|
| **`naive_last_value`** | 0.5450 | 5.8214 | 7.5521 | 65.73% | Baseline (0%) |
| **`lag168_repeat`** | 0.4549 | 4.8584 | 6.3610 | 54.43% | +16.5% |
| **`lag24_repeat`** | 0.4180 | 4.4646 | 5.8853 | 49.50% | +23.3% |
| **`seasonal_mean`** | 0.3140 | 3.3541 | 4.5432 | 34.70% | +42.4% |
| **`TiDE + RevIN (Ours)`** | **0.1616** | **1.7264** | **2.9265** | **18.23%** | **+70.3% error reduction** 🔥 |

### 2. BATADAL Water SCADA Benchmark (Covariate-Free, 19 Multi-Target Sensors)

| Model | WAPE ↓ | MAE ↓ | RMSE ↓ | sMAPE ↓ | Error Reduction vs Naive |
|---|---|---|---|---|---|
| **`naive_last_value`** | 0.4234 | 7.9083 | 21.1123 | 36.29% | Baseline (0%) |
| **`lag24_repeat`** | 0.3978 | 7.4296 | 20.8418 | 31.96% | +6.0% |
| **`lag168_repeat`** | 0.3637 | 6.7933 | 20.0294 | 29.39% | +14.1% |
| **`seasonal_mean`** | 0.3061 | 5.7178 | 13.4787 | 32.32% | +27.7% |
| **`TiDE + RevIN (Covariate-Free)`** | **0.2653** | **4.9497** | **12.7168** | **48.64%** | **+37.3% error reduction** 🔥 |

---

## References

1. **TiDE:** Das, A., Kong, W., Leach, A., Mathur, S., Sen, R., & Yu, R. (2023). *Long-term Forecasting with TiDE: Time-series Dense Encoder*. Transactions on Machine Learning Research (TMLR).
2. **RevIN:** Kim, T., Kim, J., Tae, Y., Park, C., Choi, J. H., & Choo, J. (2022). *Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift*. ICLR 2022.
3. **BATADAL:** Taormina, R., et al. (2018). *Battle of the Attack Detection Algorithms: Disclosing Cyber Attacks on Water Distribution Networks*. Journal of Water Resources Planning and Management.
