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
│   ├── run_batadal.py        # BATADAL cross-domain experiment runner
│   ├── run_ablations.py      # Systematic ablations & LSTM neural baseline
│   ├── backtest_recursive.py # Recursive 336-step benchmark table
│   ├── plot_figures.py       # Publication figure generation
│   └── package_submission.py # Archive packaging & verification script
│
├── submission/               # Production inference package & checkpoint interface
│   ├── predict.py            # Standardized CLI entrypoint
│   ├── requirements.txt      # Minimal runtime dependencies
│   └── src/
│
└── tests/                    # Automated test suite
    ├── test_dataset.py
    ├── test_model.py
    ├── test_inference.py
    ├── test_batadal.py
    ├── test_ablations.py
    └── test_submission_package.py
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

### 6. Run Systematic Ablations & LSTM Neural Baseline
Evaluate covariate importance, normalization impact, lookback sensitivity, and the LSTM baseline:
```bash
python experiments/run_ablations.py
```

### 7. Reproduce the Headline Benchmark Table
Score the trained checkpoint and every heuristic baseline on the identical held-out 336-hour window
under the recursive rollout protocol used by the leaderboard:
```bash
python experiments/backtest_recursive.py
```

---

## Empirical Benchmark Results

### 1. Operational Load Benchmark (Covariate-Rich, 96 Series, 336-Hour Horizon)

Local holdout = the last 336 hours of `train.csv`, excluded from training. **Every model below is
scored under the identical recursive protocol**: the model sees true history once, then rolls its own
24-step predictions forward 14 times to cover all 336 hours. Reproduce with
`python experiments/backtest_recursive.py`.

| Model | WAPE (Primary) ↓ | MAE ↓ | RMSE ↓ | sMAPE ↓ | Error Reduction vs Naive |
|---|---|---|---|---|---|
| **`naive_last_value`** | 0.5450 | 5.8214 | 7.5521 | 65.73% | Baseline (0%) |
| **`lag168_repeat`** | 0.4549 | 4.8584 | 6.3610 | 54.43% | +16.5% |
| **`lag24_repeat`** | 0.4180 | 4.4646 | 5.8853 | 49.50% | +23.3% |
| **`seasonal_mean`** | 0.3140 | 3.3541 | 4.5432 | 34.70% | +42.4% |
| **`TiDE + RevIN (Ours)`** | **0.1702** | **1.8174** | **2.8620** | **18.62%** | **+68.8% error reduction** 🔥 |

#### Official Leaderboard (hidden validation labels)

Scored by the course Hugging Face Space against labels we never see. WAPE is reported in percent there.

| Model | WAPE ↓ | MAE ↓ | RMSE ↓ | sMAPE ↓ |
|---|---|---|---|---|
| `naive_last_value` (provided) | 48.098% | 5.2945 | 6.9722 | 52.71% |
| `seasonal_mean` (provided) | 34.409% | 3.7876 | 5.2060 | 37.43% |
| **`tide_revin_val` (Ours)** | **20.416%** | **2.2474** | **3.4777** | **23.91%** |

The hidden validation window is a later, unseen period than the local holdout, so absolute errors are
higher there; the ranking against the baselines is unchanged. Our model reduces naive-baseline error
by **57.6%** on the official split.

### 2. BATADAL Water SCADA Benchmark (Covariate-Free, 19 Multi-Target Sensors)

| Model | WAPE ↓ | MAE ↓ | RMSE ↓ | sMAPE ↓ | Error Reduction vs Naive |
|---|---|---|---|---|---|
| **`naive_last_value`** | 0.4234 | 7.9083 | 21.1123 | 36.29% | Baseline (0%) |
| **`lag24_repeat`** | 0.3978 | 7.4296 | 20.8418 | 31.96% | +6.0% |
| **`lag168_repeat`** | 0.3637 | 6.7933 | 20.0294 | 29.39% | +14.1% |
| **`seasonal_mean`** | 0.3061 | 5.7178 | 13.4787 | 32.32% | +27.7% |
| **`TiDE + RevIN (Covariate-Free)`** | **0.2653** | **4.9497** | **12.7168** | **48.64%** | **+37.3% error reduction** 🔥 |

### 3. Systematic Ablation Studies & Neural Baseline

**Protocol note.** Ablations use a reduced budget (3 epochs, training stride 6) so that all five
variants are trained under identical, affordable conditions, and are scored on direct 24-step windows
tiled across the 336-hour holdout rather than on a recursive rollout. The numbers are therefore
internally comparable to each other but **not** comparable to the recursive figures in Section 1.
Reproduce with `python experiments/run_ablations.py`.

| Configuration | WAPE ↓ | MAE ↓ | RMSE ↓ | sMAPE ↓ | Key Insight |
|---|---|---|---|---|---|
| **`TiDE (No Covariates)`** | 0.2775 | 2.9643 | 4.2115 | 30.45% | Largest single effect: covariates cut error by **38.5%** |
| **`TiDE (Lookback L=72)`** | 0.1839 | 1.9646 | 3.1538 | 20.29% | A 3-day window costs 7.7%; the full weekly cycle matters |
| **`TiDE (No RevIN)`** | 0.1706 | 1.8218 | 3.0084 | 19.36% | **No measurable effect** at this budget (see note below) |
| **`LSTM Neural Baseline`** | 0.1679 | 1.7928 | 2.9601 | 18.88% | Competitive — marginally ahead of TiDE here |
| **`TiDE + RevIN (Full Proposed)`** | 0.1707 | 1.8227 | 3.0129 | 19.06% | Matches No-RevIN; wins on training throughput |

**Honest reading of these results.** Two of our design choices are clearly supported and two are not:

* **Known-future covariates are decisive** (0.2775 → 0.1707) and a **1-week lookback is justified**
  (0.1839 → 0.1707). These are the load-bearing choices.
* **RevIN gives no measurable accuracy gain on this benchmark** (0.1706 without vs 0.1707 with — a
  0.06% difference, well inside run-to-run noise). We retain it because it costs ~0.02% of parameters
  and is the standard remedy for the cross-unit scale heterogeneity present in this data, but we do
  not claim an accuracy benefit we did not measure.
* **The LSTM baseline is not beaten** under this reduced budget (0.1679 vs 0.1707). TiDE's advantage
  here is throughput, not accuracy: it trains ~30% faster per epoch because it has no sequential
  recurrence. Establishing an accuracy win would require training both to convergence and scoring
  both recursively — an experiment we flag as future work rather than assert.

---

## References

1. **TiDE:** Das, A., Kong, W., Leach, A., Mathur, S., Sen, R., & Yu, R. (2023). *Long-term Forecasting with TiDE: Time-series Dense Encoder*. Transactions on Machine Learning Research (TMLR).
2. **RevIN:** Kim, T., Kim, J., Tae, Y., Park, C., Choi, J. H., & Choo, J. (2022). *Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift*. ICLR 2022.
3. **BATADAL:** Taormina, R., et al. (2018). *Battle of the Attack Detection Algorithms: Disclosing Cyber Attacks on Water Distribution Networks*. Journal of Water Resources Planning and Management.
