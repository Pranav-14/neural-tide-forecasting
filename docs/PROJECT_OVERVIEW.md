# Deep Learning Bonus Project — Team Overview & Tracking Doc
**Course:** Deep Learning (SS26) — TU Darmstadt  
**Final Due Date:** 04. September 2026, 23:59 CEST  
**Exposé Status:** **Submitted** (Group 36, see [DLAM_Project_Group_36.pdf](file:///c:/Users/prana/OneDrive/Desktop/SoSe26/DLAM%202026/01_Allgemeines/docs/DLAM_Project_Group_36.pdf))  
**Leaderboard:** https://aiml-tuda-dlam-ts-project-leaderboard-2026.hf.space/  
**Dataset:** https://huggingface.co/datasets/AIML-TUDA/dlam-ts-project-data-2026

---

## What We Are Building

A deep learning model in **PyTorch** that forecasts the hourly `operational load index` across **96 correlated operational units**, **336 hours (14 days)** into the future. We must beat the provided `naive_last_value` baseline on the private test evaluation.

---

## Dataset Summary & Verified EDA

| Property | Benchmark Dataset Value |
|---|---|
| Units / Series | 96 (`series_id`: `unit_000` to `unit_095`) |
| Frequency | Hourly (`frequency: "h"`) |
| Train Range | 2023-01-01 00:00:00 to 2023-06-29 23:00:00 (180 days, 4,320 steps/series) |
| Train Rows | 414,720 rows (25 columns including `target`) |
| Validation Range | 2023-06-30 00:00:00 to 2023-07-13 23:00:00 (14 days, 336 steps/series) |
| Validation Rows | 32,256 rows (24 feature columns; targets are hidden) |
| Rollout Block | 24 steps at a time, rolled forward iteratively to 336 steps |
| Primary Metric | **WAPE** (Weighted Absolute Percentage Error) |
| Other Metrics | MAE, MSE, RMSE, MAPE, sMAPE (lower is better for all) |
| Target Range | `target` $\in [0.164, 53.000]$ (mean: 9.913, std: 5.548) |

### Features Breakdown (24 Input Features)
- **Temporal Encodings (Complete):** `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `is_weekend`, `zone_sin`, `zone_cos`
- **Known-Future Covariates (~4.5% missing):**
  `demand_forecast`, `staffing_forecast`, `queue_pressure_forecast`, `network_pressure_forecast`, `event_load_forecast`, `service_irregularity_risk_forecast`, `throughput_disruption_risk_forecast`, `upstream_quality_forecast`, `unit_reliability_forecast`
- **Operational Metadata (Complete):**
  `trend`, `nominal_capacity`, `workload_intensity`, `promotion_intensity`, `shock_risk` (~4.5% missing), `maintenance_known`
- **Target:** `target` (Operational load index)

---

## Baseline Benchmarks

### 1. Local Validation Backtest (Last 336 Steps of `train.csv`)
Evaluated using the project evaluation suite in [`src/utils.py`](file:///c:/Users/prana/OneDrive/Desktop/SoSe26/DLAM%202026/01_Allgemeines/src/utils.py):

| Baseline Model | WAPE (Primary) | MAE | RMSE | sMAPE | Role / Status |
|---|---|---|---|---|---|
| **`naive_last_value`** | **0.5450** | **5.8214** | **7.5521** | **65.73%** | **Minimum threshold to beat** |
| `lag168_repeat` | 0.4549 | 4.8584 | 6.3610 | 54.43% | Weekly cyclic baseline |
| `lag24_repeat` | 0.4180 | 4.4646 | 5.8853 | 49.50% | Daily cyclic baseline |
| **`seasonal_mean`** | **0.3140** | **3.3541** | **4.5432** | **34.70%** | **Strongest heuristic baseline** |

### 2. Generated Official Validation CSVs
Ready in [`experiments/baseline_predictions/`](file:///c:/Users/prana/OneDrive/Desktop/SoSe26/DLAM%202026/01_Allgemeines/experiments/baseline_predictions):
- `naive_last_value.csv`
- `lag24_repeat.csv`
- `lag168_repeat.csv`
- `seasonal_mean.csv`

---

## Primary Architecture: TiDE (Time-series Dense Encoder) + RevIN

### Why TiDE?
- **MLP-based encoder-decoder:** Pure `torch.nn`, linear complexity in sequence length, 5–10x faster than Transformers.
- **Native Support for Known-Future Covariates:** Maps the 9 future forecast columns directly into the decoder representation.
- **Linear Residual Skip:** Preserves immediate historical scale and trend from the lookback window.

### Architecture Specifications:
```text
Inputs:
  - Historical Target (L = 168)
  - Historical Covariates (L = 168 x D_cov)
  - Future Covariates (H = 24 x D_cov)

  ↓ [RevIN Normalization on Target]
  ↓ [Feature Projection & Flattening]
  ↓ [Dense Encoder MLP with Residual Blocks]
  → Global Context Embedding (e)
  ↓ [Temporal Decoder per-step MLP + Future Covariates]
  ↓ [+ Direct Linear Residual Skip from History]
  ↓ [RevIN Denormalization]
Outputs:
  - 24-step Forecast Horizon
```

### Planned Improvements & Additions:
1. **RevIN (Reversible Instance Normalization):** Normalizes per-instance non-stationarity and heterogeneous unit scales.
2. **Missing Value Imputation:** Forward-fill + column-wise fallback mean + missingness indicator flags.
3. **Loss Function:** Huber Loss / L1 Smooth Loss for robustness against outliers and spikes.
4. **Learning Rate Schedule:** AdamW with Cosine Annealing.

---

## Additional Dataset: BATADAL (Water Distribution Network)

- **Source:** Taormina et al., 2018 (SCADA sensor readings from a water distribution network).
- **Motivation:** Water infrastructure operational load mirrors operational unit pressure; contains **no future covariates**, demonstrating TiDE's generalization across covariate-rich and covariate-free domains.
- **Target Variables:** Multi-target forecasting for tank water levels (meters) and flow rates (LPS).
- **Preprocessing:** Filter labeled cyber-attack periods to retain normal operations; 80/20 train/validation split.

---

## Current Cleaned Project Structure

```text
01_Allgemeines/
├── .gitignore                             # Git ignore rules for caches, checkpoints, CSVs
├── README.md                              # Main navigation & project overview
│
├── docs/                                  # Project documentation & reference briefs
│   ├── bonus_project_deep_learning_SoSe2026.pdf  # Course project brief
│   ├── DLAM_Project_Group_36.pdf                 # Submitted Exposé PDF (Group 36)
│   ├── PROJECT_OVERVIEW.md                       # This tracking document
│   ├── STUDENT_INSTRUCTIONS.md                   # Instructor guidelines
│   └── expose/                                   # Exposé LaTeX source & bibliography
│       ├── expose.tex
│       └── references.bib
│
├── data/                                  # Datasets
│   ├── benchmark/                         # AIML-TUDA benchmark dataset CSVs & metadata
│   └── batadal/                           # BATADAL water network CSVs
│
├── baselines/                             # Baseline implementations & runner
│   ├── baselines.py                       # Naive, lag24, lag168, seasonal mean
│   ├── run_baselines.py
│   ├── requirements.txt
│   └── README.md
│
├── src/                                   # Core PyTorch model & pipeline
│   ├── __init__.py
│   ├── model.py                           # TiDE Architecture & RevIN module
│   ├── dataset.py                         # Sliding-window dataset & covariate handler
│   ├── train.py                           # Training loop, optimizer, scheduler
│   ├── inference.py                       # 24-step iterative rolling predictor
│   └── utils.py                           # Evaluation metrics (WAPE, MAE, RMSE, sMAPE)
│
├── experiments/                           # Experiment runners, local backtests & logs
│   ├── eda.py                             # Exploratory data analysis runner
│   ├── eval_local_baselines.py            # Local validation baseline backtester
│   └── baseline_predictions/              # Generated baseline prediction CSVs
│
├── submission/                            # Template folder for final_submission.zip
│   ├── predict.py                         # Evaluation CLI entrypoint
│   ├── requirements.txt
│   ├── checkpoint.pt                      # Saved model weights
│   └── src/                               # Model definition
│       └── model.py
│
└── report/                                # Final 4–6 Page LaTeX Report
    ├── report.tex                         # LaTeX template
    ├── references.bib                     # Bibliography
    └── figures/                           # Diagrams & experiment charts
```

---

## Progress Tracking & Checklist

### Phase 1: Setup & Exposé (Complete)
- [x] Course requirements analyzed
- [x] Project architecture chosen (TiDE + RevIN)
- [x] Additional cross-domain dataset chosen (BATADAL)
- [x] Exposé written, compiled, and submitted ([DLAM_Project_Group_36.pdf](file:///c:/Users/prana/OneDrive/Desktop/SoSe26/DLAM%202026/01_Allgemeines/docs/DLAM_Project_Group_36.pdf))
- [x] Project directory cleaned and reorganized

### Phase 2: Data Ingestion & Baselines (Complete)
- [x] Benchmark dataset downloaded into `data/benchmark/`
- [x] Exploratory Data Analysis executed ([experiments/eda.py](file:///c:/Users/prana/OneDrive/Desktop/SoSe26/DLAM%202026/01_Allgemeines/experiments/eda.py))
- [x] Metric calculation suite created ([src/utils.py](file:///c:/Users/prana/OneDrive/Desktop/SoSe26/DLAM%202026/01_Allgemeines/src/utils.py))
- [x] Provided baselines executed & prediction CSVs generated
- [x] Local backtest benchmark established on 336-hour validation split

### Phase 3: TiDE + RevIN Model Development (In Progress)
- [ ] Implement `src/dataset.py` (lookback window $L=168$, horizon $H=24$, missing-value imputation)
- [ ] Implement `src/model.py` (RevIN layer + TiDE encoder/decoder)
- [ ] Implement `src/train.py` (training loop, AdamW, Cosine Annealing, checkpointing)
- [ ] Implement `src/inference.py` (24-step iterative rolling rollout to 336 steps)
- [ ] Verify local validation performance vs `seasonal_mean` (Target WAPE < 0.314)

### Phase 4: Leaderboard Submissions & Iteration
- [ ] Format predictions to `series_id,timestamp,prediction`
- [ ] Submit baseline and TiDE predictions to Hugging Face leaderboard Space
- [ ] Hyperparameter tuning (hidden dims, dropout, loss functions)

### Phase 5: BATADAL Cross-Domain Pipeline
- [ ] Download and clean BATADAL SCADA dataset
- [ ] Train covariate-free TiDE on BATADAL multi-target setup
- [ ] Record generalization metrics (MAE/RMSE)

### Phase 6: Ablation Studies & Final Deliverables
- [ ] Ablation 1: TiDE without covariates vs TiDE with covariates
- [ ] Ablation 2: TiDE standard scaling vs TiDE + RevIN
- [ ] Ablation 3: Lookback window length sensitivity ($L=72$ vs $168$ vs $336$)
- [ ] Neural Baseline: LSTM benchmark comparison
- [ ] Write 4–6 page final report in `report/report.tex`
- [ ] Package and verify `final_submission.zip`