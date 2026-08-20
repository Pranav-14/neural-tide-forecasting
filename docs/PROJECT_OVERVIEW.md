# Deep Learning Bonus Project — Team Overview & Tracking Doc
**Course:** Deep Learning (SS26) — TU Darmstadt  
**Final Due Date:** 04. September 2026, 23:59 CEST  
**Exposé Status:** **Submitted** (Group 36, see [DLAM_Project_Group_36.pdf](file:///c:/Users/prana/OneDrive/Desktop/SoSe26/DLAM%202026/01_Allgemeines/docs/DLAM_Project_Group_36.pdf))  
**GitHub Repository:** https://github.com/Pranav-14/neural-tide-forecasting  
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

---

## Git Workflow & Issue Tracking

All development is tracked via GitHub Issues and Pull Requests on `main`:

```text
Feature Branch (e.g. feature/dataset-pipeline)
   ↓ (develop & test)
Commit with issue reference (e.g. "feat: implement sliding window dataset (#1)")
   ↓
Push feature branch to origin
   ↓
Open Pull Request linking to Issue #X
   ↓
Review, verify, and merge into main
```

### GitHub Issues Roadmap:

| Issue | Title | Status | Linked Branch / PR |
|---|---|---|---|
| [#1](https://github.com/Pranav-14/neural-tide-forecasting/issues/1) | `feat: Implement sliding-window PyTorch dataset pipeline and covariate imputation` | ✅ **Completed** | [PR #9](https://github.com/Pranav-14/neural-tide-forecasting/pull/9) (Merged) |
| [#2](https://github.com/Pranav-14/neural-tide-forecasting/issues/2) | `feat: Implement TiDE architecture with Reversible Instance Normalization (RevIN)` | ✅ **Completed** | [PR #9](https://github.com/Pranav-14/neural-tide-forecasting/pull/9) (Merged) |
| [#3](https://github.com/Pranav-14/neural-tide-forecasting/issues/3) | `feat: Build training pipeline with learning rate schedulers and checkpointing` | 🔄 **In Progress** | `feature/training-and-inference` |
| [#4](https://github.com/Pranav-14/neural-tide-forecasting/issues/4) | `feat: Build 24-step iterative rolling inference engine and submission CLI` | 🔄 **In Progress** | `feature/training-and-inference` |
| [#5](https://github.com/Pranav-14/neural-tide-forecasting/issues/5) | `experiment: Validation leaderboard submission and benchmark verification` | 📋 To Do | `feature/leaderboard-validation` |
| [#6](https://github.com/Pranav-14/neural-tide-forecasting/issues/6) | `feat: BATADAL water distribution SCADA cross-domain pipeline and experiments` | 📋 To Do | `feature/batadal-pipeline` |
| [#7](https://github.com/Pranav-14/neural-tide-forecasting/issues/7) | `experiment: Systematic ablation studies and neural baseline (LSTM)` | 📋 To Do | `feature/ablations` |
| [#8](https://github.com/Pranav-14/neural-tide-forecasting/issues/8) | `docs: Author final research report (4-6 pages LaTeX) and package submission` | 📋 To Do | `feature/final-report` |

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
- [ ] Implement `src/dataset.py` (Issue #1)
- [ ] Implement `src/model.py` (Issue #2)
- [ ] Implement `src/train.py` (Issue #3)
- [ ] Implement `src/inference.py` (Issue #4)
- [ ] Verify local validation performance vs `seasonal_mean` (Target WAPE < 0.314)

### Phase 4: Leaderboard Submissions & Iteration (Issue #5)
- [ ] Format predictions to `series_id,timestamp,prediction`
- [ ] Submit baseline and TiDE predictions to Hugging Face leaderboard Space
- [ ] Hyperparameter tuning (hidden dims, dropout, loss functions)

### Phase 5: BATADAL Cross-Domain Pipeline (Issue #6)
- [ ] Download and clean BATADAL SCADA dataset
- [ ] Train covariate-free TiDE on BATADAL multi-target setup
- [ ] Record generalization metrics (MAE/RMSE)

### Phase 6: Ablation Studies & Final Deliverables (Issue #7 & #8)
- [ ] Ablation 1: TiDE without covariates vs TiDE with covariates
- [ ] Ablation 2: TiDE standard scaling vs TiDE + RevIN
- [ ] Ablation 3: Lookback window length sensitivity ($L=72$ vs $168$ vs $336$)
- [ ] Neural Baseline: LSTM benchmark comparison
- [ ] Write 4–6 page final report in `report/report.tex`
- [ ] Package and verify `final_submission.zip`