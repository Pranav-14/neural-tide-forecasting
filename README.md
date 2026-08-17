# Deep Learning Bonus Project (SS26) — TU Darmstadt
**Multivariate Time Series Forecasting**

## Project Structure
```text
01_Allgemeines/
├── docs/                                  # Project documentation & reference briefs
│   ├── bonus_project_deep_learning_SoSe2026.pdf  # Course project brief
│   ├── DLAM_Project_Group_36.pdf                 # Submitted Exposé PDF (Group 36)
│   ├── PROJECT_OVERVIEW.md                       # Comprehensive guide & architecture plan
│   ├── STUDENT_INSTRUCTIONS.md                   # Instructor guidelines
│   └── expose/                                   # Exposé LaTeX source & references
│       ├── expose.tex
│       └── references.bib
│
├── data/                                  # Datasets (local / gitignored)
│   ├── benchmark/                         # AIML-TUDA benchmark dataset CSVs
│   └── batadal/                           # BATADAL water network SCADA dataset
│
├── baselines/                             # Baseline implementations & runner
│   ├── baselines.py                       # Naive, lag24, lag168, seasonal mean
│   ├── run_baselines.py
│   ├── requirements.txt
│   └── README.md
│
├── src/                                   # Core PyTorch model & pipeline
│   ├── __init__.py
│   ├── model.py                           # TiDE architecture + RevIN layer
│   ├── dataset.py                         # Sliding-window dataset & covariate imputation
│   ├── train.py                           # Training loop, scheduler, loss functions
│   ├── inference.py                       # 24-step iterative rolling predictor
│   └── utils.py                           # Metrics & helper utilities
│
├── experiments/                           # Experiment runners, ablations, & logs
│
├── submission/                            # Packaging directory for final_submission.zip
│   ├── predict.py                         # Evaluation CLI entrypoint
│   ├── requirements.txt                   # Inference dependencies
│   ├── checkpoint.pt                      # Model weights
│   ├── README.md
│   └── src/                               # Model definition
│       └── model.py
│
└── report/                                # Final 4-6 Page LaTeX Report
    ├── report.tex                         # LaTeX template
    ├── references.bib                     # BibTeX bibliography
    ├── README.md
    └── figures/                           # Diagrams & experiment charts
```

## Important Links
- **Dataset:** https://huggingface.co/datasets/AIML-TUDA/dlam-ts-project-data-2026
- **Leaderboard Space:** https://aiml-tuda-dlam-ts-project-leaderboard-2026.hf.space/
- **Moodle:** https://moodle.informatik.tu-darmstadt.de/course/view.php?id=2011
- **Deadline:** 04. September 2026, 23:59 CEST
