# Student Instructions

This project is about multivariate time series forecasting with deep learning.

## Important Links

- Dataset: https://huggingface.co/datasets/AIML-TUDA/dlam-ts-project-data-2026
- Hugging Face leaderboard Space: https://aiml-tuda-dlam-ts-project-leaderboard-2026.hf.space/
- Moodle submission page: https://moodle.informatik.tu-darmstadt.de/course/view.php?id=2011
- Report template: `student/report_template/`
- Code template: `student/submission_template/`
- Baseline examples: `student/baseline/`
- Deadline: 04.09.2026

Leaderboard metrics: MAE, MSE, RMSE, MAPE, sMAPE, and WAPE. Lower is better for all metrics.

Target: predict the future hourly operational load index for each `series_id`. Higher values mean more operational pressure in that unit.

## What You Need To Submit

Submit the following:

- Final report PDF.
- Reproducible code repository or archive.
- Final model archive `final_submission.zip` through the Hugging Face leaderboard Space.

## Public Validation Leaderboard

You may upload validation predictions to the Hugging Face Space. The Space scores your CSV automatically against hidden validation labels and updates the public validation leaderboard.

Before uploading, register your group in the Space and list every group member's Hugging Face username. Any listed member can submit revisions for the same group.

Before registration, submission tabs are hidden. After registration, the Space uses your Hugging Face login to select the group automatically, and the group tab changes to `Manage Group`. You do not type the group ID again. If the registration is wrong, ask an instructor to correct it.

Each validation upload also asks for a model name. Use a short technical name such as `XY_features_v1` or `ZZZ_ablation_2`. Reusing the same model name creates a new revision for that model; using a different model name creates another leaderboard row for your group. Do not add the model name as a CSV column.

The student dataset contains `train.csv`, `validation_input.csv`, `forecast_index_validation.csv`, and `metadata.json`. It does not contain validation targets, test inputs, or test targets.

Before building your model, you can generate simple baseline prediction files from `student/baseline/`.

Required prediction format:

```csv
series_id,timestamp,prediction
```

Your validation prediction file must contain one row for every row in `forecast_index_validation.csv`. The 24-hour forecast horizon is the recommended rollout block length, not the total number of rows to submit. The validation and private test indices each contain 336 hourly timestamps per series, so a 24-step model must be rolled forward until all required rows are filled.

## Final Model Submission

Final test data is private. You do not receive test labels or private test inputs. Instead, you submit a runnable model archive. During private evaluation, your script receives a private input directory containing `test_input.csv`, `forecast_index_test.csv`, and `metadata.json`.

Your `final_submission.zip` must contain:

```text
predict.py
requirements.txt
checkpoint.pt
src/  # optional
```

During private evaluation, instructors run:

```bash
python predict.py --input_dir /data/input --output_file /output/predictions.csv --checkpoint /submission/checkpoint.pt
```

Your script must write the prediction CSV to the requested output path.

When uploading the final archive, use the same model name as the validation row that corresponds to the submitted checkpoint.

The baseline examples in `student/baseline/` show simple ways to produce valid prediction CSVs. Your final model should replace these baselines with your own PyTorch forecasting method.

## Reproducibility Requirements

- Use PyTorch for the model.
- Document training and inference steps in your README.
- Fix random seeds where reasonable.
- State all important hyperparameters.
- Include each group member's contribution in the report.
- Do not depend on internet access during final inference.

## Report

The report should be 4--6 pages excluding references and cover introduction, related work, method, experiments, and conclusion.
