import pandas as pd
import numpy as np

print("=== LOADING DATASETS ===")
train = pd.read_csv("data/benchmark/train.csv")
val_in = pd.read_csv("data/benchmark/validation_input.csv")
val_idx = pd.read_csv("data/benchmark/forecast_index_validation.csv")

print(f"Train shape: {train.shape}")
print(f"Validation input shape: {val_in.shape}")
print(f"Forecast index validation shape: {val_idx.shape}")

print("\n=== SERIES ID COUNTS ===")
print(f"Train unique series: {train['series_id'].nunique()}")
print(f"Val input unique series: {val_in['series_id'].nunique()}")
print(f"Val index unique series: {val_idx['series_id'].nunique()}")

print("\n=== TIMESTAMPS ===")
print(f"Train time range: {train['timestamp'].min()} to {train['timestamp'].max()}")
print(f"Val input time range: {val_in['timestamp'].min()} to {val_in['timestamp'].max()}")
print(f"Val index time range: {val_idx['timestamp'].min()} to {val_idx['timestamp'].max()}")

steps_per_series_train = train.groupby("series_id").size()
print(f"Train steps per series (min/max/mean): {steps_per_series_train.min()} / {steps_per_series_train.max()} / {steps_per_series_train.mean():.1f}")

steps_per_series_val_in = val_in.groupby("series_id").size()
print(f"Val in steps per series (min/max/mean): {steps_per_series_val_in.min()} / {steps_per_series_val_in.max()} / {steps_per_series_val_in.mean():.1f}")

steps_per_series_val_idx = val_idx.groupby("series_id").size()
print(f"Val idx steps per series (min/max/mean): {steps_per_series_val_idx.min()} / {steps_per_series_val_idx.max()} / {steps_per_series_val_idx.mean():.1f}")

print("\n=== TARGET STATISTICS ===")
print(train["target"].describe())

print("\n=== MISSING VALUES IN TRAIN ===")
missing_train = train.isnull().sum()
print(missing_train[missing_train > 0])

print("\n=== MISSING VALUES IN VAL INPUT ===")
missing_val = val_in.isnull().sum()
print(missing_val[missing_val > 0])

print("\n=== COLUMNS COMPARISON ===")
print(f"Columns in train but not val_in: {set(train.columns) - set(val_in.columns)}")
print(f"Columns in val_in but not train: {set(val_in.columns) - set(train.columns)}")

print("\n=== VALIDATION INPUT CONTINUITY CHECK ===")
# Check how validation_input relates to forecast_index_validation
print(f"Total rows in forecast_index_validation: {len(val_idx)} (96 series * 336 steps = {96 * 336})")
print(f"Total rows in validation_input: {len(val_in)} (96 series * 672 steps = {96 * 672})")

# Let's check how many steps before and during forecast window exist in validation_input
val_in['dt'] = pd.to_datetime(val_in['timestamp'])
val_idx['dt'] = pd.to_datetime(val_idx['timestamp'])
print(f"Val in time span: {val_in['dt'].min()} to {val_in['dt'].max()}")
print(f"Val idx time span: {val_idx['dt'].min()} to {val_idx['dt'].max()}")

overlap = set(val_in['dt']).intersection(set(val_idx['dt']))
print(f"Overlap between validation_input timestamps and forecast_index timestamps: {len(overlap)} timestamps")

# Check if validation_input contains future covariate rows during the forecast period
val_in_past = val_in[val_in['dt'] < val_idx['dt'].min()]
val_in_future = val_in[val_in['dt'] >= val_idx['dt'].min()]
print(f"Rows in validation_input before forecast start: {len(val_in_past)} ({len(val_in_past)/96:.0f} steps per series)")
print(f"Rows in validation_input during forecast horizon: {len(val_in_future)} ({len(val_in_future)/96:.0f} steps per series)")
