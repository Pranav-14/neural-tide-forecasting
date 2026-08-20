"""Exploratory Data Analysis for the BATADAL Water Network SCADA dataset."""

import pandas as pd
import numpy as np

def analyze_batadal():
    print("=== LOADING BATADAL DATASETS ===")
    df1 = pd.read_csv("data/batadal/training_dataset_1.csv")
    df2 = pd.read_csv("data/batadal/training_dataset_2.csv")
    df_test = pd.read_csv("data/batadal/test_dataset.csv")

    print(f"Training dataset 1 shape: {df1.shape}")
    print(f"Training dataset 2 shape: {df2.shape}")
    print(f"Test dataset shape:       {df_test.shape}")

    # Strip column whitespace if any
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()
    df_test.columns = df_test.columns.str.strip()

    print("\n=== COLUMNS IN DATASET 1 ===")
    print(list(df1.columns))

    print("\n=== SAMPLE ROWS (HEAD) ===")
    print(df1.head(3))

    # Identify sensor types
    tank_cols = [c for c in df1.columns if c.startswith("L_T")]
    flow_cols = [c for c in df1.columns if c.startswith("F_")]
    pressure_cols = [c for c in df1.columns if c.startswith("P_")]
    status_cols = [c for c in df1.columns if c.startswith("S_")]

    print("\n=== SENSOR BREAKDOWN ===")
    print(f"Tank Level Sensors ({len(tank_cols)}): {tank_cols}")
    print(f"Flow Rate Sensors ({len(flow_cols)}):   {flow_cols}")
    print(f"Pressure Sensors ({len(pressure_cols)}):    {pressure_cols}")
    print(f"Pump/Valve Status ({len(status_cols)}):   {status_cols}")

    # Attack Flag distribution
    print("\n=== ATTACK FLAGS ===")
    if "ATT_FLAG" in df1.columns:
        print(f"Dataset 1 ATT_FLAG counts:\n{df1['ATT_FLAG'].value_counts().to_dict()}")
    else:
        print("Dataset 1 has no ATT_FLAG (all normal operation).")

    if "ATT_FLAG" in df2.columns:
        print(f"Dataset 2 ATT_FLAG counts:\n{df2['ATT_FLAG'].value_counts().to_dict()}")

    if "ATT_FLAG" in df_test.columns:
        print(f"Test Dataset ATT_FLAG counts:\n{df_test['ATT_FLAG'].value_counts().to_dict()}")

    # Missing values
    print("\n=== MISSING VALUES IN DATASET 1 ===")
    missing = df1.isna().sum()
    print(f"Total missing values: {missing.sum()}")

    # Check timestamps
    print("\n=== TIMESTAMPS ===")
    time_col = [c for c in df1.columns if "time" in c.lower() or "date" in c.lower()][0]
    print(f"Time column: {time_col}")
    print(f"Dataset 1 time span: {df1[time_col].iloc[0]} to {df1[time_col].iloc[-1]}")

if __name__ == "__main__":
    analyze_batadal()
