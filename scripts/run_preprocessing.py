from __future__ import annotations
import argparse
import os
import pandas as pd
import joblib
from src.demand_forecast.utils.io import load_yaml, ensure_dirs
from src.demand_forecast.preprocessing.pipeline import run_all_preprocessing

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline")
    parser.add_argument("--config", required=True, help="Path to params.yaml")
    parser.add_argument("--paths", required=True, help="Path to paths.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    pth = load_yaml(args.paths)["paths"]

    # Ensure directories exist
    ensure_dirs(pth["data_interim"], pth["data_processed"]) 

    # Dynamically adjust train_start after 7-day lag removal
    calendar_csv = os.path.join(pth["data_raw"], cfg["files"]["seasonal_calendar"])
    cal_df = pd.read_csv(calendar_csv)
    cal_df["Date"] = pd.to_datetime(cal_df["Date"]) 
    cal_df = cal_df.sort_values("Date").reset_index(drop=True)
    if len(cal_df) >= 8:
        new_train_start = str(cal_df["Date"].iloc[7].date())
        cfg["splits"]["train_start"] = new_train_start
        print(f"Adjusted train_start to account for lag_7: {new_train_start}")

    # Wire file configuration
    file_cfg = {
        "ingredient_need_per_menu": cfg["files"]["ingredient_need_per_menu"],
        "menu_sales_wide": cfg["files"]["menu_sales_wide"],
        "seasonal_calendar": cfg["files"]["seasonal_calendar"],
    }

    # Execute pipeline
    X_train, y_train, X_val, y_val, X_test, y_test, scaler_Y = run_all_preprocessing(
        file_cfg=file_cfg,
        split_cfg=cfg["splits"],
        sequence_cfg=cfg["sequence"],
        menu_ingredient_filter=cfg["menu_ingredient_filter"],
        binary_cols=cfg["binary_features"],
        target_cols=cfg["target_ingredients"],
        paths=pth,
    )

    # Save final scaler for model & prediction stages
    joblib.dump(scaler_Y, os.path.join(pth["artifacts"], "final_scaler.pkl"))
    print("Saved final target scaler as final_scaler.pkl")