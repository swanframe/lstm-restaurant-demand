# scripts/run_prediction.py
from __future__ import annotations

import argparse
import os

from src.demand_forecast.utils.io import load_yaml, ensure_dirs
from src.demand_forecast.prediction.predict import (
    load_context,
    predict_next_day,
    save_predictions,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate next-day predictions with trained LSTM model.")
    parser.add_argument("--config", required=True, help="Path to configs/params.yaml")
    parser.add_argument("--paths", required=True, help="Path to configs/paths.yaml")
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model name used during training (e.g., lstm_20251012-090000). If omitted, picks the first available best_model_*.keras.",
    )
    parser.add_argument(
        "--ohe-frame",
        default=None,
        help="Optional path to a pickled OHE-complete DataFrame. "
             "Defaults to data/processed/df_features_ohe.pkl if not provided.",
    )
    args = parser.parse_args()

    # Load configs
    cfg = load_yaml(args.config)
    pth = load_yaml(args.paths)["paths"]

    artifacts_dir = pth["artifacts"]
    figures_dir = pth.get("figures", "figures")
    ensure_dirs(artifacts_dir, figures_dir)

    # Resolve OHE frame path
    ohe_frame_path = args.ohe_frame or os.path.join(artifacts_dir, "df_features_ohe.pkl")
    if not os.path.exists(ohe_frame_path):
        raise FileNotFoundError(
            f"OHE features frame not found at {ohe_frame_path}. "
            "Run preprocessing first or provide --ohe-frame."
        )

    # Load model + artifacts context
    ctx = load_context(
        artifacts_dir=artifacts_dir,
        model_name=args.model_name,
        sequence_cfg=cfg["sequence"],
    )

    # Predict next day
    pred_df = predict_next_day(
        artifacts_dir=artifacts_dir,
        model_path=ctx.model_path,
        df_features_ohe_path=ohe_frame_path,
        final_feature_columns=ctx.final_feature_columns,
        target_columns=ctx.target_columns,
        scaler_X=ctx.scaler_X,
        scaler_Y=ctx.scaler_Y,
        numeric_features_scaled_by_X=ctx.numeric_features_scaled_by_X,
        sequence_length=ctx.sequence_length,
    )

    # Save
    out_csv = save_predictions(pred_df, artifacts_dir, args.model_name)
    print("Saved predictions:", out_csv)