# src/demand_forecast/prediction/predict.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model


@dataclass
class PredictionContext:
    artifacts_dir: str
    model_path: str
    final_feature_columns: List[str]
    target_columns: List[str]
    scaler_X: object | None
    scaler_Y: object
    numeric_features_scaled_by_X: List[str]
    sequence_length: int
    prediction_horizon: int


def load_context(
    artifacts_dir: str,
    model_name: str | None,
    sequence_cfg: Dict[str, int],
) -> PredictionContext:
    """
    Load model + artifacts required for prediction.
    """
    # Determine model path
    if model_name:
        candidates = [
            os.path.join(artifacts_dir, f"best_model_{model_name}.keras"),
            os.path.join(artifacts_dir, f"{model_name}.keras"),
        ]
    else:
        # Best-effort discovery: prefer a 'best_model_*.keras' if present
        candidates = [
            os.path.join(artifacts_dir, f)
            for f in sorted(os.listdir(artifacts_dir))
            if f.startswith("best_model_") and f.endswith(".keras")
        ]
        # Fallback to any .keras
        if not candidates:
            candidates = [
                os.path.join(artifacts_dir, f)
                for f in sorted(os.listdir(artifacts_dir))
                if f.endswith(".keras")
            ]

    if not candidates:
        raise FileNotFoundError("No trained .keras model found in artifacts directory.")
    model_path = candidates[0] if isinstance(candidates[0], str) else candidates[0]

    # Load artifacts
    final_feature_columns: List[str] = joblib.load(os.path.join(artifacts_dir, "final_feature_columns.pkl"))
    target_columns: List[str] = joblib.load(os.path.join(artifacts_dir, "target_columns.pkl"))
    scaler_Y = joblib.load(os.path.join(artifacts_dir, "final_scaler.pkl"))

    # Optional scaler_X + its numeric column list
    scaler_X_path = os.path.join(artifacts_dir, "scaler_X.pkl")
    num_cols_path = os.path.join(artifacts_dir, "numeric_features_scaled_by_X.pkl")
    scaler_X = joblib.load(scaler_X_path) if os.path.exists(scaler_X_path) else None
    numeric_features_scaled_by_X: List[str] = joblib.load(num_cols_path) if os.path.exists(num_cols_path) else []

    return PredictionContext(
        artifacts_dir=artifacts_dir,
        model_path=model_path,
        final_feature_columns=final_feature_columns,
        target_columns=target_columns,
        scaler_X=scaler_X,
        scaler_Y=scaler_Y,
        numeric_features_scaled_by_X=numeric_features_scaled_by_X,
        sequence_length=int(sequence_cfg["sequence_length"]),
        prediction_horizon=int(sequence_cfg["prediction_horizon"]),
    )


def _ensure_feature_columns(df: pd.DataFrame, required_cols: List[str]) -> pd.DataFrame:
    """
    Ensure all required feature columns exist; add missing ones as zeros.
    Drop unknown extras to keep column order deterministic.
    """
    df = df.copy()
    for c in required_cols:
        if c not in df.columns:
            df[c] = 0.0
    # Keep only required (and Date if present)
    keep = ["Date"] + required_cols if "Date" in df.columns else required_cols
    df = df[keep]
    return df


def _apply_scaler_X_if_any(
    df: pd.DataFrame, scaler_X, numeric_cols: List[str]
) -> pd.DataFrame:
    """
    Apply previously-fitted feature scaler_X to the known numeric feature columns.
    Other columns are left as-is.
    """
    if scaler_X is None or not numeric_cols:
        return df
    df = df.copy()
    present = [c for c in numeric_cols if c in df.columns]
    if present:
        df[present] = scaler_X.transform(df[present])
    return df


def build_recent_sequence_from_ohe_df(
    df_features_ohe_path: str,
    final_feature_columns: List[str],
    scaler_X,
    numeric_features_scaled_by_X: List[str],
    sequence_length: int,
) -> Tuple[np.ndarray, str]:
    """
    Build a single input sequence X (shape: [1, seq_len, n_features]) from the last `seq_len` rows
    of the OHE-complete (UNSCALED) DataFrame `df_features_ohe.pkl` and return the inferred
    prediction date (last Date + 1 day, ISO string).
    """
    # Load OHE-complete frame; this is unscaled per Phase 3 design
    if df_features_ohe_path.endswith(".pkl"):
        df = joblib.load(df_features_ohe_path)
    else:
        df = pd.read_pickle(df_features_ohe_path)  # fallback

    df = df.copy()
    if "Date" not in df.columns:
        raise ValueError("Expected 'Date' column in OHE features frame.")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Ensure all features exist (create missing OHE columns with zeros if needed)
    df = _ensure_feature_columns(df, final_feature_columns)

    # Scale numeric features with scaler_X (if it was fitted during preprocessing)
    df_scaled = _apply_scaler_X_if_any(df, scaler_X, numeric_features_scaled_by_X)

    # Slice last seq_len rows
    if len(df_scaled) < sequence_length:
        raise ValueError(f"Not enough rows to build a sequence of length {sequence_length}.")
    window = df_scaled.iloc[-sequence_length:].copy()

    # Build X sequence
    X = window[final_feature_columns].to_numpy(dtype=np.float32)
    X = np.expand_dims(X, axis=0)  # [1, seq_len, n_features]

    # Next prediction date is last date + 1 day
    last_date = df["Date"].iloc[-1]
    prediction_date_iso = (last_date + pd.Timedelta(days=1)).date().isoformat()

    return X, prediction_date_iso


def predict_next_day(
    artifacts_dir: str,
    model_path: str,
    df_features_ohe_path: str,
    final_feature_columns: List[str],
    target_columns: List[str],
    scaler_X,
    scaler_Y,
    numeric_features_scaled_by_X: List[str],
    sequence_length: int,
) -> pd.DataFrame:
    """
    Predict the next day's ingredient demand (real units) using the latest window of features.
    """
    # Prepare input
    X, pred_date = build_recent_sequence_from_ohe_df(
        df_features_ohe_path=df_features_ohe_path,
        final_feature_columns=final_feature_columns,
        scaler_X=scaler_X,
        numeric_features_scaled_by_X=numeric_features_scaled_by_X,
        sequence_length=sequence_length,
    )

    # Load model and predict (scaled space)
    model = load_model(model_path)
    y_pred_scaled = model.predict(X, verbose=0)[0]  # shape: [n_targets]

    # Inverse scale to original units (grams)
    y_pred_real = scaler_Y.inverse_transform(y_pred_scaled.reshape(1, -1))[0]

    # Assemble tidy DataFrame
    out = pd.DataFrame(
        {
            "Date": [pred_date] * len(target_columns),
            "Ingredient": target_columns,
            "Predicted_Required_Grams": y_pred_real,
        }
    )
    return out


def save_predictions(df: pd.DataFrame, artifacts_dir: str, model_name: str | None) -> str:
    """
    Save predictions to CSV in artifacts_dir and return the path.
    """
    fname = f"predictions_next_day_{model_name}.csv" if model_name else "predictions_next_day.csv"
    out_path = os.path.join(artifacts_dir, fname)
    df.to_csv(out_path, index=False)
    return out_path