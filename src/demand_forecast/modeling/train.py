# src/demand_forecast/modeling/train.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple, List

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

# Reproducibility (best-effort)
os.environ.setdefault("PYTHONHASHSEED", "42")
np.random.seed(42)
try:
    tf.random.set_seed(42)
except Exception:
    pass


@dataclass
class ModelConfig:
    """Configuration for the LSTM model and training loop."""
    lstm_units_layer1: int = 128
    lstm_units_layer2: int = 64
    learning_rate: float = 5e-4
    epochs: int = 200
    batch_size: int = 64
    patience: int = 20
    dropout_rate: float = 0.2
    model_name: str = "lstm_base"


def create_lstm_model(input_shape: Tuple[int, int], output_units: int, cfg: ModelConfig) -> tf.keras.Model:
    """Build a 2-layer LSTM with dropout and a linear Dense head for multi-target regression."""
    model = Sequential(
        [
            LSTM(cfg.lstm_units_layer1, return_sequences=True, input_shape=input_shape),
            Dropout(cfg.dropout_rate),
            LSTM(cfg.lstm_units_layer2),
            Dropout(cfg.dropout_rate),
            Dense(output_units),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=cfg.learning_rate),
        loss=MeanSquaredError(),
        metrics=[MeanAbsoluteError()],
    )
    return model


def _plot_history(history: tf.keras.callbacks.History, out_path: str) -> None:
    """Save training vs validation loss curve."""
    plt.figure(figsize=(10, 5))
    plt.plot(history.history.get("loss", []), label="Train Loss")
    plt.plot(history.history.get("val_loss", []), label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _per_target_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_names: List[str]) -> pd.DataFrame:
    """Compute MAE, RMSE, and MAPE per target on real-scale arrays."""
    eps = 1e-8
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    denom = np.where(np.abs(y_true) < eps, np.nan, np.abs(y_true))
    mape = np.nanmean(np.abs((y_true - y_pred) / denom) * 100.0, axis=0)

    return pd.DataFrame(
        {
            "target": target_names,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE%": mape,
        }
    )


def _attach_prediction_dates(
    df_features_ohe: pd.DataFrame,
    test_start: str,
    test_end: str,
    seq_len: int,
    horizon: int,
) -> List[pd.Timestamp]:
    """Map each test sample index to its prediction date using the raw OHE frame and split range."""
    raw_test = df_features_ohe[
        (df_features_ohe["Date"] >= pd.to_datetime(test_start))
        & (df_features_ohe["Date"] <= pd.to_datetime(test_end))
    ].reset_index(drop=True)

    dates: List[pd.Timestamp] = []
    for i in range(len(raw_test) - seq_len - horizon + 1):
        idx = i + seq_len + horizon - 1
        dates.append(pd.to_datetime(raw_test["Date"].iloc[idx]))
    return dates


def train_and_evaluate(
    artifacts_dir: str,
    figures_dir: str,
    splits_cfg: Dict[str, str],
    seq_cfg: Dict[str, int],
    model_cfg: ModelConfig,
) -> Dict[str, str]:
    """
    Train the LSTM on saved sequences and evaluate on the test set.

    Parameters
    ----------
    artifacts_dir : str
        Directory where preprocessing artifacts (npy, pkl) are stored and where models/metrics will be saved.
    figures_dir : str
        Directory to save plots.
    splits_cfg : Dict[str, str]
        Split boundaries (train/val/test) for mapping prediction dates.
    seq_cfg : Dict[str, int]
        Sequence parameters (sequence_length, prediction_horizon).
    model_cfg : ModelConfig
        Model and training hyperparameters.

    Returns
    -------
    Dict[str, str]
        Paths to key artifacts produced by training and evaluation.
    """
    os.makedirs(figures_dir, exist_ok=True)

    # Load sequences (already scaled)
    X_train = np.load(os.path.join(artifacts_dir, "X_train.npy"))
    y_train = np.load(os.path.join(artifacts_dir, "y_train.npy"))
    X_val = np.load(os.path.join(artifacts_dir, "X_val.npy"))
    y_val = np.load(os.path.join(artifacts_dir, "y_val.npy"))
    X_test = np.load(os.path.join(artifacts_dir, "X_test.npy"))
    y_test = np.load(os.path.join(artifacts_dir, "y_test.npy"))

    # Infer shapes
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_units = y_train.shape[1]

    # Build model
    model = create_lstm_model(input_shape, output_units, model_cfg)

    # Callbacks
    ckpt_path = os.path.join(artifacts_dir, f"best_model_{model_cfg.model_name}.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=model_cfg.patience, restore_best_weights=True, verbose=1),
        ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, verbose=1),
    ]

    # Train
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=model_cfg.epochs,
        batch_size=model_cfg.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Save training curve
    hist_png = os.path.join(figures_dir, f"training_history_{model_cfg.model_name}.png")
    _plot_history(history, hist_png)

    # Evaluate best model
    best_model = load_model(ckpt_path) if os.path.exists(ckpt_path) else model
    test_loss, test_mae = best_model.evaluate(X_test, y_test, verbose=0)

    # Inverse-transform to real scale
    scaler_Y = joblib.load(os.path.join(artifacts_dir, "final_scaler.pkl"))
    target_cols = joblib.load(os.path.join(artifacts_dir, "target_columns.pkl"))
    df_features_ohe: pd.DataFrame = joblib.load(os.path.join(artifacts_dir, "df_features_ohe.pkl"))
    df_features_ohe["Date"] = pd.to_datetime(df_features_ohe["Date"])

    y_pred_scaled = best_model.predict(X_test, verbose=0)
    y_pred_real = scaler_Y.inverse_transform(y_pred_scaled)
    y_true_real = scaler_Y.inverse_transform(y_test)

    # Per-target metrics
    per_target = _per_target_metrics(y_true_real, y_pred_real, target_cols)
    metrics_csv = os.path.join(artifacts_dir, f"metrics_per_target_{model_cfg.model_name}.csv")
    per_target.to_csv(metrics_csv, index=False)

    # Overall metrics
    overall = {
        "Test_MSE": float(test_loss),
        "Test_MAE": float(test_mae),
        "Targets": len(target_cols),
        "Features": input_shape[1],
        "Sequence_Length": seq_cfg["sequence_length"],
    }
    overall_csv = os.path.join(artifacts_dir, f"metrics_overall_{model_cfg.model_name}.csv")
    pd.DataFrame([overall]).to_csv(overall_csv, index=False)

    # Optional test preview with dates
    pred_dates = _attach_prediction_dates(
        df_features_ohe,
        splits_cfg["test_start"],
        splits_cfg["test_end"],
        seq_cfg["sequence_length"],
        seq_cfg["prediction_horizon"],
    )
    n = min(len(pred_dates), y_pred_real.shape[0])
    preview = pd.DataFrame(y_pred_real[:n], columns=[f"pred_{c}" for c in target_cols])
    preview[[f"true_{c}" for c in target_cols]] = pd.DataFrame(
        y_true_real[:n], columns=[f"true_{c}" for c in target_cols]
    )
    preview.insert(0, "Date", [pd.to_datetime(d).date().isoformat() for d in pred_dates[:n]])
    preview_csv = os.path.join(artifacts_dir, f"predictions_test_preview_{model_cfg.model_name}.csv")
    preview.to_csv(preview_csv, index=False)

    # Save final model copy
    final_model_path = os.path.join(artifacts_dir, f"{model_cfg.model_name}.keras")
    best_model.save(final_model_path)

    print("\n=== TRAINING SUMMARY ===")
    print(f"Test MSE: {overall['Test_MSE']:.6f}")
    print(f"Test MAE: {overall['Test_MAE']:.6f}")
    print(f"Targets: {overall['Targets']} | Features: {overall['Features']} | SeqLen: {overall['Sequence_Length']}")
    print("Saved:")
    print("-", ckpt_path)
    print("-", final_model_path)
    print("-", hist_png)
    print("-", metrics_csv)
    print("-", overall_csv)
    print("-", preview_csv)

    return {
        "best_model": ckpt_path,
        "final_model": final_model_path,
        "history_plot": hist_png,
        "metrics_per_target": metrics_csv,
        "metrics_overall": overall_csv,
        "predictions_preview": preview_csv,
    }