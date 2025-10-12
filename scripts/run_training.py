from __future__ import annotations
import argparse
import os
from datetime import datetime
from src.demand_forecast.utils.io import load_yaml, ensure_dirs
from src.demand_forecast.modeling.train import ModelConfig, train_and_evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM model")
    parser.add_argument("--config", required=True, help="Path to params.yaml")
    parser.add_argument("--paths", required=True, help="Path to paths.yaml")
    parser.add_argument("--name", default=None, help="Optional model name suffix")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    pth = load_yaml(args.paths)["paths"]

    ensure_dirs(pth["artifacts"], "figures")

    # Derive a model name suffix for clarity
    suffix = args.name or datetime.now().strftime("%Y%m%d-%H%M%S")

    model_cfg = ModelConfig(
        lstm_units_layer1=128,
        lstm_units_layer2=64,
        learning_rate=5e-4,
        epochs=200,
        batch_size=64,
        patience=20,
        dropout_rate=0.2,
        model_name=f"lstm_{suffix}",
    )

    outputs = train_and_evaluate(
        artifacts_dir=pth["artifacts"],
        figures_dir="figures",
        splits_cfg=cfg["splits"],
        seq_cfg=cfg["sequence"],
        model_cfg=model_cfg,
    )

    # Friendly summary
    print("Artifacts:")
    for k, v in outputs.items():
        print(f"- {k}: {v}")