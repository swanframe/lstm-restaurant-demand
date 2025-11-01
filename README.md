# LSTM Restaurant Raw Material Demand Prediction

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow 2.16.2](https://img.shields.io/badge/TensorFlow-2.16.2-orange)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![MLOps](https://img.shields.io/badge/MLOps-Production%20Ready-green)

Forecast **next-day ingredient demand (grams)** for restaurants using a two-layer LSTM. The repository includes a complete pipeline: data generation (synthetic), preprocessing, training, and prediction—driven by simple CLI commands and YAML configuration. Now with full Docker support for reproducible MLOps!

---

## Table of Contents

1. [Overview](#overview)
2. [Why Synthetic Data?](#why-synthetic-data)
3. [Quick Start](#quick-start)
   * [Clone the repository](#clone-the-repository)
   * [Install dependencies](#install-dependencies)
   * [Configure paths and parameters](#configure-paths-and-parameters)
4. [Docker Deployment](#docker-deployment)
5. [End-to-End Usage](#end-to-end-usage)
   * [Phase 2: Generate synthetic data](#phase-2-generate-synthetic-data)
   * [Phase 3: Preprocess](#phase-3-preprocess)
   * [Phase 4: Train](#phase-4-train)
   * [Phase 5: Predict next day](#phase-5-predict-next-day)
6. [Using Your Own Data](#using-your-own-data)
   * [CSV Schemas](#csv-schemas)
   * [Configuration Alignment](#configuration-alignment)
   * [Replacing the synthetic data](#replacing-the-synthetic-data)
7. [Interpreting Outputs](#interpreting-outputs)
8. [Project Structure](#project-structure)
9. [Model Performance](#model-performance)
10. [Troubleshooting & FAQ](#troubleshooting--faq)
11. [Contact](#contact)
12. [License](#license)

---

## Overview

This project implements a time-series pipeline for restaurant raw material planning, based on peer-reviewed research published in **SINTA (Science and Technology Index)**, *Indonesia's national journal indexing system*.
It includes:

* **Inputs:** recipe usage per menu (grams), daily menu sales (wide format), and a seasonal/calendar file.
* **Preprocessing:** merge, time-based interpolation, lag features (t-1, t-7), One-Hot Encoding for categorical factors, chronological train/val/test split, MinMax scaling, and sequence building.
* **Model:** LSTM with 2 layers (128 and 64 units) + dropout, MSE loss, MAE metric, EarlyStopping and model checkpointing (TensorFlow 2.16.2).
* **Outputs:** trained model files, metrics, plots, and next-day ingredient demand predictions in grams.
* **MLOps:** Full Docker containerization for reproducible experimentation and deployment.

This implementation mirrors the technical architecture and preprocessing logic from the published research while using synthetic data for safe public release.

---

## Why Synthetic Data?

* **Confidentiality:** Original research used proprietary restaurant data that cannot be shared. Synthetic data protects sensitive information.
* **Accessibility:** Anyone can run the full pipeline immediately without requesting data access.
* **Customization:** Synthetic data matches the expected schemas, so it’s straightforward to swap in your restaurant’s data.
* **Reproducibility:** Consistent results across different environments using Docker.

---

## Quick Start

### Clone the repository

```bash
git clone https://github.com/swanframe/lstm-restaurant-demand.git
cd lstm-restaurant-demand
```

### Install dependencies

Python 3.10+ is recommended.

**Option A — standard install**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Option B — editable install (package layout)**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

> TensorFlow 2.16.2 is required. For GPU support, ensure your CUDA/cuDNN stack matches TensorFlow’s compatibility.

### Configure paths and parameters

* Open `configs/paths.yaml` to confirm data/artifact folders.
* Open `configs/params.yaml` to review sequence length, date splits, categorical/binary features, and (later) your target ingredient list.

---

## Docker Deployment

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/) (recommended)

### Quick Start with Docker Compose (Recommended)

Run the entire pipeline with one command:

```bash
docker-compose up ml-pipeline
```

This will:
1. Build the Docker image (if not exists)
2. Generate synthetic data
3. Preprocess features
4. Train the LSTM model
5. Generate next-day predictions

### Manual Docker Execution

**Build the image:**
```bash
docker build -t restaurant-lstm .
```

**Run individual steps:**
```bash
# Generate synthetic data
docker run -v $(pwd)/data:/app/data restaurant-lstm make data-synth

# Preprocess
docker run -v $(pwd)/data:/app/data restaurant-lstm make preprocess

# Train model
docker run -v $(pwd)/data:/app/data -v $(pwd)/figures:/app/figures restaurant-lstm make train

# Generate predictions
docker run -v $(pwd)/data:/app/data restaurant-lstm make predict
```

### Development with Docker

**Interactive shell:**
```bash
docker-compose run ml-shell
# Inside container, run:
# make data-synth
# make preprocess
# make train
# make predict
```

**Run specific Python scripts:**
```bash
docker run -v $(pwd)/data:/app/data restaurant-lstm python scripts/run_training.py --config configs/params.yaml --paths configs/paths.yaml
```

### Docker Benefits

- ✅ **Reproducible environments** across different machines
- ✅ **No dependency conflicts**
- ✅ **Easy deployment** to cloud platforms (AWS, GCP, Azure)
- ✅ **Consistent results** from development to production

---

## End-to-End Usage

### Phase 2: Generate synthetic data

Creates three CSVs in `data/raw/` so you can test the full pipeline.

```bash
make data-synth
```

### Phase 3: Preprocess

Builds features, encoders/scalers, splits, and sequences; saves artifacts to `data/processed/`.

```bash
make preprocess
```

### Phase 4: Train

Trains the LSTM with EarlyStopping and saves the best model and metrics. Plots are written to `figures/`.

```bash
make train
```

### Phase 5: Predict next day

Uses the latest feature window to forecast **tomorrow’s** ingredient demand (grams).

```bash
make predict
```

The prediction CSV is saved under `data/processed/`.

---

## Using Your Own Data

Replace the synthetic CSVs with your data while keeping the same schemas. Then rerun **preprocess → train → predict**.

### CSV Schemas

**1) Recipe usage per menu** — `data/raw/ingredient_need_per_menu_in_grams.csv`

```csv
Menu,Ingredient,Required_Grams
Classic Burger,Beef Sirloin,120
Grilled Chicken Salad,Diced Chicken,110
...
```

* `Menu`: string; must match sales file
* `Ingredient`: string; must match `target_ingredients`
* `Required_Grams`: float; grams per sold portion

**2) Daily menu sales (wide)** — `data/raw/menu_sales_YYYYMMDD-YYYYMMDD.csv`

```csv
Menu,2023-01-01,2023-01-02,2023-01-03,...
Classic Burger,42,35,50,...
Grilled Chicken Salad,28,21,30,...
...
```

* `Menu`: string; must match recipe usage `Menu`
* One column per date (`YYYY-MM-DD`): integer units sold

**3) Seasonal / calendar factors** — `data/raw/seasonal_calendar_YYYYMMDD-YYYYMMDD.csv`

```csv
Date,Weekday,Public_Holiday,Weekend,Is_Monday,Is_Tuesday,Is_Wednesday,Is_Thursday,Is_Friday,Is_Saturday,Is_Sunday,Restaurant_Open,Special_Event,Seasonal_Factor
2023-01-01,0,1,1,0,0,0,0,0,1,0,0,Holiday,Holiday Season
2023-01-02,1,0,0,1,0,0,0,0,0,0,1,None,Rainy
...
```

* `Date`: ISO date
* Binary flags: `Weekday`, `Public_Holiday`, `Weekend`, weekday dummies, `Restaurant_Open`
* Categoricals: `Special_Event`, `Seasonal_Factor` (OHE is applied during preprocessing)

### Configuration Alignment

Edit `configs/params.yaml`:

* `target_ingredients`: ordered list of ingredients to forecast (defines the model output order).
* `menu_ingredient_filter`: `{Menu: [Ingredient, ...]}` pairs used to filter valid recipe lines.
* `categorical_features`, `binary_features`: ensure they match your calendar file.
* `splits`: training/validation/test date ranges.

### Replacing the synthetic data

1. Place your three CSVs (same filenames or update `files:` in `params.yaml`).
2. Update `params.yaml` as described above.
3. Run:

```bash
make preprocess
make train
make predict
```

---

## Interpreting Outputs

* `data/processed/metrics_overall_*.csv` — test MSE and MAE.
* `data/processed/metrics_per_target_*.csv` — MAE/RMSE/MAPE for each ingredient.
* `data/processed/predictions_test_preview_*.csv` — predicted vs. true values by date for the test window.
* `data/processed/predictions_next_day*.csv` — next-day forecast in grams per ingredient.
* `figures/training_history_*.png` — training/validation loss curves.

---

## Project Structure

```
lstm-restaurant-demand/
├── README.md
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── .env.example
├── Makefile
├── configs/
│   ├── params.yaml
│   └── paths.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── figures/
├── notebooks/
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── run_preprocessing.py
│   ├── run_training.py
│   └── run_prediction.py
└── src/
    └── demand_forecast/
        ├── utils/
        ├── preprocessing/
        ├── modeling/
        └── prediction/
```

---

## Model Performance

### Expected Results
- **Test MAE**: ~0.124 (scaled), ~1200-1500 grams (actual)
- **Training Time**: 10-15 minutes on CPU
- **Sequence Length**: 28 days
- **Prediction Horizon**: 1 day (next-day forecast)

### Key Features
- **Multi-target Forecasting**: Predicts 23+ ingredients simultaneously
- **Temporal Features**: Lag features (t-1, t-7) and seasonal patterns
- **Robust Preprocessing**: Handles missing data, feature scaling, and one-hot encoding
- **Early Stopping**: Prevents overfitting with automatic training optimization

---

## Troubleshooting & FAQ

### Docker Issues

**"exec: 'make': executable file not found in $PATH"**
- Use the updated Dockerfile that includes `make` installation
- Or use direct Python commands: `python scripts/run_training.py...`

**Docker out of disk space**
```bash
docker system prune
docker image prune -a
```

**Container can't write to volumes**
- Ensure directories have proper permissions: `chmod 755 data figures`

### Model Training

**TensorFlow not finding GPU / install errors**
Use CPU first (requirements as pinned). For GPU, match CUDA/cuDNN to TF 2.16.2 per TensorFlow docs.

**My data has extra categorical columns**
Add them to `categorical_features` in `params.yaml` and re-run preprocessing.

**Shape mismatches during training**
Confirm `target_ingredients` aligns with your ingredient names and that your recipe and sales files reference the same `Menu` values.

**Predictions seem off for certain ingredients**
Check that `Required_Grams` per menu is realistic, that sales are non-zero for sufficient periods, and that calendar flags reflect reality. Consider extending features (e.g., promotions) and retraining.

**ImportError: No module named 'src'**
Run:

```bash
pip install -e .
```

This installs the project in development mode so the `src` package is discoverable.

### Performance Optimization

**Training is too slow**
- Reduce `epochs` in `ModelConfig` (default: 200 → try 50)
- Reduce `batch_size` (default: 64 → try 32)
- Reduce `sequence_length` (default: 28 → try 14)

**Out of memory errors**
- Reduce the number of `target_ingredients`
- Use smaller LSTM units (128→64, 64→32)

---

## Contact

Maintainer: **Rahman**

* Email: **[arahmanwahid@outlook.com](mailto:arahmanwahid@outlook.com)**
* GitHub: **@swanframe**

Issues and feature requests are welcome via the repository’s **Issues** tab.

---

## License

**MIT — see [LICENSE](LICENSE).**