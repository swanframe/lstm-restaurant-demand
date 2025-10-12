from __future__ import annotations
import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import joblib

# Globals persisted across steps
onehot_encoder: OneHotEncoder | None = None
final_feature_columns_after_ohe: List[str] | None = None

# -----------------------------
# 1) Cleaning & aggregation
# -----------------------------

def clean_data(
    need_per_menu_csv: str,
    menu_sales_wide_csv: str,
    menu_ingredient_filter: Dict[str, List[str]],
) -> pd.DataFrame:
    """Aggregate daily total ingredient demand in grams.

    Inputs
    ------
    - need_per_menu_csv: CSV with columns [Menu, Ingredient, Required_Grams]
    - menu_sales_wide_csv: CSV with columns [Menu, <YYYY-MM-DD> ...] (wide daily sales)
    - menu_ingredient_filter: {menu: [ingredient, ...]} pairs to keep

    Output
    ------
    DataFrame with columns [Date, Ingredient, Required_Grams]
    """
    print("Step 1: Cleaning & aggregation ...")

    df_need = pd.read_csv(need_per_menu_csv)
    df_sales = pd.read_csv(menu_sales_wide_csv)

    # Build valid (menu, ingredient) pairs
    valid_pairs = {(m, ing) for m, ings in menu_ingredient_filter.items() for ing in ings}

    # Filter recipe rows
    filtered_need = df_need[
        df_need.apply(lambda r: (r["Menu"], r["Ingredient"]) in valid_pairs, axis=1)
    ]

    # Melt sales wide → long
    df_sales_long = df_sales.melt(
        id_vars="Menu", var_name="Date", value_name="Quantity"
    )
    df_sales_long["Date"] = pd.to_datetime(df_sales_long["Date"])  # ensure datetime

    # Join recipe usage with daily sales
    merged = filtered_need.merge(df_sales_long, on="Menu", how="inner")
    merged["Required_Grams"] = merged["Required_Grams"] * merged["Quantity"]

    # Aggregate per date × ingredient
    result = (
        merged.groupby(["Date", "Ingredient"], as_index=False)["Required_Grams"].sum()
        .sort_values(["Date", "Ingredient"]).reset_index(drop=True)
    )

    print("Cleaning done. Sample:")
    print(result.head())
    return result


# -----------------------------
# 2) Feature engineering (merge calendar, interpolate, pivot, lags)
# -----------------------------

def feature_engineer(
    daily_ingredient_need: pd.DataFrame,
    seasonal_calendar_csv: str,
) -> pd.DataFrame:
    """Add calendar features and build lag features, keeping time order.

    Returns a wide table with ingredients as columns + calendar features.
    """
    print("Step 2: Feature engineering ...")

    calendar = pd.read_csv(seasonal_calendar_csv)
    daily_ingredient_need["Date"] = pd.to_datetime(daily_ingredient_need["Date"]) 
    calendar["Date"] = pd.to_datetime(calendar["Date"]) 

    # Merge with calendar
    df = daily_ingredient_need.merge(calendar, on="Date", how="left")
    df = df.sort_values(["Date", "Ingredient"]).reset_index(drop=True)

    # Treat 0 demand on open days as missing for interpolation
    mask = (df["Restaurant_Open"] == 1) & (df["Required_Grams"] == 0)
    df.loc[mask, "Required_Grams"] = np.nan

    # Time interpolation per ingredient
    groups = []
    for ing, g in df.groupby("Ingredient"):
        g = g.set_index("Date")
        g["Required_Grams"] = g["Required_Grams"].interpolate(method="time")
        groups.append(g.reset_index())
    df = pd.concat(groups, ignore_index=True)

    # Pivot ingredient columns wide
    pivot_df = df.pivot_table(
        index="Date", columns="Ingredient", values="Required_Grams", aggfunc="first"
    ).reset_index()

    # Merge back calendar
    final_df = pivot_df.merge(calendar, on="Date", how="left")

    # Add lags (1 and 7) for each ingredient column
    ingredient_cols = [c for c in pivot_df.columns if c != "Date"]
    for ing in ingredient_cols:
        final_df[f"{ing}_lag_1"] = final_df[ing].shift(1)
        final_df[f"{ing}_lag_7"] = final_df[ing].shift(7)

    # Drop initial rows with NaNs due to lag_7
    lag7_cols = [f"{ing}_lag_7" for ing in ingredient_cols]
    before = len(final_df)
    final_df = final_df.dropna(subset=lag7_cols).reset_index(drop=True)
    print(f"Dropped {before - len(final_df)} rows due to lag_7 NaNs.")

    print(
        f"Feature engineering done. Rows: {len(final_df)}, Cols: {len(final_df.columns)}"
    )
    print(final_df.head())
    return final_df


# -----------------------------
# 2.5) One-Hot Encoding for categorical features
# -----------------------------

def apply_one_hot_encoding(
    df: pd.DataFrame,
    categorical_cols: List[str],
    target_cols: List[str],
    artifacts_dir: str,
) -> pd.DataFrame:
    print("Step 2.5: One-Hot Encoding categorical features ...")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]) 
    df = df.sort_values("Date").reset_index(drop=True)

    # Fill missing categories with a neutral placeholder
    placeholder = "None"
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(placeholder)

    global onehot_encoder
    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = onehot_encoder.fit_transform(df[categorical_cols])
    encoded_names = onehot_encoder.get_feature_names_out(categorical_cols)

    df_encoded = pd.DataFrame(encoded, columns=encoded_names, index=df.index)

    # Drop original categorical columns and join encoded
    df_ohe = pd.concat([df.drop(columns=categorical_cols), df_encoded], axis=1)

    # Optionally drop placeholder columns to avoid leakage of a no-op category
    placeholder_cols = [c for c in encoded_names if placeholder in c]
    df_ohe = df_ohe.drop(columns=placeholder_cols, errors="ignore")

    # Persist encoder and final feature columns (non-target, non-Date)
    joblib.dump(onehot_encoder, os.path.join(artifacts_dir, "onehot_encoder.pkl"))

    global final_feature_columns_after_ohe
    final_feature_columns_after_ohe = [
        c for c in df_ohe.columns if c != "Date" and c not in target_cols
    ]
    joblib.dump(
        final_feature_columns_after_ohe,
        os.path.join(artifacts_dir, "final_feature_columns.pkl"),
    )
    joblib.dump(target_cols, os.path.join(artifacts_dir, "target_columns.pkl"))

    print(
        f"OHE done. Added {len(encoded_names) - len(placeholder_cols)} columns. "
        f"Final feature count: {len(final_feature_columns_after_ohe)}"
    )
    return df_ohe


# -----------------------------
# 3) Chronological split
# -----------------------------

def split_data(df: pd.DataFrame, split_cfg: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Step 3: Chronological split ...")
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]) 
    df = df.sort_values("Date").reset_index(drop=True)

    # Convert split dates to datetime for proper comparison
    train_start = pd.to_datetime(split_cfg["train_start"])
    train_end = pd.to_datetime(split_cfg["train_end"])
    val_start = pd.to_datetime(split_cfg["val_start"])
    val_end = pd.to_datetime(split_cfg["val_end"])
    test_start = pd.to_datetime(split_cfg["test_start"])
    test_end = pd.to_datetime(split_cfg["test_end"])

    train = df[(df["Date"] >= train_start) & (df["Date"] <= train_end)].copy()
    val = df[(df["Date"] >= val_start) & (df["Date"] <= val_end)].copy()
    test = df[(df["Date"] >= test_start) & (df["Date"] <= test_end)].copy()

    total = len(df)
    n_train, n_val, n_test = len(train), len(val), len(test)
    print("=" * 60)
    print("TIME-ORDERED SPLIT SUMMARY".center(60))
    print("=" * 60)
    print(f"Total rows: {total}")
    print(f"- Train: {n_train} rows ({n_train/total*100:.2f}%) | {split_cfg['train_start']} → {split_cfg['train_end']}")
    print(f"- Val  : {n_val} rows ({n_val/total*100:.2f}%) | {split_cfg['val_start']} → {split_cfg['val_end']}")
    print(f"- Test : {n_test} rows ({n_test/total*100:.2f}%) | {split_cfg['test_start']} → {split_cfg['test_end']}")
    print("Actual ranges:")
    print(f"- Train: {train['Date'].min().date()} → {train['Date'].max().date()}")
    print(f"- Val  : {val['Date'].min().date()} → {val['Date'].max().date()}")
    print(f"- Test : {test['Date'].min().date()} → {test['Date'].max().date()}")
    print("=" * 60)

    return train, val, test


# -----------------------------
# 4) Scaling (features and targets)
# -----------------------------

def normalize_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    binary_cols: List[str],
    target_cols: List[str],
    encoder: OneHotEncoder,
    artifacts_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    print("Step 4: Normalization ...")

    # OHE columns to exclude from feature scaling
    ohe_cols = encoder.get_feature_names_out(encoder.feature_names_in_).tolist()
    excluded = set(binary_cols + ohe_cols + target_cols)

    # Numeric feature columns for scaler_X (excluding Date, excluded set)
    num_cols = [
        c for c in train_df.select_dtypes(include=np.number).columns
        if c != "Date" and c not in excluded
    ]

    print(f"Feature columns scaled by scaler_X ({len(num_cols)}):")
    print(num_cols)
    print(f"Target columns scaled by scaler_Y ({len(target_cols)}):")
    print(target_cols)

    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()

    if num_cols:
        train_df[num_cols] = scaler_X.fit_transform(train_df[num_cols])
        val_df[num_cols] = scaler_X.transform(val_df[num_cols])
        test_df[num_cols] = scaler_X.transform(test_df[num_cols])
        joblib.dump(num_cols, os.path.join(artifacts_dir, "numeric_features_scaled_by_X.pkl"))
    else:
        print("No numeric feature columns to scale by scaler_X.")

    train_df[target_cols] = scaler_Y.fit_transform(train_df[target_cols])
    val_df[target_cols] = scaler_Y.transform(val_df[target_cols])
    test_df[target_cols] = scaler_Y.transform(test_df[target_cols])

    joblib.dump(scaler_X, os.path.join(artifacts_dir, "scaler_X.pkl"))
    joblib.dump(scaler_Y, os.path.join(artifacts_dir, "scaler_Y.pkl"))
    print("Scaling completed. Saved scaler_X.pkl and scaler_Y.pkl.")

    return train_df, val_df, test_df, scaler_Y


# -----------------------------
# 5) Sequence building for LSTM
# -----------------------------

def create_sequences_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sequence_length: int,
    prediction_horizon: int,
    target_cols: List[str],
    final_feature_columns: List[str],
    artifacts_dir: str,
):
    print("Step 5: Building sequences ...")

    feature_cols = final_feature_columns

    def _mk_seq(data: pd.DataFrame):
        X, y = [], []
        data = data.sort_values("Date").reset_index(drop=True)
        for i in range(len(data) - sequence_length - prediction_horizon + 1):
            seq = data[feature_cols].iloc[i : i + sequence_length].values
            tgt = data[target_cols].iloc[i + sequence_length + prediction_horizon - 1].values
            X.append(seq)
            y.append(tgt)
        return np.array(X), np.array(y)

    X_train, y_train = _mk_seq(train_df)
    X_val, y_val = _mk_seq(val_df)
    X_test, y_test = _mk_seq(test_df)

    np.save(os.path.join(artifacts_dir, "X_train.npy"), X_train)
    np.save(os.path.join(artifacts_dir, "y_train.npy"), y_train)
    np.save(os.path.join(artifacts_dir, "X_val.npy"), X_val)
    np.save(os.path.join(artifacts_dir, "y_val.npy"), y_val)
    np.save(os.path.join(artifacts_dir, "X_test.npy"), X_test)
    np.save(os.path.join(artifacts_dir, "y_test.npy"), y_test)

    print("Saved sequences:")
    print(f"- Train: X {X_train.shape}, y {y_train.shape}")
    print(f"- Val  : X {X_val.shape}, y {y_val.shape}")
    print(f"- Test : X {X_test.shape}, y {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


# -----------------------------
# Orchestrator
# -----------------------------

def run_all_preprocessing(
    file_cfg: Dict[str, str],
    split_cfg: Dict[str, str],
    sequence_cfg: Dict[str, int],
    menu_ingredient_filter: Dict[str, List[str]],
    binary_cols: List[str],
    target_cols: List[str],
    paths: Dict[str, str],
):
    """Run the entire preprocessing pipeline and persist artifacts."""
    print("Starting preprocessing pipeline ...")

    artifacts_dir = paths["data_processed"]
    ensure_dir = lambda p: os.makedirs(p, exist_ok=True)
    ensure_dir(artifacts_dir)

    # 1) Cleaning
    daily_need = clean_data(
        paths["data_raw"] + "/" + file_cfg["ingredient_need_per_menu"],
        paths["data_raw"] + "/" + file_cfg["menu_sales_wide"],
        menu_ingredient_filter,
    )

    # 2) Feature engineering
    df_features = feature_engineer(
        daily_need,
        paths["data_raw"] + "/" + file_cfg["seasonal_calendar"],
    )

    # 2.5) OHE
    df_features_ohe = apply_one_hot_encoding(
        df_features.copy(),
        categorical_cols=["Special_Event", "Seasonal_Factor"],
        target_cols=target_cols,
        artifacts_dir=artifacts_dir,
    )

    # Persist OHE-complete DataFrame for analysis
    joblib.dump(df_features_ohe, os.path.join(artifacts_dir, "df_features_ohe.pkl"))

    # 3) Split
    train_df, val_df, test_df = split_data(df_features_ohe, split_cfg)

    # 4) Normalize
    train_scaled, val_scaled, test_scaled, scaler_Y = normalize_data(
        train_df.copy(),
        val_df.copy(),
        test_df.copy(),
        binary_cols,
        target_cols,
        onehot_encoder,  # type: ignore
        artifacts_dir,
    )

    # 5) Sequences
    X_train, y_train, X_val, y_val, X_test, y_test = create_sequences_data(
        train_scaled.copy(),
        val_scaled.copy(),
        test_scaled.copy(),
        sequence_cfg["sequence_length"],
        sequence_cfg["prediction_horizon"],
        target_cols,
        final_feature_columns_after_ohe,  # type: ignore
        artifacts_dir,
    )

    print("==============================================")
    print("PREPROCESSING COMPLETE")
    print("Artifacts saved under:", artifacts_dir)
    print("==============================================")

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler_Y