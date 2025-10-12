from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.demand_forecast.utils.io import load_yaml, ensure_dirs

# -----------------------------
# Synthetic vocab (menus & ingredients) - cuisine-agnostic, English-only
# -----------------------------
MENUS = [
    "Classic Burger",
    "Grilled Chicken Salad",
    "Margherita Pizza",
    "Spaghetti Bolognese",
    "Vegetable Stir Fry",
    "Sushi Platter",
    "Chicken Tacos",
    "Beef Burrito",
    "Tomato Soup",
    "Mushroom Risotto",
    "BBQ Ribs",
    "Seafood Paella",
    "Vegan Buddha Bowl",
    "Pancake Stack",
    "Chocolate Cake",
    "Lemon Tart",
    "Fruit Smoothie",
    "Iced Coffee",
]

INGREDIENTS = [
    "Diced Chicken",
    "Chicken Stock Bones",
    "Leaf Spinach",
    "Lemon",
    "Brown Rice",
    "Vermicelli Noodles",
    "Curry Spice Mix",
    "Potato Sticks",
    "White Fish Fillet",
    "Sweet Corn",
    "Button Mushrooms",
    "Wood Ear Mushrooms",
    "Green Peas",
    "Cassava Root",
    "Raisins",
    "Almonds",
    "Lamb Shoulder",
    "Beef Sirloin",
    "Fresh Wheat Noodles",
    "Olive Oil",
    "Coconut Milk",
    "Roasted Peanuts",
    "Banana",
    "Beef Meatloaf",
    "Tapioca Pearls",
    "Pandan Leaves",
    "Toasted Coconut Flakes",
    "Pastry Cream",
    "Firm Tofu",
    "Edamame",
    "Wheat Flour",
    "Vegetable Broth",
    "Shrimp",
    "Bell Peppers",
    "Tomatoes",
    "Garlic",
]

# Keep exactly 36 output targets to mirror the original architecture
TARGET_INGREDIENTS = INGREDIENTS[:36]

# A deterministic, human-readable mapping from menu to (ingredient, grams per portion)
# This stays generic and can be easily modified by users later.
MENU_INGREDIENT_MAP = {
    "Classic Burger": {
        "Beef Sirloin": 120,
        "Wheat Flour": 20,
        "Tomatoes": 25,
        "Leaf Spinach": 10,
        "Olive Oil": 5,
        "Garlic": 5,
    },
    "Grilled Chicken Salad": {
        "Diced Chicken": 110,
        "Leaf Spinach": 40,
        "Tomatoes": 30,
        "Lemon": 5,
        "Olive Oil": 5,
    },
    "Margherita Pizza": {
        "Wheat Flour": 120,
        "Tomatoes": 80,
        "Olive Oil": 10,
        "Garlic": 5,
    },
    "Spaghetti Bolognese": {
        "Wheat Flour": 90,
        "Beef Sirloin": 100,
        "Tomatoes": 60,
        "Garlic": 5,
    },
    "Vegetable Stir Fry": {
        "Fresh Wheat Noodles": 120,
        "Bell Peppers": 50,
        "Leaf Spinach": 30,
        "Garlic": 5,
        "Olive Oil": 8,
    },
    "Sushi Platter": {
        "White Fish Fillet": 90,
        "Brown Rice": 100,
        "Shrimp": 50,
        "Edamame": 30,
        "Coconut Milk": 5,
    },
    "Chicken Tacos": {
        "Diced Chicken": 90,
        "Fresh Wheat Noodles": 60,  # acts as tortilla proxy
        "Tomatoes": 30,
        "Bell Peppers": 30,
        "Garlic": 5,
    },
    "Beef Burrito": {
        "Beef Sirloin": 110,
        "Brown Rice": 100,
        "Tomatoes": 40,
        "Leaf Spinach": 20,
        "Garlic": 5,
    },
    "Tomato Soup": {
        "Tomatoes": 150,
        "Vegetable Broth": 200,
        "Olive Oil": 5,
        "Garlic": 5,
    },
    "Mushroom Risotto": {
        "Button Mushrooms": 100,
        "Brown Rice": 120,
        "Olive Oil": 8,
        "Garlic": 5,
    },
    "BBQ Ribs": {
        "Lamb Shoulder": 150,
        "Garlic": 5,
        "Olive Oil": 8,
    },
    "Seafood Paella": {
        "Shrimp": 80,
        "White Fish Fillet": 80,
        "Brown Rice": 120,
        "Bell Peppers": 40,
        "Garlic": 5,
    },
    "Vegan Buddha Bowl": {
        "Brown Rice": 100,
        "Edamame": 60,
        "Leaf Spinach": 40,
        "Bell Peppers": 30,
        "Almonds": 10,
    },
    "Pancake Stack": {
        "Wheat Flour": 100,
        "Banana": 50,
        "Pastry Cream": 30,
        "Coconut Milk": 20,
    },
    "Chocolate Cake": {
        "Wheat Flour": 120,
        "Pastry Cream": 50,
        "Toasted Coconut Flakes": 20,
        "Raisins": 10,
        "Almonds": 10,
    },
    "Lemon Tart": {
        "Wheat Flour": 100,
        "Lemon": 20,
        "Pastry Cream": 40,
        "Almonds": 10,
    },
    "Fruit Smoothie": {
        "Banana": 100,
        "Coconut Milk": 150,
        "Raisins": 10,
    },
    "Iced Coffee": {
        "Coconut Milk": 120,
        "Almonds": 5,
    },
}

SPECIAL_EVENTS = ["None", "Promo", "Local Festival", "Sports Final", "Concert Nearby"]
SEASONAL_FACTORS = ["None", "Holiday Season", "Rainy", "Dry", "Back to School"]

# -----------------------------
# Helpers
# -----------------------------

def daterange(start: datetime, end: datetime):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def build_calendar(start: str, end: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)

    rows = []
    for d in daterange(start_dt, end_dt):
        dow = d.weekday()  # Monday=0
        is_weekend = 1 if dow >= 5 else 0
        is_public_holiday = 1 if (d.month == 1 and d.day in (1,)) else 0  # Jan 1 as example
        restaurant_open = 0 if (is_public_holiday and rng.random() < 0.8) else (0 if dow == 0 and rng.random() < 0.05 else 1)

        special_event = rng.choice(SPECIAL_EVENTS, p=[0.8, 0.08, 0.05, 0.04, 0.03])
        seasonal_factor = (
            "Rainy" if d.month in (1,2,3) else
            "Dry" if d.month in (7,8,9) else
            rng.choice(["None", "Holiday Season", "Back to School"], p=[0.7, 0.2, 0.1])
        )

        rows.append({
            "Date": d.date().isoformat(),
            "Weekday": 1 if not is_weekend else 0,
            "Public_Holiday": is_public_holiday,
            "Weekend": is_weekend,
            "Is_Monday": 1 if dow == 0 else 0,
            "Is_Tuesday": 1 if dow == 1 else 0,
            "Is_Wednesday": 1 if dow == 2 else 0,
            "Is_Thursday": 1 if dow == 3 else 0,
            "Is_Friday": 1 if dow == 4 else 0,
            "Is_Saturday": 1 if dow == 5 else 0,
            "Is_Sunday": 1 if dow == 6 else 0,
            "Restaurant_Open": restaurant_open,
            "Special_Event": special_event,
            "Seasonal_Factor": seasonal_factor,
        })
    df = pd.DataFrame(rows)
    return df


def build_menu_sales_wide(start: str, end: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    dates = [d.date().isoformat() for d in daterange(start_dt, end_dt)]

    base_popularity = rng.uniform(0.7, 1.3, size=len(MENUS))

    # Weekly seasonality: weekends boost
    def weekday_multiplier(date_str: str) -> float:
        d = datetime.fromisoformat(date_str)
        if d.weekday() >= 5:
            return 1.25
        return 1.0

    data = {"Menu": MENUS}
    for ds in dates:
        # Baseline 20–80 units per menu, with popularity & weekday effects
        day_sales = (
            rng.integers(20, 81, size=len(MENUS)) * base_popularity * weekday_multiplier(ds)
        )
        # Sporadic shocks for events
        if rng.random() < 0.05:
            shock_idx = rng.integers(0, len(MENUS))
            day_sales[shock_idx] *= rng.uniform(1.5, 2.3)
        data[ds] = np.round(day_sales).astype(int)

    df = pd.DataFrame(data)
    return df


def build_ingredient_need_per_menu() -> pd.DataFrame:
    rows = []
    for menu, comp in MENU_INGREDIENT_MAP.items():
        for ing, grams in comp.items():
            rows.append({
                "Menu": menu,
                "Ingredient": ing,
                "Required_Grams": float(grams),
            })
    return pd.DataFrame(rows).sort_values(["Menu", "Ingredient"]).reset_index(drop=True)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    pth = load_yaml(args.paths)
    seed = cfg.get("random_seed", 42)

    # Paths
    raw_dir = pth["paths"]["data_raw"]
    ensure_dirs(raw_dir)

    # Date ranges from config
    train_start = cfg["splits"]["train_start"]
    train_end = cfg["splits"]["train_end"]
    val_start = cfg["splits"]["val_start"]
    test_end = cfg["splits"]["test_end"]

    overall_start = "2023-01-01"  # fixed to mirror original scope
    overall_end = str(test_end)

    # 1) Seasonal calendar
    calendar_df = build_calendar(overall_start, overall_end, seed)

    # 2) Menu sales (wide)
    sales_df = build_menu_sales_wide(overall_start, overall_end, seed + 1)

    # 3) Ingredient requirement per menu
    need_df = build_ingredient_need_per_menu()

    # Write CSVs
    calendar_path = os.path.join(raw_dir, cfg["files"]["seasonal_calendar"])
    sales_path = os.path.join(raw_dir, cfg["files"]["menu_sales_wide"])
    need_path = os.path.join(raw_dir, cfg["files"]["ingredient_need_per_menu"])

    calendar_df.to_csv(calendar_path, index=False)
    sales_df.to_csv(sales_path, index=False)
    need_df.to_csv(need_path, index=False)

    # Persist a JSON-like sidecar for Phase 3 (optional): menu→ingredient map & targets
    # Users can edit params.yaml instead; we print guidance here.
    print("Synthetic data written to:")
    print("-", calendar_path)
    print("-", sales_path)
    print("-", need_path)
    print("Targets (36) and menu→ingredient mapping are defined inside this script for transparency.")