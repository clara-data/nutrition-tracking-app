"""
Nutrition computation engine.

Given a matched fdc_id and a quantity (number of portions), computes the
total nutrients by scaling USDA per-100g values by the food's gram weight.
"""
import logging
from datetime import date

import pandas as pd

from backend.models import DailyTotals, FoodNutrition, NutrientValue
import backend.data_loader as data_loader
import backend.matcher as matcher
from backend.config import USER_INPUT_FILE, NUTRIENT_FOODS



logger = logging.getLogger(__name__)

NUTRIENT_UNITS = {
    "calories":     "kcal",
    "protein":      "g",
    "fat":          "g",
    "carbohydrate": "g",
    "fiber":        "g",
    "sodium":       "mg",
}

NUTRIENT_KEYS = list(NUTRIENT_UNITS.keys())


def read_user_input() -> pd.DataFrame:
    """
    Read user_input.csv.

    Expected columns: food_name, quantity
    Returns a DataFrame with those two columns; bad rows are skipped.
    """
    try:
        df = pd.read_csv(USER_INPUT_FILE, on_bad_lines="skip")
    except FileNotFoundError:
        raise FileNotFoundError(f"user_input.csv not found at {USER_INPUT_FILE}")

    required = {"food_name", "quantity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"user_input.csv is missing columns: {missing}")

    df["food_name"] = df["food_name"].astype(str).str.strip()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df = df.dropna(subset=["food_name", "quantity"])
    df = df[df["quantity"] > 0]
    return df.reset_index(drop=True)


def compute_food_nutrition(food_name: str, quantity: float) -> FoodNutrition:
    """
    Match food_name, retrieve nutrients per 100g, scale by portion × quantity.

    Returns FoodNutrition with warning=None on success, or a warning message
    when some nutrient data is missing.
    """
    food_match = matcher.find_best_match(food_name)

    if not food_match.matched:
        # Return a zero-nutrient record so callers can collect unrecognized items
        return FoodNutrition(
            food_name=food_name,
            usda_match=food_match.usda_description,
            fdc_id=food_match.fdc_id,
            quantity=quantity,
            gram_weight=0.0,
            calories=0.0,
            protein=0.0,
            fat=0.0,
            carbohydrate=0.0,
            fiber=0.0,
            sodium=0.0,
            warning=f"No confident match found (best score {food_match.score:.0f}/100)",
        )

    gram_weight = data_loader.get_default_portion_grams(food_match.fdc_id)
    nutrients = data_loader.get_nutrients(food_match.fdc_id)

    # Scale from per-100g to actual intake: quantity × gram_weight × (value/100)
    scale = quantity * gram_weight / 100.0

    missing = [k for k, v in nutrients.items() if v == 0.0]
    warning = f"Missing nutrient data for: {', '.join(missing)}" if missing else None

    return FoodNutrition(
        food_name=food_name,
        usda_match=food_match.usda_description,
        fdc_id=food_match.fdc_id,
        quantity=quantity,
        gram_weight=gram_weight,
        calories=round(nutrients.get("calories", 0.0) * scale, 1),
        protein=round(nutrients.get("protein", 0.0) * scale, 1),
        fat=round(nutrients.get("fat", 0.0) * scale, 1),
        carbohydrate=round(nutrients.get("carbohydrate", 0.0) * scale, 1),
        fiber=round(nutrients.get("fiber", 0.0) * scale, 1),
        sodium=round(nutrients.get("sodium", 0.0) * scale, 1),
        warning=warning,
    )


def compute_daily_totals(items: list[FoodNutrition], day: date) -> DailyTotals:
    """Sum all FoodNutrition items into a DailyTotals object."""
    def _total(key: str) -> float:
        return round(sum(getattr(item, key) for item in items if item.warning is None
                         or "Missing" in (item.warning or "")), 1)

    return DailyTotals(
        date=day,
        calories=NutrientValue(_total("calories"), "kcal"),
        protein=NutrientValue(_total("protein"), "g"),
        fat=NutrientValue(_total("fat"), "g"),
        carbohydrate=NutrientValue(_total("carbohydrate"), "g"),
        fiber=NutrientValue(_total("fiber"), "g"),
        sodium=NutrientValue(_total("sodium"), "mg"),
    )


def process_user_input() -> tuple[list[FoodNutrition], list[str], DailyTotals]:
    """
    Full pipeline: read user_input.csv → match → compute → totals.

    Returns:
        items         — FoodNutrition per food line
        unrecognized  — food names that failed matching
        daily_totals  — summed DailyTotals for today
    """
    df = read_user_input()
    today = date.today()
    items: list[FoodNutrition] = []
    unrecognized: list[str] = []

    for _, row in df.iterrows():
        nutrition = compute_food_nutrition(row["food_name"], float(row["quantity"]))
        items.append(nutrition)
        if nutrition.warning and "No confident match" in nutrition.warning:
            unrecognized.append(row["food_name"])
            logger.warning("Unrecognized food: %s", row["food_name"])
        else:
            logger.info(
                "Matched '%s' → '%s' (%.0f%%) | %.1f kcal",
                row["food_name"], nutrition.usda_match,
                matcher.find_best_match(row["food_name"]).score,
                nutrition.calories,
            )

    recognized = [i for i in items if i.food_name not in unrecognized]
    daily_totals = compute_daily_totals(recognized, today)
    return items, unrecognized, daily_totals
