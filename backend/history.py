"""
Persist and query user intake history (append-only CSV).

history.csv columns:
    date, food_name, usda_match, fdc_id, quantity,
    calories, protein, fat, carbohydrate, fiber, sodium
"""
import logging
from datetime import date, timedelta

import pandas as pd

from backend.config import HISTORY_FILE, HISTORY_COLUMNS
from backend.models import (
    DailyTotals, FoodNutrition, HistoryComparison, NutrientTrend, NutrientValue,
)

logger = logging.getLogger(__name__)

NUTRIENT_KEYS = ["calories", "protein", "fat", "carbohydrate", "fiber", "sodium"]
NUTRIENT_UNITS = {
    "calories": "kcal", "protein": "g", "fat": "g",
    "carbohydrate": "g", "fiber": "g", "sodium": "mg",
}


# ---------------------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------------------

def load_history() -> pd.DataFrame:
    """Load history.csv; return empty DataFrame with correct columns if absent."""
    if HISTORY_FILE.exists():
        try:
            df = pd.read_csv(HISTORY_FILE, parse_dates=["date"], on_bad_lines="skip")
            return df
        except Exception as exc:
            logger.warning("Could not read history.csv (%s) — starting fresh.", exc)
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def append_to_history(items: list[FoodNutrition], today: date) -> None:
    """Append today's food items to history.csv."""
    rows = []
    for item in items:
        if item.warning and "No confident match" in item.warning:
            continue  # skip unrecognized items
        rows.append({
            "date": today.isoformat(),
            "food_name": item.food_name,
            "usda_match": item.usda_match,
            "fdc_id": item.fdc_id,
            "quantity": item.quantity,
            "calories": item.calories,
            "protein": item.protein,
            "fat": item.fat,
            "carbohydrate": item.carbohydrate,
            "fiber": item.fiber,
            "sodium": item.sodium,
        })

    if not rows:
        logger.warning("No recognized items to append to history.")
        return

    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    df.to_csv(
        HISTORY_FILE,
        mode="a",
        header=not HISTORY_FILE.exists(),
        index=False,
    )
    logger.info("Appended %d rows to history.csv", len(rows))


# ---------------------------------------------------------------------------
# 7-day comparison
# ---------------------------------------------------------------------------

def get_7day_comparison(today: date) -> HistoryComparison | None:
    """
    Build a HistoryComparison covering the 7 days *before* today.

    Returns None if there is no prior history.
    """
    history = load_history()
    if history.empty:
        return None

    history["date"] = pd.to_datetime(history["date"]).dt.date
    cutoff = today - timedelta(days=7)
    past = history[(history["date"] >= cutoff) & (history["date"] < today)]

    if past.empty:
        return None

    # Daily aggregates for the past 7 days
    daily = (
        past.groupby("date")[NUTRIENT_KEYS]
        .sum()
        .reset_index()
        .sort_values("date")
    )

    trends: dict[str, NutrientTrend] = {}
    for key in NUTRIENT_KEYS:
        values = daily[key].tolist() if key in daily.columns else []
        trends[key] = NutrientTrend(
            values=[round(v, 1) for v in values],
            trend=_compute_trend(values),
            delta_vs_avg=_delta_vs_avg(values),
        )

    return HistoryComparison(**trends)


def get_today_totals_from_history(today: date) -> DailyTotals | None:
    """
    Read today's already-logged rows from history and return summed DailyTotals.
    Useful to check if we'd be double-counting.
    """
    history = load_history()
    if history.empty:
        return None

    history["date"] = pd.to_datetime(history["date"]).dt.date
    today_rows = history[history["date"] == today]
    if today_rows.empty:
        return None

    sums = today_rows[NUTRIENT_KEYS].sum()
    return DailyTotals(
        date=today,
        calories=NutrientValue(round(sums["calories"], 1), "kcal"),
        protein=NutrientValue(round(sums["protein"], 1), "g"),
        fat=NutrientValue(round(sums["fat"], 1), "g"),
        carbohydrate=NutrientValue(round(sums["carbohydrate"], 1), "g"),
        fiber=NutrientValue(round(sums["fiber"], 1), "g"),
        sodium=NutrientValue(round(sums["sodium"], 1), "mg"),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_trend(values: list[float]) -> str:
    """Simple linear trend over a list of daily values."""
    if len(values) < 2:
        return "stable"
    first_half = sum(values[: len(values) // 2]) / max(len(values) // 2, 1)
    second_half = sum(values[len(values) // 2 :]) / max(len(values) - len(values) // 2, 1)
    diff = second_half - first_half
    threshold = first_half * 0.05  # 5% relative change
    if diff > threshold:
        return "increasing"
    if diff < -threshold:
        return "decreasing"
    return "stable"


def _delta_vs_avg(values: list[float]) -> float:
    """Latest value minus average of all values (0 if fewer than 2 values)."""
    if len(values) < 2:
        return 0.0
    avg = sum(values) / len(values)
    return round(values[-1] - avg, 1)
