"""
pipeline.py — Full nutrition pipeline

Reads:
    user_data/user_input.csv   — today's food log (food_name, quantity)
    user_data/history.csv      — past daily entries (auto-created if absent)

Runs:
    1. Daily Intake Summary     — per-food nutrition + daily totals
    2. 7-Day Historical Comparison — trends and delta vs average
    3. Dietary Recommendations  — DRI gap analysis + food suggestions

Writes:
    user_data/output.csv       — all three result sections in one file
    user_data/history.csv      — today's items appended
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from datetime import date

import backend.data_loader as data_loader
import backend.matcher as matcher
import backend.nutrition_engine as engine
import backend.history as hist_module
import backend.recommendations as rec_module
from backend.config import USER_INPUT_FILE, HISTORY_FILE

OUTPUT_FILE = ROOT / "user_data" / "output.csv"


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_section(f, title: str, df: pd.DataFrame) -> None:
    """Write a labelled section block into an open file handle."""
    f.write(f"### {title}\n")
    df.to_csv(f, index=False)
    f.write("\n")


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def build_daily_summary(items, daily_totals) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns two DataFrames:
        items_df  — one row per food item
        totals_df — one row with the day's totals
    """
    item_rows = []
    for item in items:
        item_rows.append({
            "food_name":    item.food_name,
            "usda_match":   item.usda_match,
            "fdc_id":       item.fdc_id,
            "quantity":     item.quantity,
            "gram_weight":  item.gram_weight,
            "calories":     item.calories,
            "protein":      item.protein,
            "fat":          item.fat,
            "carbohydrate": item.carbohydrate,
            "fiber":        item.fiber,
            "sodium":       item.sodium,
            "warning":      item.warning or "",
        })
    items_df = pd.DataFrame(item_rows)

    totals_df = pd.DataFrame([{
        "date":         daily_totals.date.isoformat(),
        "calories":     daily_totals.calories.value,
        "protein":      daily_totals.protein.value,
        "fat":          daily_totals.fat.value,
        "carbohydrate": daily_totals.carbohydrate.value,
        "fiber":        daily_totals.fiber.value,
        "sodium":       daily_totals.sodium.value,
    }])

    return items_df, totals_df


def build_history_df(comparison) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per past day (from the comparison object).
    Columns: date_index, calories, protein, fat, carbohydrate, fiber, sodium,
             trend, delta_vs_avg
    """
    nutrient_keys = ["calories", "protein", "fat", "carbohydrate", "fiber", "sodium"]

    # Determine max number of past days recorded
    n_days = max(len(getattr(comparison, k).values) for k in nutrient_keys)

    rows = []
    for i in range(n_days):
        row = {"day": f"day-{n_days - i} (oldest→recent)"}
        for key in nutrient_keys:
            trend_obj = getattr(comparison, key)
            row[key] = trend_obj.values[i] if i < len(trend_obj.values) else ""
        rows.append(row)

    # Append a trend summary row per nutrient
    trend_rows = []
    for key in nutrient_keys:
        trend_obj = getattr(comparison, key)
        trend_rows.append({
            "day":          f"TREND ({key})",
            key:            trend_obj.trend,
            "delta_vs_avg": trend_obj.delta_vs_avg,
        })

    return pd.DataFrame(rows), pd.DataFrame(trend_rows)


def build_recommendations_df(recs) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per nutrient recommendation.
    Suggestions are flattened into suggestion_1 … suggestion_3 columns.
    """
    rows = []
    for r in recs:
        row = {
            "nutrient": r.nutrient,
            "status":   r.status,
            "current":  r.current,
            "target":   r.target,
            "unit":     r.unit,
            "gap":      r.gap,
        }
        for j, s in enumerate(r.suggestions[:3], start=1):
            row[f"suggestion_{j}"] = f"{s.food} ({s.serving}) +{s.amount}{r.unit}"
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    today = date.today()

    # ------------------------------------------------------------------
    # 1. Load USDA data and build matcher index
    # ------------------------------------------------------------------
    print("Loading USDA data …")
    data_loader.load()
    print("Building TF-IDF index …")
    matcher.build_index()

    # ------------------------------------------------------------------
    # 2. Process user_input.csv
    # ------------------------------------------------------------------
    print(f"\nReading {USER_INPUT_FILE} …")
    items, unrecognized, daily_totals = engine.process_user_input()

    if unrecognized:
        print(f"  ⚠ Unrecognized foods (excluded from totals): {unrecognized}")

    for item in items:
        status = f"{item.calories:.1f} kcal" if not (item.warning and "No confident" in item.warning) else "NO MATCH"
        print(f"  {item.food_name:<20} → {item.usda_match[:40]:<40}  {status}")

    # ------------------------------------------------------------------
    # 3. Append to history.csv
    # ------------------------------------------------------------------
    hist_module.append_to_history(items, today)
    print(f"\nAppended to {HISTORY_FILE}")

    # ------------------------------------------------------------------
    # 4. 7-day comparison
    # ------------------------------------------------------------------
    comparison = hist_module.get_7day_comparison(today)

    # ------------------------------------------------------------------
    # 5. Recommendations
    # ------------------------------------------------------------------
    recs = rec_module.build_recommendations(daily_totals)

    # ------------------------------------------------------------------
    # 6. Build output DataFrames
    # ------------------------------------------------------------------
    items_df, totals_df = build_daily_summary(items, daily_totals)

    if comparison:
        history_df, trend_df = build_history_df(comparison)
    else:
        history_df = pd.DataFrame(columns=["day", "calories", "protein", "fat",
                                            "carbohydrate", "fiber", "sodium"])
        trend_df   = pd.DataFrame()

    recs_df = build_recommendations_df(recs) if recs else pd.DataFrame(
        columns=["nutrient", "status", "current", "target", "unit", "gap"]
    )

    # Unrecognized foods
    unrecognized_df = pd.DataFrame({"unrecognized_food": unrecognized}) if unrecognized \
        else pd.DataFrame({"unrecognized_food": ["none"]})

    # ------------------------------------------------------------------
    # 7. Write output.csv
    # ------------------------------------------------------------------
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="") as f:
        _write_section(f, f"1. DAILY INTAKE — Per Food ({today})",
                       items_df.drop(columns=["fdc_id", "quantity"]))
        _write_section(f, f"2. DAILY INTAKE — Totals ({today})", totals_df)
        _write_section(f, "3. DIETARY RECOMMENDATIONS",            recs_df)

    print(f"\nOutput written to {OUTPUT_FILE}")
    print(f"\n=== Daily Totals ({today}) ===")
    print(f"  Calories:     {daily_totals.calories.value:.1f} kcal")
    print(f"  Protein:      {daily_totals.protein.value:.1f} g")
    print(f"  Fat:          {daily_totals.fat.value:.1f} g")
    print(f"  Carbohydrate: {daily_totals.carbohydrate.value:.1f} g")
    print(f"  Fiber:        {daily_totals.fiber.value:.1f} g")
    print(f"  Sodium:       {daily_totals.sodium.value:.1f} mg")

    if recs:
        print(f"\n=== Recommendations ({len(recs)} nutrients flagged) ===")
        for r in recs:
            print(f"  {r.status.upper():<10} {r.nutrient:<15} "
                  f"{r.current:.1f}/{r.target} {r.unit}  (gap: {r.gap:.1f})")


if __name__ == "__main__":
    run()
