"""
Loads USDA parquet files into Pandas at startup and exposes lookup functions.

Column mapping (new parquet schema):
  food.parquet          : food_id, food_description, food_category
  food_nutrient.parquet : food_nutrient_id, food_id, nutrient_id, food_nutrient_amount
  nutrient.parquet      : nutrient_id, nutrient_description, nutrient_unit_description, ...
  food_portion.parquet  : optional — food_id, seq_num, gram_weight, ...
                          Falls back to DEFAULT_PORTION_GRAMS if file is absent.
"""
import logging
import pandas as pd
from backend.config import (
    FOOD_FILE, FOOD_NUTRIENT_FILE, NUTRIENT_FILE, FOOD_PORTION_FILE,
    TARGET_NUTRIENTS, DEFAULT_PORTION_GRAMS,
    SR_LEGACY_ID_MIN, SR_LEGACY_ID_MAX,
)


logger = logging.getLogger(__name__)

_food_df: pd.DataFrame = pd.DataFrame()
_pool_df: pd.DataFrame = pd.DataFrame()          # matching pool (all foods)
_nutrient_pivot: pd.DataFrame = pd.DataFrame()   # food_id → {calories, protein, …}
_portion_df: pd.DataFrame = pd.DataFrame()       # may stay empty if file absent


def load() -> None:
    """Load and pre-process all USDA tables. Call once at application startup."""
    global _food_df, _pool_df, _nutrient_pivot, _portion_df

    # --- food.parquet ---
    logger.info("Loading food.parquet …")
    _food_df = pd.read_parquet(FOOD_FILE, columns=["food_id", "food_description", "food_category"])
    _food_df["food_id"] = _food_df["food_id"].astype(int)

        # Matching pool: SR Legacy range only (curated whole-food dataset).
    # Using the full 2M-food parquet causes noisy matches from branded products.
    _pool_df = (
        _food_df[
            _food_df["food_id"].between(SR_LEGACY_ID_MIN, SR_LEGACY_ID_MAX) &
            _food_df["food_description"].notna()
        ]
        .copy()
        .reset_index(drop=True)
    )
    _pool_df["description_lower"] = _pool_df["food_description"].str.lower()
    logger.info("Matching pool: %d SR Legacy foods", len(_pool_df))

    # --- food_nutrient.parquet — filter to 6 target nutrients ---
    logger.info("Loading food_nutrient.parquet (large file) …")
    fn_df = pd.read_parquet(
        FOOD_NUTRIENT_FILE,
        columns=["food_id", "nutrient_id", "food_nutrient_amount"],
    )
    fn_df = fn_df[fn_df["nutrient_id"].isin(TARGET_NUTRIENTS)].copy()
    fn_df["food_id"]    = fn_df["food_id"].astype(int)
    fn_df["nutrient_id"] = fn_df["nutrient_id"].astype(int)

    # Map nutrient_id → column key (e.g. 1008 → "calories")
    id_to_key = {int(k): v for k, v in TARGET_NUTRIENTS.items()}
    fn_df["nutrient_key"] = fn_df["nutrient_id"].map(id_to_key)

    _nutrient_pivot = (
        fn_df.pivot_table(
            index="food_id",
            columns="nutrient_key",
            values="food_nutrient_amount",
            aggfunc="first",
        )
        .reset_index()
    )
    # Ensure every target column exists even if no food has that nutrient
    for col in id_to_key.values():
        if col not in _nutrient_pivot.columns:
            _nutrient_pivot[col] = float("nan")
    logger.info("Nutrient pivot: %d foods × %d nutrients", *_nutrient_pivot.shape)

    # --- food_portion.parquet (optional) ---
    if FOOD_PORTION_FILE.exists():
        logger.info("Loading food_portion.csv …")
        _portion_df = pd.read_csv(
            FOOD_PORTION_FILE,
            usecols=["food_id", "seq_num", "gram_weight"],
            dtype={"food_id": "Int64"},
            on_bad_lines="skip",
        )
        _portion_df = _portion_df.dropna(subset=["food_id"]).copy()
        _portion_df["food_id"] = _portion_df["food_id"].astype(int)
        logger.info("Portion data loaded: %d rows", len(_portion_df))
    else:
        logger.warning(
            "food_portion.parquet not found — defaulting to %.0fg per unit.", DEFAULT_PORTION_GRAMS
        )

    logger.info("USDA data loaded.")


# ---------------------------------------------------------------------------
# Public lookup functions
# ---------------------------------------------------------------------------

def get_pool() -> pd.DataFrame:
    """Return the full matching pool DataFrame."""
    return _pool_df


def get_nutrients(food_id: int) -> dict[str, float]:
    """
    Return nutrient values per 100g for a given food_id.
    Missing nutrients are returned as 0.0.
    """
    row = _nutrient_pivot[_nutrient_pivot["food_id"] == food_id]
    if row.empty:
        return {col: 0.0 for col in TARGET_NUTRIENTS.values()}
    record = row.iloc[0].to_dict()
    record.pop("food_id", None)
    return {k: (0.0 if pd.isna(v) else float(v)) for k, v in record.items()}


def get_default_portion_grams(food_id: int) -> float:
    """
    Return gram weight of the default (first by seq_num) portion.
    Falls back to DEFAULT_PORTION_GRAMS if no portion data exists.
    """
    if _portion_df.empty:
        return DEFAULT_PORTION_GRAMS
    rows = _portion_df[_portion_df["food_id"] == food_id].sort_values("seq_num")
    if rows.empty:
        return DEFAULT_PORTION_GRAMS
    return float(rows.iloc[0]["gram_weight"])


def has_data() -> bool:
    return not _food_df.empty
