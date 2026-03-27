from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).parent.parent
USDA_DATA_DIR = ROOT / "USDA_data"
USER_DATA_DIR = ROOT / "user_data"
USER_INPUT_FILE = USER_DATA_DIR / "user_input.csv"
HISTORY_FILE = USER_DATA_DIR / "history.csv"

# --- USDA parquet file names ---
FOOD_FILE          = USDA_DATA_DIR / "food.parquet"
FOOD_NUTRIENT_FILE = USDA_DATA_DIR / "food_nutrient.parquet"
NUTRIENT_FILE      = USDA_DATA_DIR / "nutrient.parquet"
FOOD_PORTION_FILE  = USDA_DATA_DIR / "food_portion.csv"   # optional — 100g fallback if absent

# --- Default portion weight when food_portion.csv has no entry ---
DEFAULT_PORTION_GRAMS = 100.0

# --- SR Legacy food_id range (curated whole-food dataset, ~7,700 foods) ---
# These IDs cover the USDA SR Legacy dataset — common whole/minimally-processed
# foods with complete nutrient profiles. Used as the primary matching pool
# instead of the full 2M-food parquet (which includes noisy branded products).
SR_LEGACY_ID_MIN = 167512
SR_LEGACY_ID_MAX = 175215

# --- Nutrients to track: {nutrient_id: output_key} ---
TARGET_NUTRIENTS = {
    1008: "calories",      # kcal
    1003: "protein",       # g
    1004: "fat",           # g
    1005: "carbohydrate",  # g
    1079: "fiber",         # g
    1093: "sodium",        # mg
}

# --- Matching thresholds ---
MATCH_THRESHOLD = 25   # minimum TF-IDF cosine score (0–100) to accept a match
MAX_CANDIDATES = 50    # top-N candidates from substring pre-filter passed to TF-IDF re-rank

# --- Dietary Reference Intakes (DRI) — general adult, 2000 kcal diet ---
DRI_TARGETS = {
    "calories":     {"target": 2000, "unit": "kcal", "direction": "target"},
    "protein":      {"target": 50,   "unit": "g",    "direction": "min"},
    "fat":          {"target": 78,   "unit": "g",    "direction": "max"},
    "carbohydrate": {"target": 275,  "unit": "g",    "direction": "target"},
    "fiber":        {"target": 28,   "unit": "g",    "direction": "min"},
    "sodium":       {"target": 2300, "unit": "mg",   "direction": "max"},
}

# --- History CSV columns ---
HISTORY_COLUMNS = [
    "date", "food_name", "usda_match", "fdc_id", "quantity",
    "calories", "protein", "fat", "carbohydrate", "fiber", "sodium",
]

# --- Recommendation foods per deficient nutrient ---
NUTRIENT_FOODS = {
    "protein": [
        {"food": "eggs",           "serving": "2 large",     "amount": 12.0},
        {"food": "chicken breast", "serving": "100g",        "amount": 31.0},
        {"food": "Greek yogurt",   "serving": "1 cup",       "amount": 17.0},
        {"food": "tuna",           "serving": "100g canned", "amount": 25.0},
        {"food": "lentils",        "serving": "1 cup cooked","amount": 18.0},
    ],
    "fiber": [
        {"food": "lentils",     "serving": "1 cup cooked", "amount": 15.6},
        {"food": "chia seeds",  "serving": "2 tbsp",       "amount": 7.8},
        {"food": "avocado",     "serving": "1 medium",     "amount": 6.7},
        {"food": "oat bran",    "serving": "1 cup cooked", "amount": 5.7},
        {"food": "black beans", "serving": "1 cup cooked", "amount": 15.0},
    ],
    "carbohydrate": [
        {"food": "brown rice",   "serving": "1 cup cooked", "amount": 45.0},
        {"food": "oatmeal",      "serving": "1 cup cooked", "amount": 27.0},
        {"food": "sweet potato", "serving": "1 medium",     "amount": 26.0},
        {"food": "banana",       "serving": "1 medium",     "amount": 27.0},
    ],
    "calories": [
        {"food": "avocado",    "serving": "1 medium",  "amount": 240.0},
        {"food": "almonds",    "serving": "1 oz",      "amount": 164.0},
        {"food": "peanut butter","serving":"2 tbsp",   "amount": 190.0},
    ],
    # Sodium: excess is the concern — no suggestion foods
    "fat": [
        {"food": "olive oil",    "serving": "1 tbsp",  "amount": 14.0},
        {"food": "avocado",      "serving": "1 medium","amount": 21.0},
        {"food": "salmon",       "serving": "100g",    "amount": 13.0},
    ],
}
