from dataclasses import dataclass, field
from typing import Optional
from datetime import date


@dataclass
class FoodMatch:
    """Result of matching one user food name to a USDA record."""
    input_name: str
    usda_description: str
    fdc_id: int
    score: float          # similarity score 0–100
    matched: bool         # False if below threshold


@dataclass
class NutrientValue:
    value: float
    unit: str


@dataclass
class FoodNutrition:
    """Nutritional content for one food item after quantity scaling."""
    food_name: str
    usda_match: str
    fdc_id: int
    quantity: float
    gram_weight: float        # grams for the given quantity
    calories: float
    protein: float
    fat: float
    carbohydrate: float
    fiber: float
    sodium: float
    warning: Optional[str] = None   # set when nutrients are partially missing


@dataclass
class DailyTotals:
    date: date
    calories: NutrientValue
    protein: NutrientValue
    fat: NutrientValue
    carbohydrate: NutrientValue
    fiber: NutrientValue
    sodium: NutrientValue


@dataclass
class NutrientTrend:
    values: list[float]           # last 7 days oldest→newest
    trend: str                    # "increasing" | "decreasing" | "stable"
    delta_vs_avg: float           # today − 7-day average


@dataclass
class HistoryComparison:
    calories: NutrientTrend
    protein: NutrientTrend
    fat: NutrientTrend
    carbohydrate: NutrientTrend
    fiber: NutrientTrend
    sodium: NutrientTrend


@dataclass
class FoodSuggestion:
    food: str
    serving: str
    amount: float     # nutrient amount per serving


@dataclass
class NutrientRecommendation:
    nutrient: str
    current: float
    target: float
    unit: str
    status: str                          # "deficient" | "excess" | "ok"
    gap: float                           # absolute difference from target
    suggestions: list[FoodSuggestion] = field(default_factory=list)


@dataclass
class ProcessResult:
    """Full response returned after processing user_input.csv."""
    date: date
    items: list[FoodNutrition]
    unrecognized: list[str]
    daily_totals: DailyTotals
    history: Optional[HistoryComparison]   # None if first day
    recommendations: list[NutrientRecommendation]
