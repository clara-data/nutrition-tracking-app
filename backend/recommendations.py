"""
DRI gap analysis and food suggestions.

Compares daily totals against DRI targets and returns a recommendation
for each nutrient that is deficient or excessive.
"""
from backend.config import DRI_TARGETS, NUTRIENT_FOODS
from backend.models import DailyTotals, FoodSuggestion, NutrientRecommendation

NUTRIENT_KEYS = ["calories", "protein", "fat", "carbohydrate", "fiber", "sodium"]


def build_recommendations(totals: DailyTotals) -> list[NutrientRecommendation]:
    """
    Compare each nutrient in totals against DRI_TARGETS.

    direction="min"    → flag if current < target  (deficient)
    direction="max"    → flag if current > target  (excess)
    direction="target" → flag if |current - target| > 10% of target
    """
    recommendations: list[NutrientRecommendation] = []

    for key in NUTRIENT_KEYS:
        dri = DRI_TARGETS[key]
        current = getattr(totals, key).value
        target = dri["target"]
        unit = dri["unit"]
        direction = dri["direction"]

        status, gap = _evaluate(current, target, direction)

        if status == "ok":
            continue

        suggestions = _get_suggestions(key, status)

        recommendations.append(NutrientRecommendation(
            nutrient=key,
            current=round(current, 1),
            target=target,
            unit=unit,
            status=status,
            gap=round(abs(gap), 1),
            suggestions=suggestions,
        ))

    return recommendations


def _evaluate(current: float, target: float, direction: str) -> tuple[str, float]:
    """Return (status, gap). gap is signed: positive = above target."""
    gap = current - target
    if direction == "min":
        return ("deficient", gap) if gap < 0 else ("ok", 0.0)
    if direction == "max":
        return ("excess", gap) if gap > 0 else ("ok", 0.0)
    # direction == "target": within ±10% is ok
    if abs(gap) > target * 0.10:
        status = "excess" if gap > 0 else "deficient"
        return status, gap
    return "ok", 0.0


def _get_suggestions(nutrient: str, status: str) -> list[FoodSuggestion]:
    """
    Return food suggestions.

    - Deficient nutrient → suggest foods high in that nutrient.
    - Excess nutrient    → no additions (sodium: explicitly empty).
    """
    if status == "excess":
        return []

    foods = NUTRIENT_FOODS.get(nutrient, [])
    return [
        FoodSuggestion(
            food=f["food"],
            serving=f["serving"],
            amount=f["amount"],
        )
        for f in foods[:3]   # top-3 suggestions
    ]
