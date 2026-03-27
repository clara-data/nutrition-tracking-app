import json
import re
from typing import Callable, List, Dict, Any


def _extract_json_array(text: str) -> list:
    """
    Tries to extract a JSON array from an LLM response.
    Accepts either raw JSON or a response with extra surrounding text.
    """
    text = text.strip()

    # Best case: full response is already JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    # Fallback: find first JSON array in the text
    match = re.search(r"\[\s*{.*}\s*]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

    return []


def _coerce_quantity(value: Any) -> int:
    """
    Converts quantity to a usable integer.
    Defaults to 1 if missing or invalid.
    """
    try:
        qty = int(round(float(value)))
        return max(qty, 1)
    except Exception:
        return 1


def _normalize_item(item: dict) -> dict | None:
    """
    Normalizes one parsed item into:
    {
        "food_name": str,
        "quantity": int
    }
    """
    if not isinstance(item, dict):
        return None

    food_name = str(
        item.get("food_name")
        or item.get("usda_food")
        or item.get("food")
        or ""
    ).strip()

    if not food_name:
        return None

    quantity = _coerce_quantity(item.get("quantity", 1))

    return {
        "food_name": food_name,
        "quantity": quantity,
    }


def build_meal_parser_prompt(
    meal_text: str,
    candidate_foods: List[str] | None = None,
    max_candidates: int = 250,
) -> List[Dict[str, str]]:
    """
    Returns chat-style messages for an LLM call.

    If candidate_foods is provided, the model is instructed to choose only
    from that list. This is optional, but helps grounding.
    """
    candidate_foods = candidate_foods or []
    trimmed_candidates = candidate_foods[:max_candidates]

    system_prompt = """
You convert a user's freeform meal description into a JSON array of food items.

Return JSON only.
Do not use markdown.
Do not explain anything.
Do not include any text before or after the JSON.

Output format:
[
  {"food_name": "exact food name", "quantity": 1}
]

Rules:
- Split combined meal descriptions into separate foods when appropriate.
- Use the best reasonable food match.
- If quantity is unclear, use 1.
- Quantity must be a positive integer.
- Do not estimate calories or nutrients.
- Do not include notes or comments.
"""

    if trimmed_candidates:
        user_prompt = f"""
User meal description:
{meal_text}

Choose food_name values only from this allowed list:
{json.dumps(trimmed_candidates, ensure_ascii=False)}

Return JSON only.
"""
    else:
        user_prompt = f"""
User meal description:
{meal_text}

Return a JSON array of best-guess food items.
Use concise, specific food names.
Return JSON only.
"""

    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]


def parse_meal_with_llm(
    meal_text: str,
    llm_call: Callable[[List[Dict[str, str]]], str],
    candidate_foods: List[str] | None = None,
    max_candidates: int = 250,
) -> List[Dict[str, Any]]:
    """
    Main entry point.

    Parameters
    ----------
    meal_text:
        Freeform user text like:
        "I had 2 eggs, toast with butter, and coffee"

    llm_call:
        A function you provide that accepts a list of chat messages and
        returns the model's raw text response.

    candidate_foods:
        Optional USDA food names to constrain the model.

    Returns
    -------
    List[dict]
        Example:
        [
            {"food_name": "Egg, whole, cooked, scrambled", "quantity": 2},
            {"food_name": "Bread, whole-wheat, commercially prepared", "quantity": 2},
            {"food_name": "Butter, salted", "quantity": 1}
        ]
    """
    if not str(meal_text).strip():
        return []

    messages = build_meal_parser_prompt(
        meal_text=meal_text,
        candidate_foods=candidate_foods,
        max_candidates=max_candidates,
    )

    raw_response = llm_call(messages)
    parsed_items = _extract_json_array(raw_response)

    normalized_items = []
    for item in parsed_items:
        normalized = _normalize_item(item)
        if normalized:
            normalized_items.append(normalized)

    return normalized_items