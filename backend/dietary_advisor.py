import json
import re


def _compact_totals_df(totals_df):
    if totals_df.empty:
        return []

    preferred_cols = [
        "calories",
        "protein",
        "fat",
        "carbohydrate",
        "fiber",
        "sodium",
    ]

    available_cols = [c for c in preferred_cols if c in totals_df.columns]
    trimmed = totals_df[available_cols].head(1) if available_cols else totals_df.head(1)

    return trimmed.fillna("").to_dict(orient="records")


def _compact_recommendations_df(recommendations_df):
    if recommendations_df.empty:
        return []

    preferred_cols = [
        "nutrient",
        "status",
        "current",
        "target",
        "unit",
        "gap",
        "suggestions",
    ]

    available_cols = [c for c in preferred_cols if c in recommendations_df.columns]
    trimmed = recommendations_df[available_cols] if available_cols else recommendations_df

    return trimmed.fillna("").to_dict(orient="records")


def build_dietary_advice_prompt(totals_df, recommendations_df):
    totals_records = _compact_totals_df(totals_df)
    recommendation_records = _compact_recommendations_df(recommendations_df)

    return f"""
You are a nutrition interpretation assistant.

You will be given:
1. daily nutrient totals
2. rule-based nutrient recommendations

Your job:
- give a short plain-English interpretation of the person's diet for today
- identify likely deficiencies, excesses, or imbalances
- mention notable strengths too
- suggest practical foods that could help address any likely deficiencies
- keep suggestions realistic and food-based
- do not give medical advice
- do not mention that you are an AI
- keep the response brief

Return ONLY valid JSON in this exact format:
{{
  "summary": "2-3 sentence overall interpretation",
  "strengths": ["max 3 items"],
  "concerns": ["max 3 items"],
  "food_suggestions": [
    {{
      "issue": "low fiber",
      "foods": ["lentils", "oats", "berries"]
    }}
  ]
}}

DAILY_TOTALS:
{json.dumps(totals_records, ensure_ascii=False)}

RULE_BASED_RECOMMENDATIONS:
{json.dumps(recommendation_records, ensure_ascii=False)}
""".strip()


def extract_json_object(text: str):
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {
            "summary": text,
            "strengths": [],
            "concerns": [],
            "food_suggestions": [],
        }

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {
            "summary": text,
            "strengths": [],
            "concerns": [],
            "food_suggestions": [],
        }


def get_llm_dietary_advice(totals_df, recommendations_df, llm_call):
    prompt = build_dietary_advice_prompt(
        totals_df=totals_df,
        recommendations_df=recommendations_df,
    )

    raw_response = llm_call(prompt)
    parsed = extract_json_object(raw_response)

    return {
        "summary": parsed.get("summary", ""),
        "strengths": parsed.get("strengths", []) or [],
        "concerns": parsed.get("concerns", []) or [],
        "food_suggestions": parsed.get("food_suggestions", []) or [],
    }