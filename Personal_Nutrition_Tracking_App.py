import sys
from pathlib import Path
import io
import json
import hashlib

APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import pandas as pd
import streamlit as st

import rbi_pipeline as nutrition_pipeline
import backend.data_loader as food_data_loader
from backend.meal_parser import parse_meal_with_llm
from backend.dietary_advisor import get_llm_dietary_advice
from backend.llm_client import call_meal_parser_llm, call_dietary_advice_llm

st.set_page_config(page_title="Nutrition Tracker", layout="wide")
##rbi



##

USER_DATA_FOLDER = APP_ROOT / "user_data"
USER_DATA_FOLDER.mkdir(parents=True, exist_ok=True)

FOOD_LOG_CSV_PATH = USER_DATA_FOLDER / "user_input.csv"
PIPELINE_OUTPUT_PATH = USER_DATA_FOLDER / "output.csv"


# =================================
# data helpers
# =================================

@st.cache_data
def load_available_food_descriptions():
    food_data_loader.load()
    df = food_data_loader.get_pool()

    if "food_description" in df.columns:
        col = "food_description"
    elif "description" in df.columns:
        col = "description"
    else:
        return []

    return (
        df[col]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
##Mar 26 rbi
@st.cache_data
def load_food_pool():
    food_data_loader.load()
    return food_data_loader.get_pool().copy()

@st.cache_data
def load_food_portions():
    return pd.read_csv(APP_ROOT / "USDA_data" / "food_portion.csv")

#if st.button("DEBUG: show food pool columns"):
#    df_debug = load_food_pool()
#    st.write(df_debug.columns.tolist())
#    st.write(df_debug.head())

#if st.button("DEBUG: show food portion columns"):
#    df_portion = load_food_portions()
#    st.write(df_portion.columns.tolist())
#    st.write(df_portion.head())


def get_default_portion_for_food(food_name):
    food_df = load_food_pool()
    portion_df = load_food_portions()

    matches = food_df[
        food_df["food_description"].astype(str).str.strip() == str(food_name).strip()
    ]

    if matches.empty:
        return {
            "unit": "serving",
            "portion_description": "1 serving",
            "gram_weight": None,
        }

    food_id = matches.iloc[0]["food_id"]
    portion_matches = portion_df[portion_df["food_id"] == food_id].copy()

    if portion_matches.empty:
        return {
            "unit": "serving",
            "portion_description": "1 serving",
            "gram_weight": None,
        }

    if "seq_num" in portion_matches.columns:
        portion_matches = portion_matches.sort_values("seq_num")

    row = portion_matches.iloc[0]

    amount = row["amount"] if pd.notna(row["amount"]) else 1

    portion_description = None
    if pd.notna(row["portion_description"]):
        portion_description = str(row["portion_description"]).strip()
        if portion_description == "":
            portion_description = None

    modifier = None
    if pd.notna(row["modifier"]):
        modifier = str(row["modifier"]).strip()
        if modifier == "":
            modifier = None

    gram_weight = row["gram_weight"] if pd.notna(row["gram_weight"]) else None

    if portion_description:
        label = portion_description
    elif modifier:
        label = f"{amount} {modifier}"
    else:
        label = "1 serving"

    return {
        "unit": modifier if modifier else "serving",
        "portion_description": label,
        "gram_weight": gram_weight,
    }


##

def csv_lines_to_df(lines):
    if not lines:
        return pd.DataFrame()

    try:
        return pd.read_csv(io.StringIO("\n".join(lines)))
    except Exception:
        return pd.DataFrame()


def parse_pipeline_output(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    section_1_idx = next((i for i, line in enumerate(lines) if "### 1" in line), None)
    section_2_idx = next((i for i, line in enumerate(lines) if "### 2" in line), None)
    section_3_idx = next((i for i, line in enumerate(lines) if "### 3" in line), None)

    if section_1_idx is None or section_2_idx is None or section_3_idx is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def section(start, end=None):
        chunk = lines[start:end] if end is not None else lines[start:]
        return [line for line in chunk if line.strip() and not line.startswith("###")]

    matched_df = csv_lines_to_df(section(section_1_idx + 1, section_2_idx))
    totals_df = csv_lines_to_df(section(section_2_idx + 1, section_3_idx))
    recs_df = csv_lines_to_df(section(section_3_idx + 1))

    return matched_df, totals_df, recs_df


def daily_totals_summary(df):
    if df.empty:
        return pd.DataFrame()

    row = df.iloc[0]
    return pd.DataFrame(
        {
            "Nutrient": ["Calories", "Protein", "Fat", "Carbohydrate", "Fiber", "Sodium"],
            "Value": [
                row.get("calories", ""),
                row.get("protein", ""),
                row.get("fat", ""),
                row.get("carbohydrate", ""),
                row.get("fiber", ""),
                row.get("sodium", ""),
            ],
            "Unit": ["kcal", "g", "g", "g", "g", "mg"],
        }
    )


def recommendation_cards(df):
    if df.empty:
        return []

    suggestion_cols = [c for c in df.columns if "suggest" in c.lower()]
    cards = []

    for _, row in df.iterrows():
        if suggestion_cols:
            suggestions = [
                str(row.get(col, "")).strip()
                for col in suggestion_cols
                if str(row.get(col, "")).strip()
            ]
        else:
            raw = str(row.get("suggestions", "")).strip()
            suggestions = [item.strip() for item in raw.split(";") if item.strip()] if raw else []

        cards.append(
            {
                "nutrient": row.get("nutrient", ""),
                "status": row.get("status", ""),
                "current": row.get("current", ""),
                "target": row.get("target", ""),
                "unit": row.get("unit", ""),
                "gap": row.get("gap", ""),
                "suggestions": suggestions,
            }
        )

    return cards


def make_analysis_fingerprint(totals_df, recommendations_df):
    payload = {
        "totals": totals_df.fillna("").to_dict(orient="records"),
        "recommendations": recommendations_df.fillna("").to_dict(orient="records"),
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# =================================
# state
# =================================

def init_state():
    st.session_state.setdefault("logged_food_items", [])
    st.session_state.setdefault("matched_foods_df", pd.DataFrame())
    st.session_state.setdefault("daily_totals_df", pd.DataFrame())
    st.session_state.setdefault("recommendations_df", pd.DataFrame())

    st.session_state.setdefault("llm_dietary_advice", None)
    st.session_state.setdefault("llm_advice_fingerprint", None)

    st.session_state.setdefault("freeform_meal_text", "")
    st.session_state.setdefault("selected_food", None)
    st.session_state.setdefault("active_input", "describe")
    st.session_state.setdefault("clear_food_inputs", False)


# =================================
# actions
# =================================

def clear_food_inputs_if_needed():
    if st.session_state.clear_food_inputs:
        st.session_state.freeform_meal_text = ""
        st.session_state.selected_food = None
        st.session_state.clear_food_inputs = False


#def add_food(food_name):
#    for item in st.session_state.logged_food_items:
#        if item["food_name"] == food_name:
#            item["quantity"] += 1
#           return

#    st.session_state.logged_food_items.append(
#        {
#            "food_name": food_name,
#            "quantity": 1,
#        }
#    )

##rbi

def add_food(food_name):
    for item in st.session_state.logged_food_items:
        if item["food_name"] == food_name:
            item["quantity"] += 1
            return

    portion_info = get_default_portion_for_food(food_name)

    st.session_state.logged_food_items.append(
        {
            "food_name": food_name,
            "quantity": 1,
            "unit": portion_info["unit"],
            "portion_description": portion_info["portion_description"],
            "gram_weight": portion_info["gram_weight"],
        }
    )

##


#def add_items_to_food_log(items):
#    for new_item in items:
#        food_name = str(new_item.get("food_name", "")).strip()
#        quantity_raw = new_item.get("quantity", 1)

#        if not food_name:
#            continue

#        try:
#            quantity = int(quantity_raw)
#        except Exception:
#            quantity = 1

#        quantity = max(quantity, 1)

#        for existing in st.session_state.logged_food_items:
#            if existing["food_name"] == food_name:
#                existing["quantity"] += quantity
#                break
#        else:
#            st.session_state.logged_food_items.append(
#                {
#                    "food_name": food_name,
#                    "quantity": quantity,
#                }
#            )

##rbi

def add_items_to_food_log(items):
    for new_item in items:
        food_name = str(new_item.get("food_name", "")).strip()
        quantity_raw = new_item.get("quantity", 1)

        if not food_name:
            continue

        try:
            quantity = int(quantity_raw)
        except Exception:
            quantity = 1

        quantity = max(quantity, 1)

        for existing in st.session_state.logged_food_items:
            if existing["food_name"] == food_name:
                existing["quantity"] += quantity
                break
        else:
            portion_info = get_default_portion_for_food(food_name)

            st.session_state.logged_food_items.append(
                {
                    "food_name": food_name,
                    "quantity": quantity,
                    "unit": portion_info["unit"],
                    "portion_description": portion_info["portion_description"],
                    "gram_weight": portion_info["gram_weight"],
                }
            )

##

def save_food_log():
    pd.DataFrame(st.session_state.logged_food_items).to_csv(FOOD_LOG_CSV_PATH, index=False)


def run_base_analysis():
    save_food_log()
    nutrition_pipeline.run()

    matched_df, totals_df, recs_df = parse_pipeline_output(PIPELINE_OUTPUT_PATH)

    st.session_state.matched_foods_df = matched_df
    st.session_state.daily_totals_df = totals_df
    st.session_state.recommendations_df = recs_df

    # Invalidate old LLM advice whenever analysis changes
    st.session_state.llm_dietary_advice = None
    st.session_state.llm_advice_fingerprint = None


def run_llm_advice():
    if st.session_state.daily_totals_df.empty and st.session_state.recommendations_df.empty:
        return

    fingerprint = make_analysis_fingerprint(
        st.session_state.daily_totals_df,
        st.session_state.recommendations_df,
    )

    if (
        st.session_state.llm_advice_fingerprint == fingerprint
        and st.session_state.llm_dietary_advice is not None
    ):
        return

    try:
        st.session_state.llm_dietary_advice = get_llm_dietary_advice(
            totals_df=st.session_state.daily_totals_df,
            recommendations_df=st.session_state.recommendations_df,
            llm_call=call_dietary_advice_llm,
        )
        st.session_state.llm_advice_fingerprint = fingerprint
    except Exception as e:
        st.session_state.llm_dietary_advice = {
            "summary": "",
            "strengths": [],
            "concerns": [f"LLM dietary interpretation failed: {e}"],
            "food_suggestions": [],
        }
        st.session_state.llm_advice_fingerprint = fingerprint


# =================================
# ui helpers
# =================================

def metric_card(label, value, unit):
    st.markdown(
        f"""
        <div style="
            background-color:#f7f7f7;
            border-radius:12px;
            padding:18px 10px;
            text-align:center;
            border-top:4px solid #2ecc71;
            color:#222;">
            <div style="font-size:14px; color:#666; margin-bottom:6px;">
                {str(label).upper()}
            </div>
            <div style="font-size:34px; font-weight:700; line-height:1.1;">
                {value}
            </div>
            <div style="font-size:14px; color:#666;">
                {unit}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def recommendation_card(card):
    suggestions_html = "".join(f"<li>{s}</li>" for s in card["suggestions"])

    st.markdown(
        f"""
        <div style="
            background-color:#f7f7f7;
            border-left:4px solid #f39c12;
            border-radius:12px;
            padding:16px 18px;
            margin-bottom:14px;
            color:#222;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div style="font-size:28px; font-weight:700;">
                    {card["nutrient"]}
                </div>
                <div style="
                    background:#fde9b6;
                    color:#8a5a00;
                    padding:4px 10px;
                    border-radius:999px;
                    font-size:14px;
                    font-weight:600;">
                    {card["status"]}
                </div>
            </div>
            <div style="margin-top:10px; font-size:16px; color:#444;">
                Current: {card["current"]} {card["unit"]}
                &nbsp; | &nbsp;
                Target: {card["target"]} {card["unit"]}
                &nbsp; | &nbsp;
                Gap: {card["gap"]} {card["unit"]}
            </div>
            <ul style="margin-top:10px; font-size:16px; color:#333;">
                {suggestions_html}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =================================
# ui sections
# =================================

def render_food_input_area():
    clear_food_inputs_if_needed()
    foods = load_available_food_descriptions()

    st.markdown("### Add Foods")

    mode = st.radio(
        "Choose input method",
        options=["Describe a Meal", "Search Specific Food"],
        horizontal=True,
        index=0 if st.session_state.active_input == "describe" else 1,
        label_visibility="collapsed",
    )

    st.session_state.active_input = "describe" if mode == "Describe a Meal" else "search"

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    if st.session_state.active_input == "describe":
        st.markdown(
            """
            <div style="
                background-color:#f7f7f7;
                border-radius:12px;
                padding:14px;
                border:1px solid #e6e6e6;
                margin-bottom:10px;
            ">
                <div style="font-size:18px; font-weight:700; color:#222; margin-bottom:4px;">
                    Describe a Meal
                </div>
                <div style="font-size:14px; color:#666;">
                    Type what you ate in natural language, then add the parsed foods to your current list.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.text_area(
            "",
            key="freeform_meal_text",
            height=180,
            placeholder="Example: 2 scrambled eggs, toast with butter, and orange juice",
            label_visibility="collapsed",
        )

        if st.button("Add Parsed Foods", use_container_width=True, key="add_parsed_foods"):
            meal_text = st.session_state.freeform_meal_text.strip()

            if not meal_text:
                st.warning("Enter a meal description first.")
            else:
                with st.spinner("Parsing meal..."):
                    parsed_items = parse_meal_with_llm(
                        meal_text=meal_text,
                        llm_call=call_meal_parser_llm,
                    )

                if not parsed_items:
                    st.warning("Couldn't parse any foods from that description.")
                else:
                    add_items_to_food_log(parsed_items)
                    st.session_state.clear_food_inputs = True
                    st.success("Added parsed foods to current foods.")
                    st.rerun()

    else:
        st.markdown(
            """
            <div style="
                background-color:#f7f7f7;
                border-radius:12px;
                padding:14px;
                border:1px solid #e6e6e6;
                margin-bottom:10px;
            ">
                <div style="font-size:18px; font-weight:700; color:#222; margin-bottom:4px;">
                    Search Specific Food
                </div>
                <div style="font-size:14px; color:#666;">
                    Search the USDA food list and add one exact food directly to your current list.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.selectbox(
            "",
            options=foods,
            index=None,
            key="selected_food",
            placeholder="Start typing to search foods",
            label_visibility="collapsed",
        )

        if st.button("Add Selected Food", use_container_width=True, key="add_selected_food"):
            selected_food = st.session_state.selected_food

            if not selected_food:
                st.warning("Choose a food first.")
            else:
                add_food(selected_food)
                st.session_state.clear_food_inputs = True
                st.success("Added selected food to current foods.")
                st.rerun()


def render_logged_foods():
    st.markdown("### Current Foods")

    if not st.session_state.logged_food_items:
        st.caption("No foods added yet.")
        return

    for i, item in enumerate(st.session_state.logged_food_items):
        name_col, minus_col, qty_col, plus_col, remove_col = st.columns([4, 1, 1, 1, 1])

        #with name_col:
        #    st.markdown(f"**{item['food_name']}**")

        
    ##rbi    
        with name_col:
            portion_text = item.get("portion_description") or item.get("unit") or "serving"
            st.markdown(f"**{item['food_name']}**")
            st.caption(f"{item['quantity']} × {portion_text}")
    ##

        with minus_col:
            minus = st.button("-", key=f"minus_{i}")

        with qty_col:
            st.markdown(
                f"<div style='text-align:center; font-size:18px; font-weight:600;'>{item['quantity']}</div>",
                unsafe_allow_html=True,
            )

        with plus_col:
            plus = st.button("+", key=f"plus_{i}")

        with remove_col:
            remove = st.button("✕", key=f"remove_{i}")

        if minus:
            if item["quantity"] > 1:
                item["quantity"] -= 1
                st.rerun()

        if plus:
            item["quantity"] += 1
            st.rerun()

        if remove:
            st.session_state.logged_food_items.pop(i)
            st.rerun()


def render_food_log_panel():
    st.markdown("## Today's Food Log")
    render_food_input_area()
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    render_logged_foods()

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    analyze_col, llm_col = st.columns(2)

    with analyze_col:
        analyze_clicked = st.button("Analyze", use_container_width=True)

    with llm_col:
        llm_clicked = st.button("Generate LLM Advice", use_container_width=True)

    if analyze_clicked:
        if not st.session_state.logged_food_items:
            st.warning("Add at least one food before analyzing.")
        else:
            with st.spinner("Running nutrition analysis..."):
                run_base_analysis()
            st.rerun()

    if llm_clicked:
        if not st.session_state.logged_food_items:
            st.warning("Add at least one food before generating advice.")
        else:
            with st.spinner("Running nutrition analysis and generating LLM dietary advice..."):
                run_base_analysis()
                run_llm_advice()
            st.rerun()


def render_daily_totals():
    st.markdown("## Daily Totals")

    summary = daily_totals_summary(st.session_state.daily_totals_df)
    if summary.empty:
        st.caption("No daily totals available yet.")
        return

    cols_top = st.columns(3)
    cols_bottom = st.columns(3)

    for col, (_, row) in zip(cols_top, summary.iloc[:3].iterrows()):
        with col:
            metric_card(row["Nutrient"], row["Value"], row["Unit"])

    for col, (_, row) in zip(cols_bottom, summary.iloc[3:].iterrows()):
        with col:
            metric_card(row["Nutrient"], row["Value"], row["Unit"])


def render_matched_foods():
    st.markdown("## Matched Foods")

    if st.session_state.matched_foods_df.empty:
        st.caption("No matched foods to display yet.")
        return

    st.dataframe(
        st.session_state.matched_foods_df,
        use_container_width=True,
        hide_index=True,
    )


def render_recommendations():
    st.markdown("## Dietary Recommendations")

    cards = recommendation_cards(st.session_state.recommendations_df)
    if not cards:
        st.caption("No recommendations available yet.")
        return

    for card in cards:
        recommendation_card(card)


def render_llm_dietary_advice():
    st.markdown("## LLM Dietary Interpretation")

    advice = st.session_state.llm_dietary_advice
    if not advice:
        st.caption("No LLM interpretation generated yet.")
        return

    summary_text = advice.get("summary", "").strip()
    strengths = advice.get("strengths", []) or []
    concerns = advice.get("concerns", []) or []
    food_suggestions = advice.get("food_suggestions", []) or []

    with st.container(border=True):
        st.markdown("### Overall Interpretation")
        if summary_text:
            st.write(summary_text)
        else:
            st.caption("No summary returned.")

        st.markdown("### Strengths")
        if strengths:
            for item in strengths:
                st.write(f"- {item}")
        else:
            st.caption("No major strengths identified.")

        st.markdown("### Concerns")
        if concerns:
            for item in concerns:
                st.write(f"- {item}")
        else:
            st.caption("No major concerns identified.")

        st.markdown("### Foods That May Help")
        if food_suggestions:
            for item in food_suggestions:
                issue = str(item.get("issue", "")).strip()
                foods = item.get("foods", []) or []
                foods_text = ", ".join(str(f).strip() for f in foods if str(f).strip())

                if issue and foods_text:
                    st.write(f"- **{issue}:** {foods_text}")
                elif foods_text:
                    st.write(f"- {foods_text}")
        else:
            st.caption("No food suggestions returned.")


def render_analysis_panel():
    render_llm_dietary_advice()
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    render_daily_totals()
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    render_matched_foods()
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    render_recommendations()




# =================================
# app
# =================================

def main():
    init_state()

    left_col, right_col = st.columns([1, 1.25], gap="large")

    with left_col:
        render_food_log_panel()

    with right_col:
        render_analysis_panel()


if __name__ == "__main__":
    main()