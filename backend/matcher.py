"""
Food name matching: full TF-IDF cosine search over the SR Legacy pool.

The pool is ~7,700 SR Legacy foods — small enough that a full cosine
similarity search is fast (~ms) and avoids the truncation bugs of a
pre-filter approach (first-N substring hits often miss the correct entry).

Matching quality improvements (applied in order):
  1. Plural normalization — "apples" → "apple", "bananas" → "banana"
  2. Query expansion — reversed word order added so "whole milk" also
     matches "milk, whole" (USDA inverted naming convention).
  3. Prefix boost (×1.3) — descriptions whose first word equals the
     query's first word; "Apples, raw" beats "Babyfood, juice, apple".
  4. First-segment boost (×1.2) — descriptions whose first comma-segment
     contains any query word; "Milk, whole" beats "Yogurt, plain, whole
     milk" for the query "whole milk".
  5. Raw/fresh boost (×1.1) — when the query has no preparation keyword,
     lightly prefer raw/fresh descriptions; "Egg, whole, raw, fresh"
     beats "Egg, whole, dried" for the query "egg".

Usage:
    from backend import matcher, data_loader
    data_loader.load()
    matcher.build_index()
    match = matcher.find_best_match("apple")
"""
import re
import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.config import MATCH_THRESHOLD
from backend.models import FoodMatch
import backend.data_loader as data_loader



logger = logging.getLogger(__name__)

_vectorizer: TfidfVectorizer | None = None
_tfidf_matrix = None        # sparse matrix: pool_size × n_grams
_pool: pd.DataFrame = pd.DataFrame()

_PREFIX_BOOST    = 1.3   # description starts with query's first word
_FIRST_SEG_BOOST = 1.2   # first comma-segment contains a query word
_RAW_BOOST       = 1.2   # single-word queries prefer raw/fresh descriptions
_DRIED_PENALTY   = 0.80  # descriptions with "dried"/"dehydrated" when query is prep-agnostic
_PART_PENALTY    = 0.85  # descriptions specifying a food part ("white", "yolk") not in query

# Words that indicate a preparation method — used to decide when to apply penalties
_PREP_KEYWORDS = frozenset([
    "raw", "fresh", "cooked", "boiled", "baked", "fried", "roasted",
    "dried", "dehydrated", "frozen", "canned", "smoked", "pickled",
    "grilled", "steamed", "stewed", "braised", "poached", "toasted",
])

# Words that specify a part of an animal/food (penalised when not in query)
_FOOD_PART_KEYWORDS = frozenset([
    "white", "yolk", "liver", "kidney", "heart", "gizzard",
    "skin", "bone", "marrow", "blood", "tripe",
])


def _normalize(text: str) -> str:
    """
    Lowercase, strip parenthetical notes, and strip common plural suffixes.
    "apples" → "apple", "bananas" → "banana", "berries" → "berry".
    Parenthetical notes ("Includes foods for USDA's Food Distribution Program")
    add many non-discriminative tokens that hurt cosine similarity scores.
    """
    # Remove parenthetical / bracketed notes before tokenizing
    text = re.sub(r"\([^)]*\)", "", text.lower())
    tokens = re.findall(r"\b[a-z]+\b", text)
    result = []
    for t in tokens:
        if t.endswith("ies") and len(t) > 4:
            t = t[:-3] + "y"
        elif t.endswith("ves") and len(t) > 4:
            t = t[:-3] + "f"
        elif t.endswith("es") and len(t) > 4 and t[-3] in "sxz":
            t = t[:-2]
        elif (
            t.endswith("s")
            and len(t) > 3
            and not t.endswith(("ss", "us", "ous", "ias", "eas"))
        ):
            t = t[:-1]
        result.append(t)
    return " ".join(result)


def _has_prep_keyword(q: str) -> bool:
    """Return True if the query already contains a preparation method word."""
    return bool(set(q.lower().split()) & _PREP_KEYWORDS)


def _has_food_part_keyword(q: str) -> bool:
    """Return True if the query specifies a food part (e.g. 'egg white')."""
    return bool(set(q.lower().split()) & _FOOD_PART_KEYWORDS)


def build_index() -> None:
    """Build the TF-IDF index over the SR Legacy pool. Call once after data_loader.load()."""
    global _vectorizer, _tfidf_matrix, _pool

    _pool = data_loader.get_pool().copy()
    logger.info("Building TF-IDF index over %d SR Legacy foods …", len(_pool))

    # Normalised full description for TF-IDF indexing and prefix-boost check
    _pool["description_norm"] = _pool["description_lower"].apply(_normalize)

    # First comma-segment normalised — used by first-segment boost
    _pool["first_seg_norm"] = (
        _pool["description_lower"]
        .str.split(",")
        .str[0]
        .str.strip()
        .apply(_normalize)
    )

    _vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    _tfidf_matrix = _vectorizer.fit_transform(_pool["description_norm"])
    logger.info("TF-IDF index built: %s", _tfidf_matrix.shape)


def find_best_match(query: str) -> FoodMatch:
    """
    Return the best SR Legacy match for a user query using full TF-IDF search.
    Returns FoodMatch with matched=False if best score < MATCH_THRESHOLD.
    """
    if _vectorizer is None or _tfidf_matrix is None:
        raise RuntimeError("matcher.build_index() has not been called.")
    return _tfidf_match_global(query.strip())


def find_top_k(query: str, k: int = 5) -> list[FoodMatch]:
    """Return top-k matches for interactive disambiguation."""
    if _vectorizer is None or _tfidf_matrix is None:
        raise RuntimeError("matcher.build_index() has not been called.")

    sims = _score(query.strip())
    top_indices = np.argpartition(sims, -k)[-k:]
    top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

    return [
        FoodMatch(
            input_name=query,
            usda_description=_pool.iloc[idx]["food_description"],
            fdc_id=int(_pool.iloc[idx]["food_id"]),
            score=round(float(sims[idx]) * 100, 1),
            matched=float(sims[idx]) * 100 >= MATCH_THRESHOLD,
        )
        for idx in top_indices
    ]


def _score(q: str) -> np.ndarray:
    """
    Return a boosted similarity array for query q over the entire pool.
    Applies all five matching improvements (see module docstring).
    """
    q_norm = _normalize(q)
    words = q_norm.split()

    # Query expansion: include reversed word order so "whole milk" also
    # matches USDA's "Milk, whole" (which indexes the bigram "milk whole").
    if len(words) >= 2:
        q_expanded = q_norm + " " + " ".join(reversed(words))
    else:
        q_expanded = q_norm

    q_vec = _vectorizer.transform([q_expanded])
    sims = cosine_similarity(q_vec, _tfidf_matrix).flatten()

    # 1. Prefix boost: description starts with query's first word
    q_first = words[0] if words else q_norm
    starts_with = _pool["description_norm"].str.startswith(q_first)
    sims[starts_with.values] *= _PREFIX_BOOST

    # 2. First-segment boost: primary category of food matches a query word.
    #    "Milk, whole, …" → first_seg = "milk" — boosted for "whole milk".
    #    "Yogurt, plain, whole milk" → first_seg = "yogurt" — not boosted.
    q_words = set(words)
    first_seg_match = _pool["first_seg_norm"].apply(
        lambda s: bool(set(s.split()) & q_words)
    )
    sims[first_seg_match.values] *= _FIRST_SEG_BOOST

    # 3. Raw/fresh boost for single-word queries: "egg", "apple", "banana" etc.
    #    naturally refer to the unprocessed form.  Multi-word queries ("brown rice",
    #    "chicken breast") are more specific and don't get this boost to avoid
    #    preferring raw rice or raw chicken over cooked forms.
    if len(words) == 1 and not _has_prep_keyword(q):
        has_raw = _pool["description_norm"].str.contains(r"\braw\b|\bfresh\b", regex=True)
        sims[has_raw.values] *= _RAW_BOOST

    # 5. Dried/dehydrated penalty: when the query has no prep keyword, down-rank
    #    descriptions with "dried"/"dehydrated" so that fresh forms are preferred.
    if not _has_prep_keyword(q):
        is_dried = _pool["description_norm"].str.contains(
            r"\bdried\b|\bdehydrated\b", regex=True
        )
        sims[is_dried.values] *= _DRIED_PENALTY

    # 6. Food-part penalty: when the query doesn't name a specific part, down-rank
    #    descriptions that do (egg white, egg yolk, liver, etc.).
    #    Fixes: "egg" → "Egg, whole, raw, fresh" over "Egg, white, raw, fresh".
    if not _has_food_part_keyword(q):
        q_words = set(q_norm.split())
        has_part = _pool["description_norm"].apply(
            lambda s: bool((set(s.split()) & _FOOD_PART_KEYWORDS) - q_words)
        )
        sims[has_part.values] *= _PART_PENALTY

    return sims


def _tfidf_match_global(q: str) -> FoodMatch:
    """Full TF-IDF cosine search with all matching boosts applied."""
    sims = _score(q)
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx]) * 100
    best_row = _pool.iloc[best_idx]
    return FoodMatch(
        input_name=q,
        usda_description=best_row["food_description"],
        fdc_id=int(best_row["food_id"]),
        score=round(best_score, 1),
        matched=best_score >= MATCH_THRESHOLD,
    )
