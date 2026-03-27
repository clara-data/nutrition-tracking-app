import re
from rapidfuzz import fuzz


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def singularize_token(token: str) -> str:
    token = token.lower().strip()

    if len(token) <= 3:
        return token

    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"  # berries -> berry

    if token.endswith(("ches", "shes", "xes", "zes", "ses")) and len(token) > 4:
        return token[:-2]  # dishes -> dish

    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]  # apples -> apple

    return token


def normalize_tokens(tokens):
    return [singularize_token(tok) for tok in tokens]


def build_food_index(options):
    index = []

    for opt in options:
        norm = normalize_text(opt)
        tokens = norm.split()
        norm_tokens = normalize_tokens(tokens)

        index.append({
            "display": opt,
            "norm": norm,
            "tokens": tokens,
            "token_set": set(tokens),
            "norm_tokens": norm_tokens,
            "norm_token_set": set(norm_tokens),
            "norm_token_string": " ".join(norm_tokens),
        })

    return index


def score_option(query: str, row: dict) -> float:
    q = normalize_text(query)
    q_tokens = q.split()
    q_norm_tokens = normalize_tokens(q_tokens)
    q_norm_string = " ".join(q_norm_tokens)

    norm = row["norm"]
    tokens = row["tokens"]
    norm_tokens = row["norm_tokens"]
    norm_token_set = row["norm_token_set"]
    norm_token_string = row["norm_token_string"]

    score = 0.0

    # Exact raw phrase
    if norm == q:
        score += 300

    # Exact normalized phrase (apple == apples)
    if norm_token_string == q_norm_string:
        score += 260

    # Exact single-token normalized match
    if len(q_norm_tokens) == 1 and q_norm_tokens[0] in norm_token_set:
        score += 220

    # Raw full-string prefix
    if norm.startswith(q):
        score += 140

    # First-token normalized prefix
    if q_norm_tokens and norm_tokens and norm_tokens[0].startswith(q_norm_tokens[0]):
        score += 100

    # Normalized token overlap
    overlap = len(set(q_norm_tokens) & norm_token_set)
    score += overlap * 50

    # Ordered prefix-token matches
    ordered_matches = 0
    for qt in q_norm_tokens:
        if any(tok.startswith(qt) for tok in norm_tokens):
            ordered_matches += 1
    score += ordered_matches * 30

    # Whole-word raw match
    if q and re.search(rf"\b{re.escape(q)}\b", norm):
        score += 80
    # Weak raw substring fallback
    elif q in norm:
        score += 15

    # Fuzzy signals
    score += fuzz.ratio(q, norm) * 0.20
    score += fuzz.partial_ratio(q, norm) * 0.10
    score += fuzz.token_set_ratio(q_norm_string, norm_token_string) * 0.35

    # Slight penalty for very long labels
    score -= max(len(norm) - 40, 0) * 0.5

    return score


def get_food_suggestions_from_options(query: str, options, limit: int = 15):
    if not options:
        return []

    query = normalize_text(query)
    if not query:
        return options[:limit]

    index = build_food_index(options)

    q_tokens = query.split()
    q_norm_tokens = normalize_tokens(q_tokens)

    candidates = []
    for row in index:
        raw_match = (
            query in row["norm"]
            or row["norm"].startswith(query)
            or any(tok.startswith(query) for tok in row["tokens"])
        )

        normalized_token_match = any(qt in row["norm_token_set"] for qt in q_norm_tokens)

        normalized_prefix_match = any(
            tok.startswith(qt)
            for qt in q_norm_tokens
            for tok in row["norm_tokens"]
        )

        word_match = re.search(rf"\b{re.escape(query)}\b", row["norm"]) is not None

        if raw_match or normalized_token_match or normalized_prefix_match or word_match:
            candidates.append(row)

    if not candidates:
        candidates = index

    scored = [(row["display"], score_option(query, row)) for row in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)

    seen = set()
    results = []
    for display, _ in scored:
        if display not in seen:
            seen.add(display)
            results.append(display)
        if len(results) >= limit:
            break

    return results