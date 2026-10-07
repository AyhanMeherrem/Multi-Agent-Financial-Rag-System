# Pure scoring helpers for the eval harness. No LLM, network or index access here,
# so these can be reused by verify_golden_set.py and unit tests.
import re

_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"',
                         "–": "-", "—": "-", "‑": "-",
                         "\xa0": " ", " ": " ", " ": " "})


def normalize_text(text: str) -> str:
    # Lowercase, unify quotes/dashes, drop "$" and collapse whitespace so that
    # "Total net sales $ 391,035" and "Total net sales 391,035" compare equal
    text = text.translate(_QUOTES).lower().replace("$", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r" %", "%", text)
    return text.strip()


def find_snippet_ranks(snippets: list, nodes: list, companies: list, years: list) -> list:
    # nodes: list of (company, year, text) sorted by score, best first.
    # Returns the 1-based rank of the first node that contains each snippet and matches
    # the expected company/year (None if not found). Empty companies/years means "any".
    ranks = []
    for snippet in snippets:
        target = normalize_text(snippet)
        rank = None
        for i, (company, year, text) in enumerate(nodes, start=1):
            if companies and company not in companies:
                continue
            if years and year not in years:
                continue
            if target in normalize_text(text):
                rank = i
                break
        ranks.append(rank)
    return ranks


_SCALES = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
_NUMBER_RE = re.compile(
    r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s*(%|percent|thousand|million|billion|trillion)?",
    re.IGNORECASE,
)


def parse_numbers(text: str) -> list:
    # Returns (value, tolerance, unit) for every number in text. Tolerance is half of the
    # last written digit, so "391.0 billion" matches 391,035 million but "392 billion" does not.
    found = []
    for whole, decimals, unit in _NUMBER_RE.findall(text):
        decimals = decimals or ""
        value = float(whole.replace(",", "") + ("." + decimals if decimals else ""))
        tolerance = 0.5 * 10 ** (-len(decimals))
        unit = (unit or "").lower()
        if unit == "percent":
            unit = "%"
        found.append((value, tolerance, unit))
    return found


def number_in_answer(reference: str, answer: str) -> bool:
    # reference examples: "391,035 million", "75.4 billion", "46.9%", "6.08", "228,000"
    parsed = parse_numbers(reference)
    if not parsed:
        return False
    ref_value, _, ref_unit = parsed[0]
    if ref_unit in _SCALES:
        ref_value *= _SCALES[ref_unit]
    for value, tolerance, unit in parse_numbers(answer.translate(_QUOTES).replace("$", " ")):
        if ref_unit == "%":
            candidates = [(value, tolerance)] if unit == "%" else []
        elif ref_unit in _SCALES:
            if unit in _SCALES:
                candidates = [(value * _SCALES[unit], tolerance * _SCALES[unit])]
            elif unit == "":
                # Filing tables are "in millions", so a bare 391,035 means 391,035 million
                candidates = [(value * 1e6, tolerance * 1e6), (value, tolerance)]
            else:
                candidates = []
        else:
            candidates = [(value, tolerance)] if unit == "" else []
        if any(abs(c - ref_value) <= tol + 1e-9 for c, tol in candidates):
            return True
    return False


_REFUSAL_RE = re.compile(
    r"i can only answer|not (?:available|provided|included|contained|found|present|mentioned|disclosed|covered)"
    r"|does not (?:contain|include|provide|mention|cover)|do not (?:contain|include|provide|have)"
    r"|no (?:information|data|details|mention)|cannot (?:answer|provide|determine|find)|can't|unable to"
    r"|outside (?:the |my )?scope|not in the (?:indexed|provided|retrieved)",
    re.IGNORECASE,
)


def is_refusal(answer: str) -> bool:
    return bool(_REFUSAL_RE.search(answer or ""))


def contains_forbidden(answer: str, forbidden: list) -> bool:
    lowered = (answer or "").lower()
    return any(f.lower() in lowered for f in forbidden)


def percentile(values: list, pct: float):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]
