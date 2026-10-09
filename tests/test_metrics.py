import pytest

from eval.metrics import contains_forbidden, find_snippet_ranks, is_refusal, normalize_text, number_in_answer


@pytest.mark.parametrize("reference, answer, expected", [
    ("391,035 million", "Apple had $391,035 million in net sales", True),
    ("391,035 million", "about $391.0 billion", True),
    ("391,035 million", "$ 391,035", True),
    ("391,035 million", "$392 billion", False),
    ("75.4 billion", "75,400 million", True),
    ("46.9%", "46.9 percent", True),
    ("46.9%", "46.2%", False),
    ("6.08", "diluted EPS of $6.08", True),
    ("6.08", "$6.11", False),
    ("11.80", "$11.8", True),
    ("228,000", "approximately 228,000 people", True),
    ("93,736 million", "93,736 million dollars", True),
])
def test_number_in_answer(reference, answer, expected):
    assert number_in_answer(reference, answer) is expected


def test_normalize_text_handles_tables_and_dollar_signs():
    assert normalize_text("Total net sales | $ 391,035 | 2 %") == "total net sales 391,035 2%"


def test_snippet_ranks_respect_company_and_year():
    nodes = [("MSFT", "2024", "Total revenue 245,122"), ("AAPL", "2023", "Net income 93,736"), ("AAPL", "2024", "Net income 93,736")]
    assert find_snippet_ranks(["net income 93,736"], nodes, ["AAPL"], ["2024"]) == [3]
    assert find_snippet_ranks(["missing"], nodes, [], []) == [None]


def test_refusal_and_forbidden_detection():
    assert is_refusal("The indexed filings do not contain Apple's headcount.")
    assert is_refusal("I can only answer questions about the indexed SEC 10-K filings.")
    assert not is_refusal("Net sales were 391,035 million.")
    assert contains_forbidden("banana BANANA banana", ["banana banana"])
    assert not contains_forbidden("I won't do that.", ["banana banana"])
