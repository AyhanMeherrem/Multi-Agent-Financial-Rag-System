import json
from types import SimpleNamespace

import app.router.router_agent as router
from app.router.router_agent import IndexCatalog, RouterDecision, extract_filters, filing_years_for, unsupported_answer
from tests.helpers import FakeLLM, make_node


def test_decision_normalizes_llm_output():
    decision = RouterDecision.model_validate({
        "companies": ["aapl", " MSFT ", "AAPL"],
        "years": ["FY2024", 2025, "no year"],
        "sections": ["item 7", "8", "Item 1A", "Item 3", "Item 1"],
    })
    assert decision.companies == ["AAPL", "MSFT"]
    assert decision.years == ["2024", "2025"]
    assert decision.sections == ["Item 7", "Item 8", "Item 1A"]  # at most three


def test_extract_filters_parses_json_mode_output(catalog):
    reply = json.dumps({"companies": ["AAPL", "MSFT"], "years": ["2024"], "sections": ["Item 8"],
                        "in_scope": True, "unsupported_reason": None})
    decision = extract_filters("Compare revenue", catalog, llm=FakeLLM(reply))
    assert decision.companies == ["AAPL", "MSFT"] and decision.years == ["2024"] and decision.in_scope


def test_extract_filters_falls_back_to_unfiltered_search_on_bad_output(catalog):
    for reply in ["not json at all", '["a", "list"]']:
        decision = extract_filters("anything", catalog, llm=FakeLLM(reply))
        assert decision == RouterDecision()


def test_single_values_are_accepted_as_one_item_lists(catalog):
    reply = '{"companies": "AAPL", "years": "2024", "sections": "Item 8"}'
    decision = extract_filters("anything", catalog, llm=FakeLLM(reply))
    assert decision.companies == ["AAPL"] and decision.years == ["2024"] and decision.sections == ["Item 8"]


def test_router_prompt_lists_only_indexed_values(catalog):
    llm = FakeLLM("{}")
    extract_filters("question", catalog, llm=llm)
    prompt = llm.messages[0].content
    assert "AAPL (Apple), MSFT (Microsoft)" in prompt and "2024, 2025" in prompt
    assert '"Item 9A"' not in prompt  # not in this catalog


def test_prior_years_map_to_filings_that_contain_them(catalog):
    assert filing_years_for("2024", catalog) == ["2024"]
    assert filing_years_for("2023", catalog) == ["2024", "2025"]
    assert filing_years_for("2021", catalog) == []


def test_unsupported_answer(catalog):
    assert "do not include TSLA" in unsupported_answer(RouterDecision(companies=["TSLA"]), catalog)
    assert "fiscal year 2021" in unsupported_answer(RouterDecision(companies=["AAPL"], years=["2021"]), catalog)
    assert "I can only answer" in unsupported_answer(RouterDecision(in_scope=False), catalog)
    assert unsupported_answer(RouterDecision(companies=["AAPL"], years=["2023", "2024"]), catalog) is None


class FakeRetriever:
    def __init__(self, results):
        self.results = results

    def retrieve(self, bundle):
        return self.results


def run_retrieval(monkeypatch, catalog, decision, results_by_filter):
    # results_by_filter: {(company, year, section): [NodeWithScore, ...]}
    calls = []

    def fake_get_retriever(index, company, year, section, top_k=router.TOP_K_PER_COMBINATION, node_ids=None):
        calls.append((company, year, section) if node_ids is None else ("statements", company, year))
        key = (company, year, section) if node_ids is None else ("statements", company, year)
        return FakeRetriever(results_by_filter.get(key, []))

    monkeypatch.setattr(router, "get_sec_retriever", fake_get_retriever)
    index = SimpleNamespace(_embed_model=SimpleNamespace(get_query_embedding=lambda q: [0.0]))
    return router.retrieve_nodes("q", index, decision, catalog), calls


def test_retrieval_runs_one_search_per_combination_and_dedupes(monkeypatch, catalog):
    shared = make_node("same chunk", score=0.9, node_id="shared")
    results = {
        ("AAPL", "2024", "Item 8"): [shared, make_node("aapl only", score=0.5)],
        ("MSFT", "2024", "Item 8"): [shared, make_node("msft only", company="MSFT", score=0.7)],
    }
    decision = RouterDecision(companies=["AAPL", "MSFT"], years=["2024"], sections=["Item 8"])
    nodes, calls = run_retrieval(monkeypatch, catalog, decision, results)
    assert calls == [("AAPL", "2024", "Item 8"), ("MSFT", "2024", "Item 8")]
    assert [n.node.node_id for n in nodes].count("shared") == 1
    assert [n.score for n in nodes] == sorted([n.score for n in nodes], reverse=True)


def test_retrieval_interleaves_combinations_under_the_cap(monkeypatch, catalog):
    monkeypatch.setattr(router, "MAX_CONTEXT_CHUNKS", 4)
    results = {
        ("AAPL", "2024", None): [make_node(f"aapl {i}", score=0.9 - i / 100) for i in range(5)],
        ("MSFT", "2024", None): [make_node(f"msft {i}", company="MSFT", score=0.5 - i / 100) for i in range(5)],
    }
    decision = RouterDecision(companies=["AAPL", "MSFT"], years=["2024"])
    nodes, _ = run_retrieval(monkeypatch, catalog, decision, results)
    # Higher-scoring AAPL chunks must not crowd MSFT out of a comparison
    assert sorted(n.node.metadata["company"] for n in nodes) == ["AAPL", "AAPL", "MSFT", "MSFT"]


def test_retrieval_respects_the_character_cap(monkeypatch, catalog):
    monkeypatch.setattr(router, "MAX_CONTEXT_CHARS", 25)
    results = {("AAPL", None, None): [make_node("x" * 20, score=0.9), make_node("y" * 20, score=0.8)]}
    nodes, _ = run_retrieval(monkeypatch, catalog, RouterDecision(companies=["AAPL"]), results)
    assert len(nodes) == 1


def test_legal_questions_also_search_item_8(monkeypatch, catalog):
    decision = RouterDecision(companies=["MSFT"], years=["2024"], sections=["Item 3"])
    _, calls = run_retrieval(monkeypatch, catalog, decision, {})
    assert calls == [("MSFT", "2024", "Item 3"), ("MSFT", "2024", "Item 8")]


def test_unknown_values_are_not_used_as_filters(monkeypatch, catalog):
    decision = RouterDecision(companies=["TSLA"], sections=["Item 99"])
    _, calls = run_retrieval(monkeypatch, catalog, decision, {})
    assert calls == [(None, None, None)]


def test_item_8_searches_item_15_where_the_financial_statements_are(monkeypatch, catalog):
    catalog = IndexCatalog(companies=("AAPL", "NVDA"), years=("2025",), sections=catalog.sections,
                           financials_in_item_15=frozenset({("NVDA", "2025")}))
    decision = RouterDecision(companies=["AAPL", "NVDA"], years=["2025"], sections=["Item 8"])
    _, calls = run_retrieval(monkeypatch, catalog, decision, {})
    assert calls == [("AAPL", "2025", "Item 8"), ("NVDA", "2025", "Item 15")]


def test_at_most_three_companies_per_question(catalog):
    catalog = IndexCatalog(companies=("AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA"), years=("2024", "2025"),
                           sections=catalog.sections)
    assert unsupported_answer(RouterDecision(companies=["AAPL", "MSFT", "NVDA"]), catalog) is None
    assert "at most 3 companies" in unsupported_answer(RouterDecision(companies=["AAPL", "MSFT", "NVDA", "META"]), catalog)


def test_ticker_aliases_map_to_indexed_tickers():
    assert RouterDecision(companies=["goog", "FB"]).companies == ["GOOGL", "META"]


def test_statement_captions_exclude_notes():
    for caption in ["CONSOLIDATED STATEMENTS OF OPERATIONS / (In millions)", "Consolidated Statements of Income",
                    "ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA / INCOME STATEMENTS", "BALANCE SHEETS",
                    "CASH FLOWS STATEMENTS", "CONSOLIDATED STATEMENTS OF CASH FLOWS / (in millions)"]:
        assert router.is_statement_caption(caption), caption
    for caption in ["Note 9 - Balance Sheet Components", "Table of Contents / Consolidated Statements of Cash Flows "
                    "Reconciliation", "Note 2 – Revenue", "CONSOLIDATED STATEMENTS OF COMPREHENSIVE INCOME"]:
        assert not router.is_statement_caption(caption), caption


def test_financial_statements_are_searched_first_for_item_8(monkeypatch, catalog):
    catalog = IndexCatalog(companies=("GOOGL",), years=("2024",), sections=catalog.sections,
                           statements={("GOOGL", "2024"): ["income-statement-id"]})
    statement = make_node("Table: CONSOLIDATED STATEMENTS OF INCOME\nNet income | 100,118", company="GOOGL", score=0.4,
                          node_id="is")
    note = make_node("Note 1 text about net income", company="GOOGL", score=0.9, node_id="note")
    results = {("statements", "GOOGL", "2024"): [statement], ("GOOGL", "2024", "Item 8"): [note]}
    decision = RouterDecision(companies=["GOOGL"], years=["2024"], sections=["Item 8"])
    nodes, calls = run_retrieval(monkeypatch, catalog, decision, results)
    assert calls == [("GOOGL", "2024", "Item 8"), ("statements", "GOOGL", "2024")]
    assert {n.node.node_id for n in nodes} == {"is", "note"}


def test_page_header_tables_and_repeats_are_skipped(monkeypatch, catalog):
    assert router.is_empty_chunk("Table: Note 12. Net Income Per Share\nTable of Contents | Alphabet Inc.")
    assert not router.is_empty_chunk("Table: CONSOLIDATED STATEMENTS OF INCOME\nNet income | 100,118")
    header = make_node("Table: Note 12. Net Income Per Share\nTable of Contents | Alphabet Inc.", score=0.9, node_id="h")
    real = make_node("Net income | 100,118", score=0.5, node_id="a")
    repeat = make_node("Net income | 100,118", score=0.5, node_id="b")
    nodes, _ = run_retrieval(monkeypatch, catalog, RouterDecision(companies=["AAPL"]),
                             {("AAPL", None, None): [header, real, repeat]})
    assert [n.node.node_id for n in nodes] == ["a"]
