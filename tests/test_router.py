import json
from types import SimpleNamespace

import app.router.router_agent as router
from app.router.router_agent import RouterDecision, extract_filters, filing_years_for, unsupported_answer
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
    assert "AAPL, MSFT" in prompt and "2024, 2025" in prompt
    assert '"Item 9A"' not in prompt  # not in this catalog


def test_prior_years_map_to_filings_that_contain_them(catalog):
    assert filing_years_for("2024", catalog) == ["2024"]
    assert filing_years_for("2023", catalog) == ["2024", "2025"]
    assert filing_years_for("2021", catalog) == []


def test_unsupported_answer(catalog):
    assert "do not include GOOGL" in unsupported_answer(RouterDecision(companies=["GOOGL"]), catalog)
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

    def fake_get_retriever(index, company, year, section, top_k=router.TOP_K_PER_COMBINATION):
        calls.append((company, year, section))
        return FakeRetriever(results_by_filter.get((company, year, section), []))

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
    decision = RouterDecision(companies=["GOOGL"], sections=["Item 99"])
    _, calls = run_retrieval(monkeypatch, catalog, decision, {})
    assert calls == [(None, None, None)]
