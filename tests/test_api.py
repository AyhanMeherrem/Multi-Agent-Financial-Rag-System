import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

import app.main as main
from app.llm_errors import LLMRateLimited, LLMUnavailable
from app.router.router_agent import IndexCatalog
from tests.helpers import make_node

HEADERS = {"X-Internal-Key": "test-key"}


def fake_answer_query(query, index):
    answer = "Net sales were 391,035 million [AAPL | FY2024 | Item 8]."
    filters = {"companies": ["AAPL"], "years": ["2024"], "year": "2024", "sections": ["Item 8"]}
    nodes = [make_node("Total net sales | 391,035", filing_url="https://www.sec.gov/Archives/x.htm")]
    return answer, filters, nodes


@pytest.fixture
def client(monkeypatch, catalog):
    # No index, model or Groq: the lifespan gets a placeholder index and the pipeline is stubbed
    monkeypatch.setattr(main, "load_index_from_qdrant", lambda: object())
    monkeypatch.setattr(main, "get_catalog", lambda index: catalog)
    monkeypatch.setattr(main, "answer_query", fake_answer_query)
    main.limiter.reset()
    with TestClient(main.app) as test_client:
        yield test_client


def test_health_needs_no_key(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "index_loaded": True}


def test_query_happy_path(client):
    response = client.post("/query", json={"query": "What were Apple's net sales in 2024?"}, headers=HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("Net sales were 391,035 million")
    assert body["companies"] == ["AAPL"] and body["year"] == "2024" and body["years"] == ["2024"]
    assert body["sources"] == [{"company": "AAPL", "year": "2024", "section": "Item 8",
                                "url": "https://www.sec.gov/Archives/x.htm", "snippet": "Total net sales | 391,035"}]
    assert "AAPL" in body["source_urls"]  # kept for backward compatibility


@pytest.mark.parametrize("headers", [{}, {"X-Internal-Key": "wrong"}])
def test_query_rejects_missing_or_wrong_key(client, headers):
    assert client.post("/query", json={"query": "q"}, headers=headers).status_code == 401


def test_query_fails_closed_without_configured_key(client, monkeypatch):
    monkeypatch.setattr(main, "BACKEND_API_KEY", None)
    assert client.post("/query", json={"query": "q"}, headers=HEADERS).status_code == 503


@pytest.mark.parametrize("query", ["", "x" * 501])
def test_query_length_is_validated(client, query):
    assert client.post("/query", json={"query": query}, headers=HEADERS).status_code == 422


@pytest.mark.parametrize("error, status", [(LLMRateLimited("429"), 429), (LLMUnavailable("down"), 503), (RuntimeError("boom"), 502)])
def test_llm_errors_map_to_clear_status_codes(client, monkeypatch, error, status):
    def failing(query, index):
        raise error
    monkeypatch.setattr(main, "answer_query", failing)
    response = client.post("/query", json={"query": "q"}, headers=HEADERS)
    assert response.status_code == status
    assert "boom" not in response.text  # internal errors are not leaked


def test_rate_limit_is_per_end_user(client):
    alice = {**HEADERS, "X-End-User-IP": "1.1.1.1"}
    codes = [client.post("/query", json={"query": "q"}, headers=alice).status_code for _ in range(11)]
    assert codes[:10] == [200] * 10 and codes[10] == 429
    bob = {**HEADERS, "X-End-User-IP": "2.2.2.2"}
    assert client.post("/query", json={"query": "q"}, headers=bob).status_code == 200


def make_request(headers: dict, client_host="10.0.0.5") -> Request:
    return Request({"type": "http", "client": (client_host, 1234),
                    "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()]})


def test_client_ip_uses_forwarded_ip_only_with_valid_key(monkeypatch):
    monkeypatch.setattr(main, "TRUST_PROXY_HEADERS", True)
    assert main.get_client_ip(make_request({"X-Internal-Key": "test-key", "X-End-User-IP": "1.2.3.4"})) == "1.2.3.4"
    assert main.get_client_ip(make_request({"X-End-User-IP": "1.2.3.4"})) == "10.0.0.5"
    assert main.get_client_ip(make_request({"X-Internal-Key": "test-key", "X-End-User-IP": "not-an-ip"})) == "10.0.0.5"


def test_client_ip_ignores_forwarded_ip_when_proxy_headers_are_off(monkeypatch):
    monkeypatch.setattr(main, "TRUST_PROXY_HEADERS", False)
    assert main.get_client_ip(make_request({"X-Internal-Key": "test-key", "X-End-User-IP": "1.2.3.4"})) == "10.0.0.5"


def test_repeated_question_is_served_from_cache(client, monkeypatch):
    calls = []

    def counting(query, index):
        calls.append(query)
        return fake_answer_query(query, index)
    monkeypatch.setattr(main, "answer_query", counting)
    first = client.post("/query", json={"query": "What were Apple's net sales in 2024?"}, headers=HEADERS).json()
    second = client.post("/query", json={"query": "what were apple's  net sales in 2024"}, headers=HEADERS).json()
    assert len(calls) == 1
    assert first["cached"] is False and second["cached"] is True
    assert {**second, "cached": False} == first


def test_errors_are_not_cached(client, monkeypatch):
    def failing(query, index):
        raise LLMRateLimited("429")
    monkeypatch.setattr(main, "answer_query", failing)
    assert client.post("/query", json={"query": "q"}, headers=HEADERS).status_code == 429
    monkeypatch.setattr(main, "answer_query", fake_answer_query)
    response = client.post("/query", json={"query": "q"}, headers=HEADERS)
    assert response.status_code == 200 and response.json()["cached"] is False


def test_catalog_lists_indexed_companies_and_years(client, monkeypatch, catalog):
    catalog = IndexCatalog(companies=("AAPL", "NVDA"), years=("2024", "2025"), sections=catalog.sections,
                           filings=(("AAPL", "2024"), ("AAPL", "2025"), ("NVDA", "2025")))
    monkeypatch.setattr(main, "get_catalog", lambda index: catalog)
    assert client.get("/catalog").status_code == 401
    body = client.get("/catalog", headers=HEADERS).json()
    assert body == {"companies": [{"ticker": "AAPL", "name": "Apple", "years": ["2024", "2025"]},
                                  {"ticker": "NVDA", "name": "NVIDIA", "years": ["2025"]}],
                    "max_companies_per_question": 3}


def test_daily_limit_counts_only_new_answers(client, monkeypatch):
    client.app.state.daily_limit.limit = 2
    for query in ["first", "first", "second"]:  # the repeated question comes from the cache
        assert client.post("/query", json={"query": query}, headers=HEADERS).status_code == 200
    response = client.post("/query", json={"query": "third"}, headers=HEADERS)
    assert response.status_code == 429 and "daily question limit" in response.json()["detail"]
    assert client.post("/query", json={"query": "first"}, headers=HEADERS).status_code == 200


def test_failed_requests_do_not_use_the_daily_limit(client, monkeypatch):
    client.app.state.daily_limit.limit = 1
    def failing(query, index):
        raise LLMUnavailable("down")
    monkeypatch.setattr(main, "answer_query", failing)
    assert client.post("/query", json={"query": "q"}, headers=HEADERS).status_code == 503
    monkeypatch.setattr(main, "answer_query", fake_answer_query)
    assert client.post("/query", json={"query": "q"}, headers=HEADERS).status_code == 200
