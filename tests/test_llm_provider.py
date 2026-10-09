from llama_index.llms.groq import Groq

from app.llm_provider import OPENROUTER_URL, get_llm


def test_groq_is_the_default(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test")
    llm = get_llm("openai/gpt-oss-20b", reasoning_effort="low", temperature=0.0, json_mode=True)
    assert isinstance(llm, Groq)
    assert llm.additional_kwargs == {"reasoning_effort": "low", "response_format": {"type": "json_object"}}


def test_openrouter_uses_the_same_model_names(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    llm = get_llm("openai/gpt-oss-120b", reasoning_effort="medium")
    assert llm.api_base == OPENROUTER_URL and llm.model == "openai/gpt-oss-120b"
    assert llm.additional_kwargs["extra_body"]["reasoning"] == {"effort": "medium"}
    assert "response_format" not in llm.additional_kwargs


def test_missing_key_is_reported(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    try:
        get_llm("openai/gpt-oss-120b", reasoning_effort="medium")
    except ValueError as e:
        assert "OPENROUTER_API_KEY" in str(e)
    else:
        raise AssertionError("expected a missing key error")
