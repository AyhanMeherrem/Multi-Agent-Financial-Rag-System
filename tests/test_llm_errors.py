import httpx
import openai
import pytest

from app.llm_errors import LLMRateLimited, LLMUnavailable, chat


def api_error(cls, status, code):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return cls("error", response=response, body={"error": {"code": code, "message": "error"}})


class FailingLLM:
    def __init__(self, error):
        self.error = error

    def chat(self, messages):
        raise self.error


@pytest.mark.parametrize("error, expected", [
    (api_error(openai.RateLimitError, 429, "rate_limit_exceeded"), LLMRateLimited),
    (api_error(openai.InternalServerError, 500, None), LLMUnavailable),
    (api_error(openai.BadRequestError, 400, "blocked_api_access"), LLMUnavailable),  # spend limit reached
    (api_error(openai.BadRequestError, 400, "invalid_request"), openai.BadRequestError),
])
def test_groq_errors_are_mapped(error, expected):
    with pytest.raises(expected):
        chat(FailingLLM(error), [])
