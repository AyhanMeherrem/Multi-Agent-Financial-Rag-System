# Groq is called through LlamaIndex's OpenAI-compatible client. The Groq LLM objects use
# max_retries=1: one quick retry rides out a short per-minute limit (the client waits for Groq's
# retry-after when it is under a minute), while a daily limit fails at once instead of keeping the
# visitor waiting for minutes. These app-level errors let the API answer with a clear 429 or 503
# instead of a generic failure.
import logging

import openai

logger = logging.getLogger("uvicorn.error")


class LLMRateLimited(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        # Groq names the limit that was hit, e.g. "... on tokens per day (TPD)". A daily limit does
        # not clear in a minute, so the API tells the visitor to come back later instead.
        self.daily = "per day" in message


class LLMUnavailable(Exception):
    pass


def log_usage(llm, response) -> None:
    # Real token counts per call, to size the per-minute and per-day Groq limits
    raw = getattr(response, "raw", None)
    usage = raw.get("usage") if isinstance(raw, dict) else getattr(raw, "usage", None)
    if usage is None:
        return
    def get(key):
        return usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
    logger.info("%s: %s prompt + %s completion tokens", getattr(llm, "model", "llm"),
                get("prompt_tokens"), get("completion_tokens"))


def chat(llm, messages):
    try:
        response = llm.chat(messages)
    except openai.RateLimitError as e:
        raise LLMRateLimited(str(e)) from e
    except (openai.APIConnectionError, openai.InternalServerError) as e:  # includes timeouts and 5xx
        raise LLMUnavailable(str(e)) from e
    except openai.BadRequestError as e:
        # Groq answers every request with this 400 once the organization's monthly spend limit is reached
        body = e.body if isinstance(e.body, dict) else {}
        if "blocked_api_access" in (e.code, (body.get("error") or {}).get("code")):
            raise LLMUnavailable(str(e)) from e
        raise
    except openai.APIStatusError as e:
        # OpenRouter: 402 when the credits run out, 403 when the key's own spending limit is reached
        if e.status_code in (402, 403):
            raise LLMUnavailable(str(e)) from e
        raise
    log_usage(llm, response)
    return response
