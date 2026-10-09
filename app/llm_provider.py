# Builds the chat LLM clients. Groq and OpenRouter serve the same open-weight gpt-oss models behind
# an OpenAI-compatible API, so switching between them (LLM_PROVIDER in .env) changes only the
# endpoint, the key and how a few request options are passed, not the prompts or the models.
import os

from llama_index.llms.groq import Groq
from llama_index.llms.openai_like import OpenAILike

OPENROUTER_URL = "https://openrouter.ai/api/v1"


def llm_provider() -> str:
    return os.getenv("LLM_PROVIDER", "groq").strip().lower()


def require_key(name: str) -> str:
    key = os.getenv(name)
    if not key or key.startswith("your_"):
        raise ValueError(f"{name} is missing or unconfigured in .env file!")
    return key


def get_llm(model: str, reasoning_effort: str, temperature: float | None = None, json_mode: bool = False) -> OpenAILike:
    # max_retries=1: one quick retry rides out a short per-minute limit, see app/llm_errors.py
    options = {"max_retries": 1, "timeout": 60.0}
    if temperature is not None:
        options["temperature"] = temperature
    json_format = {"response_format": {"type": "json_object"}} if json_mode else {}

    if llm_provider() == "openrouter":
        # OpenRouter takes the reasoning effort as a "reasoning" object and routes each request to one
        # of several hosts; require_parameters keeps it to hosts that support every option sent here
        # (JSON mode for the router). Options the OpenAI client does not know go in extra_body.
        extra_body = {"reasoning": {"effort": reasoning_effort}, "provider": {"require_parameters": True}}
        return OpenAILike(model=model, api_base=OPENROUTER_URL, api_key=require_key("OPENROUTER_API_KEY"),
                          is_chat_model=True, context_window=131072,
                          additional_kwargs={"extra_body": extra_body, **json_format}, **options)

    return Groq(model=model, api_key=require_key("GROQ_API_KEY"),
                additional_kwargs={"reasoning_effort": reasoning_effort, **json_format}, **options)
