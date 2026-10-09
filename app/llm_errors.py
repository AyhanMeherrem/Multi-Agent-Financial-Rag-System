# Groq is called through LlamaIndex's OpenAI-compatible client, which already retries 429s, 5xx
# responses and connection errors with exponential backoff (max_retries on the Groq LLM objects).
# When the retries are used up, these app-level errors let the API answer with a clear 429 or 503
# instead of a generic failure.
import openai


class LLMRateLimited(Exception):
    pass


class LLMUnavailable(Exception):
    pass


def chat(llm, messages):
    try:
        return llm.chat(messages)
    except openai.RateLimitError as e:
        raise LLMRateLimited(str(e)) from e
    except (openai.APIConnectionError, openai.InternalServerError) as e:  # includes timeouts and 5xx
        raise LLMUnavailable(str(e)) from e
