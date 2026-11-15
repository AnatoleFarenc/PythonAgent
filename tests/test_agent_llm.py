"""LLMClient: error mapping, retries and key rotation (no network).

Replaces Agent-Smith's manual test_provider_errors.py script with real
assertions; the HTTP calls are mocked and sleeping is disabled.
"""
from src.agent.exceptions import (FatalError, KeyQuotaError,
                                  RequestTooLargeError, ServerError)
from src.agent.llm import LLMClient
from src.agent.providers import Provider, get_provider

from unittest.mock import MagicMock, patch
import httpx2 as httpx
import openai
import pytest

CREATE = "openai.resources.chat.completions.Completions.create"
REQUEST = httpx.Request("POST", "http://test/v1/chat/completions")

LOCAL = Provider("local", "http://test/v1", "served-model", None, 16384, 512)
REMOTE = Provider("remote", "http://test/v1", "m", "TEST_API_KEYS",
                  32768, 512)


def ok(text: str = "Thought: hi\nCode:\n```python\nprint(1)\n```"):
    usage = MagicMock(prompt_tokens=100, completion_tokens=20)
    return MagicMock(choices=[MagicMock(message=MagicMock(content=text))],
                     usage=usage)


def http_error(status: int, text: str = "error", headers=None):
    response = httpx.Response(status, text=text, headers=headers or {},
                              request=REQUEST)
    cls = {400: openai.BadRequestError, 401: openai.AuthenticationError,
           413: openai.APIStatusError, 429: openai.RateLimitError,
           500: openai.InternalServerError}[status]
    return cls(message=text, response=response, body=None)


@pytest.fixture
def sleeps():
    return []


def client(provider, sleeps) -> LLMClient:
    return LLMClient(provider, sleep=sleeps.append)


def test_success_returns_text_and_usage(sleeps):
    with patch(CREATE, return_value=ok()) as create:
        reply = client(LOCAL, sleeps).complete([])
    assert reply.text.startswith("Thought: hi")
    assert (reply.input_tokens, reply.output_tokens, reply.retries) == \
        (100, 20, 0)
    assert create.call_args.kwargs["model"] == "served-model"
    assert "Observation:" in create.call_args.kwargs["stop"]


def test_unclosed_fence_is_closed(sleeps):
    with patch(CREATE, return_value=ok("Code:\n```python\nprint(1)")):
        assert client(LOCAL, sleeps).complete([]).text.endswith("\n```")


def test_server_error_is_retried(sleeps):
    with patch(CREATE, side_effect=[http_error(500), ok()]):
        reply = client(LOCAL, sleeps).complete([])
    assert reply.retries == 1
    assert sleeps == [5.0]


def test_server_error_gives_up_after_retries(sleeps):
    with patch(CREATE, side_effect=[http_error(500)] * 10):
        with pytest.raises(ServerError):
            client(LOCAL, sleeps).complete([])


def test_unreachable_local_server_says_so(sleeps):
    error = openai.APIConnectionError(request=REQUEST)
    with patch(CREATE, side_effect=[error] * 10):
        with pytest.raises(ServerError, match="vllm serve"):
            client(LOCAL, sleeps).complete([])


@pytest.mark.parametrize("error", [
    http_error(413),
    http_error(400, "This model's maximum context length is 16384 tokens"),
])
def test_context_overflow_is_request_too_large(sleeps, error):
    with patch(CREATE, side_effect=[error]):
        with pytest.raises(RequestTooLargeError):
            client(LOCAL, sleeps).complete([])


def test_other_400_is_fatal(sleeps):
    with patch(CREATE, side_effect=[http_error(400, "unknown model")]):
        with pytest.raises(FatalError):
            client(LOCAL, sleeps).complete([])


def test_401_without_keys_is_fatal(sleeps):
    with patch(CREATE, side_effect=[http_error(401)]):
        with pytest.raises(FatalError):
            client(LOCAL, sleeps).complete([])


def test_rate_limit_rotates_key_and_waits_header(sleeps, monkeypatch):
    monkeypatch.setenv("TEST_API_KEYS", "k1, k2")
    llm = client(REMOTE, sleeps)
    assert llm._client.api_key == "k1"
    limited = http_error(429, headers={"retry-after": "10s"})
    with patch(CREATE, side_effect=[limited, ok()]):
        reply = llm.complete([])
    assert llm._client.api_key == "k2"
    assert reply.retries == 1
    assert sleeps == [12.0]


def test_all_keys_exhausted(sleeps, monkeypatch):
    monkeypatch.setenv("TEST_API_KEYS", "k1,k2")
    with patch(CREATE, side_effect=[http_error(401)] * 5):
        with pytest.raises(KeyQuotaError, match="exhausted"):
            client(REMOTE, sleeps).complete([])


def test_missing_keys_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("TEST_API_KEYS", raising=False)
    with pytest.raises(ValueError, match="TEST_API_KEYS"):
        LLMClient(REMOTE)


def test_local_model_is_asked_to_the_server(sleeps):
    provider = Provider("local", "http://test/v1", None, None, 16384, 512)
    models = MagicMock(data=[MagicMock(id="Qwen/Qwen2.5-Coder-7B")])
    with patch("openai.resources.models.Models.list", return_value=models):
        assert client(provider, sleeps).model == "Qwen/Qwen2.5-Coder-7B"


def test_get_provider():
    assert get_provider("groq", "other").model == "other"
    with pytest.raises(ValueError, match="unknown provider"):
        get_provider("nope")
