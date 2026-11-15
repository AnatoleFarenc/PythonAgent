"""Chat client for any OpenAI-compatible provider (local vLLM or API).

Taken from Agent-Smith's ``Orchestrator.call_api`` / ``query_llm``, moved
out of the orchestrator so that the agent loop only sees ``complete()``.
"""
import time
from dataclasses import dataclass
from typing import Callable, Sequence

from openai import (OpenAI, APIStatusError,
                    APIConnectionError, APITimeoutError)
from openai.types.chat import ChatCompletionMessageParam

from .exceptions import (KeyQuotaError, ServerError, FatalError,
                         RequestTooLargeError)
from .key_manager import KeyManager
from .parse_header import parse_reset_header
from .providers import Provider

# Generation stops before the model invents the sandbox's answer or starts
# a second step on its own.
STOP_SEQUENCES = [
    "Observation:",
    "Observation\n",
    "```Thought",
    "```\nThought",
]

# vLLM answers 400 (not 413) when the prompt does not fit the context.
CONTEXT_ERROR_MARKERS = ("maximum context length", "context length",
                         "too many tokens", "prompt is too long")

MAX_SERVER_RETRIES = 3
SERVER_RETRY_DELAY = 5.0
DEFAULT_RATE_LIMIT_WAIT = 65.0
REQUEST_TIMEOUT = 300.0


@dataclass(slots=True)
class LLMReply:
    """One answer of the model and what it cost.

    Attributes:
        text: Generated text.
        input_tokens: Prompt tokens reported by the provider (0 if unknown).
        output_tokens: Generated tokens reported by the provider.
        time_ms: Duration of the successful request.
        retries: Failed attempts before it (quota, server errors).
    """

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    time_ms: float = 0.0
    retries: int = 0


class LLMClient:
    """Send chat requests with retries and, for APIs, key rotation.

    Attributes:
        provider: Provider settings.
        model: Model id actually used.
    """

    def __init__(self, provider: Provider, temperature: float = 0.2,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        """Create the client; for a local server, find the served model.

        Args:
            provider: Where to send the requests.
            temperature: Sampling temperature (low = more deterministic).
            sleep: Waiting function, replaced in the tests.

        Raises:
            ValueError: If the provider needs keys and none is set.
            ServerError: If the local server cannot be reached to find
                its model.
        """
        self.provider = provider
        self.temperature = temperature
        self._sleep = sleep
        self._keys = (KeyManager(provider.api_key_env)
                      if provider.api_key_env else None)
        self._client = self._make_client()
        self.model = provider.model or self._served_model()

    def _make_client(self) -> OpenAI:
        """OpenAI client for the current key.

        ``max_retries=0``: the SDK would otherwise retry 429/5xx by itself,
        hiding them from the rotation and retry logic below.
        """
        key = self._keys.get_key() if self._keys else "EMPTY"
        return OpenAI(base_url=self.provider.base_url, api_key=key,
                      max_retries=0, timeout=REQUEST_TIMEOUT)

    def _served_model(self) -> str:
        """Ask the server which model it serves (vLLM serves exactly one)."""
        try:
            models = self._client.models.list().data
        except (APIConnectionError, APITimeoutError) as e:
            raise ServerError(self._unreachable_message()) from e
        if not models:
            raise FatalError(f"{self.provider.base_url} serves no model")
        return models[0].id

    def _unreachable_message(self) -> str:
        """Error text for a server that does not answer."""
        text = f"cannot reach {self.provider.base_url}"
        if self.provider.api_key_env is None:
            text += " — is `vllm serve` running?"
        return text

    def call_api(self, messages: Sequence[ChatCompletionMessageParam]
                 ) -> LLMReply:
        """Send one request and map API errors to domain exceptions.

        Raises:
            RequestTooLargeError: The prompt does not fit: shrink it.
            KeyQuotaError: Quota or rate limit of the current key.
            ServerError: 5xx or network error, worth retrying.
            FatalError: Anything else (bad model name, bad request...).
        """
        try:
            start_time = time.perf_counter()
            response = self._client.chat.completions.create(
                model=self.model,
                messages=list(messages),
                stop=STOP_SEQUENCES,
                max_tokens=self.provider.max_output_tokens,
                temperature=self.temperature,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
        except APIStatusError as e:
            raise self._map_status_error(e) from e
        except (APIConnectionError, APITimeoutError) as e:
            raise ServerError(self._unreachable_message()) from e
        except Exception as e:
            raise FatalError(f"Unexpected error: {e}") from e

        content = response.choices[0].message.content or ""
        # The stop sequence "```Thought" eats the closing fence of the
        # code block: put it back when the number of fences is odd.
        if content.count("```") % 2 == 1:
            content = f"{content.rstrip()}\n```"
        usage = response.usage
        return LLMReply(
            text=content,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            time_ms=elapsed_ms,
        )

    def _map_status_error(self, e: APIStatusError) -> Exception:
        """Turn an HTTP error into the exception the retry logic expects."""
        status = e.status_code
        body = getattr(e.response, "text", "") or str(e)
        if status == 413 or (status == 400 and any(
                m in body.lower() for m in CONTEXT_ERROR_MARKERS)):
            # Waiting cannot make it pass: the context must shrink.
            return RequestTooLargeError(f"Request too large ({status}): {e}")
        if status == 429:
            self._sleep(self._rate_limit_wait(e))
            return KeyQuotaError(f"Rate limit exceeded ({status}): {e}")
        if status in (401, 402, 403):
            if self._keys is None:
                return FatalError(
                    f"The server refused the request ({status}): {body}")
            return KeyQuotaError(f"Quota or invalid key ({status}): {e}")
        if 500 <= status < 600:
            return ServerError(f"Server error ({status}): {e}")
        return FatalError(f"Unrecoverable client error ({status}): {body}")

    @staticmethod
    def _rate_limit_wait(e: APIStatusError) -> float:
        """Seconds to wait after a 429, read from the provider headers."""
        try:
            headers = e.response.headers
            raw = (headers.get("x-ratelimit-reset-tokens")
                   or headers.get("retry-after"))
            if raw:
                return max(parse_reset_header(raw) + 2.0, 5.0)
        except Exception:
            pass
        return DEFAULT_RATE_LIMIT_WAIT

    def complete(self, messages: Sequence[ChatCompletionMessageParam]
                 ) -> LLMReply:
        """Get one answer, retrying server errors and rotating keys.

        Raises:
            RequestTooLargeError: The caller must shrink the context.
            KeyQuotaError: Every key is exhausted (or the local server
                keeps rate limiting).
            ServerError: Still failing after MAX_SERVER_RETRIES retries.
            FatalError: Unrecoverable error.
        """
        retries = 0
        server_retries = 0
        while True:
            try:
                reply = self.call_api(messages)
            except KeyQuotaError:
                retries += 1
                if self._keys is None:
                    if retries > MAX_SERVER_RETRIES:
                        raise
                    continue
                try:
                    self._keys.rotate()
                except ValueError as quota_err:
                    raise KeyQuotaError(
                        f"All {self.provider.name} API keys have been "
                        "exhausted.") from quota_err
                self._client = self._make_client()
            except ServerError:
                server_retries += 1
                retries += 1
                if server_retries > MAX_SERVER_RETRIES:
                    raise
                self._sleep(SERVER_RETRY_DELAY)
            else:
                if self._keys is not None:
                    self._keys.reset_attempt()
                reply.retries = retries
                return reply
