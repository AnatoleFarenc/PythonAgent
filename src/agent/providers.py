"""LLM providers: the local vLLM server or a remote API.

vLLM (``vllm serve``) and the remote providers all speak the OpenAI chat
API, so one client works for every provider: only the URL, the model and
the key change. A provider without ``api_key_env`` is a local server and
needs no key.
"""

from dataclasses import dataclass, replace
import os


@dataclass(frozen=True)
class Provider:
    """Connection settings of one LLM provider.

    Attributes:
        name: Name given on the command line (--provider).
        base_url: OpenAI-compatible endpoint.
        model: Model id; None = ask the server (/v1/models), local only.
        api_key_env: Environment variable holding the comma-separated API
            keys, or None for a local server without key.
        context_window: Tokens the model accepts (prompt + answer); for
            vLLM it must match ``--max-model-len``.
        max_output_tokens: Tokens generated per answer at most.
    """

    name: str
    base_url: str
    model: str | None
    api_key_env: str | None
    context_window: int
    max_output_tokens: int


PROVIDERS: dict[str, Provider] = {
    "local": Provider(
        name="local",
        base_url=os.environ.get("LOCAL_LLM_URL", "http://localhost:8000/v1"),
        model=None,
        api_key_env=None,
        context_window=int(os.environ.get("LOCAL_LLM_CONTEXT", "16384")),
        max_output_tokens=2048,
    ),
    "groq": Provider(
        name="groq",
        base_url="https://api.groq.com/openai/v1/",
        model="qwen/qwen3.8-27b",
        api_key_env="GROQ_API_KEYS",
        context_window=131072,
        max_output_tokens=4096,
    ),
    "openrouter": Provider(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="openrouter/free",
        api_key_env="OPENROUTER_API_KEYS",
        context_window=32768,
        max_output_tokens=4096,
    ),
    "google": Provider(
        name="google",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model="gemini-3.5-flash-lite",
        api_key_env="GOOGLE_API_KEYS",
        context_window=131072,
        max_output_tokens=4096,
    ),
}


def get_provider(name: str, model: str | None = None) -> Provider:
    """Return provider *name*, with *model* overriding its default model.

    Raises:
        ValueError: If the provider is unknown.
    """
    try:
        provider = PROVIDERS[name]
    except KeyError:
        raise ValueError(
            f"unknown provider '{name}', choose one of: "
            f"{', '.join(PROVIDERS)}") from None
    return replace(provider, model=model) if model else provider
