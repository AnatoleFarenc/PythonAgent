from typing import Optional
from openai import (
    APIStatusError, APIConnectionError,
    RateLimitError, AuthenticationError,
    PermissionDeniedError, InternalServerError,
)


class OrchestratorError(Exception):
    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status

    @classmethod
    def from_exeption(cls, e: Exception) -> "OrchestratorError":
        if isinstance(e, OrchestratorError):
            return e

        if isinstance(e, APIConnectionError):
            return ServerError(f"Network/connection error: {e}")

        if isinstance(e, APIStatusError):
            status = e.status_code

            if status == 413:
                return RequestTooLargeError(
                    f"Request too large ({status}): {e}", status)

            # 429
            if isinstance(e, RateLimitError):
                return KeyQuotaError(
                    f"Rate limit exceeded ({status}): {e}", status,
                    retry_after=_retry_after(e))

            # 401 - 403 / 402
            if isinstance(e, (AuthenticationError, 
                              PermissionDeniedError)) or status == 402:
                return KeyQuotaError(
                    f"Invalid key or quota exceeded ({status}): {e}", status)

            if isinstance(e, InternalServerError):                 # 5xx
                return ServerError(f"Server error ({status}): {e}", status)

            return FatalError(
                f"Unrecoverable client error ({status}): {e.response.text}",
                status)

        return FatalError(f"Unexpected error: {e}")


class RequestTooLargeError(OrchestratorError):
    """RequestTooLargeError"""


class KeyQuotaError(OrchestratorError):
    def __init__(self, message: str, status: int | None = None,
                 retry_after: Optional[float] = None) -> None:
        super().__init__(message, status)
        self.retry_after = retry_after


class ServerError(OrchestratorError):
    """ServerError"""


class FatalError(OrchestratorError):
    """FatalError"""


def _retry_after(e: APIStatusError, default: float = 65.0) -> float:
    headers = e.response.headers
    raw = (headers.get("x-ratelimit-reset-tokens")
           or headers.get("retry-after"))
    if not raw:
        return default
    try:
        return max(parse_reset_header(raw) + 2.0, 5.0)
    except ValueError:
        return default


def parse_reset_header(raw: object) -> float:
    """Parse ratelimit reset strings (e.g. '1m26.4s', '42.9s') into seconds."""
    if not isinstance(raw, (str, int, float)):
        return 65.0
    s = str(raw).strip()
    try:
        if s.endswith("ms"):
            return float(s[:-2]) / 1000.0
        if "m" in s and s.endswith("s"):
            parts = s.rstrip("s").split("m")
            mins = float(parts[0])
            secs = float(parts[1]) if parts[1] else 0.0
            return mins * 60.0 + secs
        elif s.endswith("s"):
            return float(s[:-1])
        elif s.endswith("m"):
            return float(s[:-1]) * 60.0
        return float(s)
    except Exception:
        return 65.0
