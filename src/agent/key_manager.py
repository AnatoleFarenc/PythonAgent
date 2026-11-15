"""API key loading and rotation for LLM providers (from Agent-Smith)."""
import os


class KeyManager:
    """Manage the API keys of one provider and rotate them on quota errors.

    The keys come from one environment variable holding a comma-separated
    list, e.g. ``GROQ_API_KEYS="key1, key2"`` (usually set in ``.env``).
    """

    def __init__(self, env_var: str) -> None:
        """Load the keys of *env_var*.

        Raises:
            ValueError: If the variable is missing or holds no key.
        """
        self.env_var = env_var
        self.keys = self.load_api_keys()
        if not self.keys:
            raise ValueError(
                f"No API key found: set {env_var} in .env "
                "(comma-separated list of keys)")
        self.current_index = 0
        self.attempts = 0

    def load_api_keys(self) -> list[str]:
        """Load and sanitize the API keys from the environment variable."""
        return [k.strip() for k in os.environ.get(self.env_var, "").split(",")
                if k.strip()]

    def get_key(self) -> str:
        """Return the currently active API key."""
        return self.keys[self.current_index]

    def rotate(self) -> None:
        """Rotate to the next API key or raise if all are exhausted."""
        self.current_index = (self.current_index + 1) % len(self.keys)
        self.attempts += 1
        if self.attempts >= len(self.keys):
            raise ValueError("All keys have exceeded their quota")

    def reset_attempt(self) -> None:
        """Reset the consecutive failure attempts counter upon success."""
        self.attempts = 0
