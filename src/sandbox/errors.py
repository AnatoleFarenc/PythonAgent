"""Exceptions of the sandbox, kept apart so the worker does not need mcp."""


class McpToolError(Exception):
    """Raised when an MCP tool reports an error."""


class McpConnectionError(Exception):
    """Raised when the MCP server cannot be reached or initialized."""
