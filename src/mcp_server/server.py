"""MCP server exposing code tools on one project to a coding agent.

Tools:
    search_code  -- semantic + keyword search (the RAG HybridIndex)
    grep         -- exact regex search
    read_file    -- numbered lines of a file, by range
    list_files   -- folder content / glob
  with --writable only:
    edit_file    -- replace a unique string, returns the diff
    create_file  -- create a new file (never overwrites)

Run (stdio, what McpBridge launches):
    python -m src.mcp_server --root path/to/project
Run over HTTP:
    python -m src.mcp_server --root . --transport streamable-http --port 8765

Launch it from the repository root: the index cache lives in ``data/chunk``
relative to the current directory.

Without --writable nothing writes to the workspace. Writes are confined
to the root, never touch .git/.venv/..., and a .py file is only written
if it still parses. Running code is left to the sandbox.
"""

import argparse
import asyncio
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

from ..rag.config import AMBEDING_MODEL, USE_EMBEDDING
from .code_search import CodeSearch
from .config import DEFAULT_SEARCH_K, SERVER_NAME
from .workspace import Workspace

INSTRUCTIONS = (
    "Tools on the user's project. Paths are relative to the "
    "project root. Use search_code to find code by meaning, grep for an "
    "exact name, then read_file on the lines you need instead of whole "
    "files."
)

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False,
                        idempotent_hint=False, open_world_hint=False)


async def run_tool(fn: Callable[..., str], *args: Any) -> str:
    """Run a blocking tool body in a thread.

    Expected failures (bad path, bad regex, index not ready...) are raised
    as ToolError so that the model receives the message and can fix its
    call; any other exception is a crash and MCPServer hides its text.
    """
    try:
        return await asyncio.to_thread(fn, *args)
    except (ValueError, RuntimeError) as e:
        raise ToolError(str(e)) from e


def build_server(workspace: Workspace, search: CodeSearch,
                 writable: bool = False) -> MCPServer:
    """Create the MCP server and register the tools.

    edit_file and create_file are only registered when *writable*: a
    read-only server does not even show them to the model.

    The index starts building when the server starts (lifespan); its
    prints are redirected to stderr by CodeSearch.
    """

    @asynccontextmanager
    async def lifespan(_: MCPServer) -> AsyncIterator[None]:
        search.start()
        yield

    server = MCPServer(SERVER_NAME, instructions=INSTRUCTIONS,
                       lifespan=lifespan)

    @server.tool(annotations=READ_ONLY)
    async def search_code(query: str, k: int = DEFAULT_SEARCH_K) -> str:
        """Find the Python code most related to a natural-language query.

        Use it when you don't know the exact name (e.g. "where are retries
        handled"). Returns up to k chunks (max 10) with file:lines.

        Args:
            query: What you are looking for, in plain words or identifiers.
            k: Number of chunks to return.
        """
        return await run_tool(search.search, query, k)

    @server.tool(annotations=READ_ONLY)
    async def grep(pattern: str, path: str = ".", glob: str = "*.py",
                   ignore_case: bool = False) -> str:
        """Search a regular expression in files, returns file:line: text.

        Use it for exact names (function, class, variable, error message).

        Args:
            pattern: Python regular expression, e.g. "def load_config".
            path: File or folder to search, relative to the project root.
            glob: File name filter, e.g. "*.py", "*.toml", "*".
            ignore_case: Case-insensitive search.
        """
        return await run_tool(
            workspace.grep, pattern, path, glob, ignore_case)

    @server.tool(annotations=READ_ONLY)
    async def read_file(path: str, start_line: int = 1,
                        end_line: int | None = None) -> str:
        """Read lines of a text file, each prefixed by its line number.

        Returns at most 400 lines; read big files by range.

        Args:
            path: File path relative to the project root.
            start_line: First line to read (1-based).
            end_line: Last line to read (inclusive); omit to read to the end.
        """
        return await run_tool(
            workspace.read_file, path, start_line, end_line)

    @server.tool(annotations=READ_ONLY)
    async def list_files(path: str = ".", pattern: str = "*") -> str:
        """List files and folders (folders end with /).

        Args:
            path: Folder relative to the project root.
            pattern: Glob: "*" for one level, "**/*.py" for recursive.
        """
        return await run_tool(workspace.list_files, path, pattern)

    if not writable:
        return server

    def edit_and_reindex(path: str, old_str: str, new_str: str) -> str:
        result = workspace.edit_file(path, old_str, new_str)
        search.notify_changed(workspace.resolve(path))
        return result

    def create_and_reindex(path: str, content: str) -> str:
        result = workspace.create_file(path, content)
        search.notify_changed(workspace.resolve(path))
        return result

    @server.tool(annotations=WRITE)
    async def edit_file(path: str, old_str: str, new_str: str) -> str:
        """Replace one exact, unique piece of text in a file.

        old_str must be copied from read_file WITHOUT the line numbers and
        appear exactly once (add surrounding lines if needed). Returns the
        diff. A Python file that would not parse is not written.

        Args:
            path: File path relative to the project root.
            old_str: Exact text to replace.
            new_str: Replacement text.
        """
        return await run_tool(edit_and_reindex, path, old_str, new_str)

    @server.tool(annotations=WRITE)
    async def create_file(path: str, content: str) -> str:
        """Create a new file with this content (fails if it exists).

        Args:
            path: New file path relative to the project root.
            content: Full content of the file.
        """
        return await run_tool(create_and_reindex, path, content)

    return server


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m src.mcp_server", description="Code MCP server.")
    parser.add_argument("--root", default=".",
                        help="project folder exposed to the agent")
    parser.add_argument("--transport", default="stdio",
                        choices=["stdio", "streamable-http"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-embedding", action="store_true",
                        help="BM25 only: no bge-m3, no GPU, weaker search")
    parser.add_argument("--writable", action="store_true",
                        help="add edit_file and create_file")
    parser.add_argument("--log-file", default=None,
                        help="append the server's logs (stderr) to this "
                             "file instead of the terminal")
    return parser.parse_args(argv)


def redirect_stderr(path: str) -> None:
    """Send everything written on stderr (fd 2) to *path*.

    Done at the file descriptor level, so the logs of the libraries
    (sentence-transformers, huggingface, the indexer) go there too. The
    MCP protocol uses stdin/stdout only and is not affected.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    log = open(path, "a", buffering=1, encoding="utf-8")
    os.dup2(log.fileno(), 2)


def main(argv: list[str] | None = None) -> None:
    """Entry point: build the workspace and run the server."""
    args = parse_args(argv)
    if args.log_file:
        redirect_stderr(args.log_file)
    workspace = Workspace(args.root)
    search = CodeSearch(
        workspace, args.root, AMBEDING_MODEL,
        use_embedding=USE_EMBEDDING and not args.no_embedding)
    server = build_server(workspace, search, args.writable)
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
