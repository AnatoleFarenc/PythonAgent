"""Command line entry point of the interactive sandbox."""
from .config import RED, RESET
from .repl import repl
from .api import Sandbox
from .errors import McpConnectionError
from ..models_agent.SandboxConfig import SandboxFile, load_config

from pathlib import Path
import argparse
import sys


def main() -> int:
    """Parse the arguments, start the sandbox and run the REPL.

    Returns:
        The exit code: 0 on success, 1 if the configuration or the MCP
        connection is invalid, 130 on KeyboardInterrupt.
    """
    parser = argparse.ArgumentParser(
        description="Launche the intercative sandbox"
    )
    parser.add_argument(
        "config",
        nargs='?',
        default=None,
        help="path to config file"
    )
    mcp_group = parser.add_mutually_exclusive_group()
    mcp_group.add_argument(
        "--mcp-stdio",
        type=str,
        default=None,
        help="Command to launch MCP server over stdio"
    )
    mcp_group.add_argument(
        "--mcp-server",
        type=str,
        default=None,
        help="Url of MCP server"
    )

    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Host directory mounted in /workspace (read-only by default)"
    )
    parser.add_argument(
        "--workspace-writable",
        action="store_true",
        help="Mount the workspace read-write"
    )

    args = parser.parse_args()

    settings = SandboxFile()
    if args.config is not None:
        try:
            settings = load_config(args.config)
        except (ValueError, RuntimeError) as e:
            print(f"{RED}Invalid configuration: {e}{RESET}", file=sys.stderr)
            return 1
    isolation = settings.isolation
    if args.workspace is not None:
        isolation = isolation.model_copy(update={"workspace": args.workspace})
    if args.workspace_writable:
        isolation = isolation.model_copy(update={"workspace_writable": True})
    mcp_connection = args.mcp_stdio or args.mcp_server

    try:
        sandbox = Sandbox(mcp_connection, settings.sandbox, True,
                          isolation=isolation)
    except ValueError as e:
        print(f"{RED}Invalid workspace: {e}{RESET}", file=sys.stderr)
        return 1
    except McpConnectionError as e:
        print(f"{RED}Cannot start the sandbox: {e}{RESET}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print()
        return 130

    try:
        with sandbox:
            repl(sandbox)
    except McpConnectionError as e:
        print(f"{RED}Cannot start the sandbox: {e}{RESET}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print()
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
