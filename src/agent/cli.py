"""Chat with the coding assistant in the terminal.

    python -m src.agent                       # local vLLM, current folder
    python -m src.agent --provider groq       # remote API (keys in .env)
    python -m src.agent --root ../my_project --read-only

Run it from the pythonRag repository root (the MCP server and its index
cache are found relative to it). Commands: /reset, exit (or quit).

While the model, the sandbox and the code index load, only a spinner is
shown; the logs of the MCP server go to MCP_LOG_FILE.
"""
import argparse
import os
import shlex
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from ..models_agent.IsolationConfig import IsolationConfig
from ..models_agent.SandboxConfig import SandboxConfig
from ..sandbox.api import Sandbox
from ..sandbox.config import PROJECT_ROOT
from ..sandbox.errors import McpConnectionError
from .exceptions import ProviderRequestError, ServerError
from .llm import LLMClient
from .providers import PROVIDERS, Provider, get_provider
from .session import DEFAULT_MAX_STEPS, Session, Step
from .spinner import Spinner

DIM, RED, BOLD, RESET = "\033[2m", "\033[31m", "\033[1m", "\033[0m"
SHOWN_OBSERVATION_CHARS = 1200
# search_code may wait ~20 s for the index on the first call.
TOOL_TIMEOUT_SECONDS = 30
MCP_LOG_FILE = PROJECT_ROOT / "data" / "logs" / "mcp_server.log"
LOADING_TEXT = "Chargement du modèle et du RAG"
# Typed alone, these quit the chat (with or without the slash).
EXIT_COMMANDS = ("exit", "quit", "/exit", "/quit")
# How long to wait for `vllm serve` to finish loading its model.
MODEL_WAIT_SECONDS = 600
# Search run until the index answers (its result is not shown).
WARMUP_CODE = 'search_code("warmup", k=1)'
INDEX_LOADING_MARKER = "still being built"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        prog="python -m src.agent", description="Local coding assistant.")
    parser.add_argument("--provider", default="local",
                        choices=list(PROVIDERS),
                        help="local = vllm serve, others need API keys")
    parser.add_argument("--model", default=None,
                        help="override the provider's default model")
    parser.add_argument("--root", default=".",
                        help="project the assistant works on")
    parser.add_argument("--read-only", action="store_true",
                        help="no edit_file / create_file")
    parser.add_argument("--no-embedding", action="store_true",
                        help="BM25-only code search (no bge-m3)")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--unsafe-local-sandbox", action="store_true",
                        help="run the sandbox worker without Docker/gVisor")
    return parser.parse_args(argv)


def sandbox_factory(args: argparse.Namespace):
    """Return a function starting a sandbox wired to the code MCP server."""
    command = [sys.executable, "-m", "src.mcp_server", "--root", args.root,
               "--log-file", str(MCP_LOG_FILE)]
    if not args.read_only:
        command.append("--writable")
    if args.no_embedding:
        command.append("--no-embedding")
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
    isolation = IsolationConfig(
        backend="local" if args.unsafe_local_sandbox else "docker")
    config = SandboxConfig(max_tool_time_seconds=TOOL_TIMEOUT_SECONDS)

    def make() -> Sandbox:
        return Sandbox(shlex.join(command), config, mcp_env=env,
                       isolation=isolation)
    return make


def connect_llm(provider: Provider) -> LLMClient:
    """Create the client; for a local server, wait until it serves.

    ``make serve`` takes about a minute before answering: the assistant
    can be started at the same time and waits for it.

    Raises:
        ServerError: The local server still does not answer after
            MODEL_WAIT_SECONDS.
        ValueError / ProviderRequestError: Other configuration errors.
    """
    deadline = time.monotonic() + MODEL_WAIT_SECONDS
    while True:
        try:
            return LLMClient(provider)
        except ServerError:
            if (provider.api_key_env is not None
                    or time.monotonic() > deadline):
                raise
            time.sleep(1)


def wait_for_index(session: Session) -> str | None:
    """Run a search until the code index is built.

    Returns:
        None when the index works, otherwise the error to show (the
        assistant still works with grep / read_file).
    """
    while True:
        error = session.sandbox.execute(WARMUP_CODE).error
        if INDEX_LOADING_MARKER not in error:
            return error or None


def start_session(llm: LLMClient, provider: Provider,
                  args: argparse.Namespace) -> Session:
    """Start the sandbox (and its MCP server) and wait for the index."""
    session = Session(llm, sandbox_factory(args),
                      provider.context_window, provider.max_output_tokens,
                      args.max_steps)
    try:
        index_error = wait_for_index(session)
    except BaseException:
        session.close()
        raise
    if index_error:
        session.index_error = index_error
    return session


def thought_of(llm_output: str) -> str:
    """The text the model wrote before its code block."""
    head = llm_output.split("```", 1)[0]
    head = head.replace("Code:", "").replace("Thought:", "").strip()
    return head or "(no thought)"


def show_step(step: Step) -> None:
    """Print one step in dim text: thought, code, start of observation."""
    print(f"{DIM}[{step.number}] {thought_of(step.llm_output)}")
    if step.code:
        print(step.code)
    observation = step.observation
    if len(observation) > SHOWN_OBSERVATION_CHARS:
        observation = observation[:SHOWN_OBSERVATION_CHARS] + " [...]"
    print(f"→ {observation}{RESET}\n")


def chat(session: Session) -> None:
    """Read user messages until /exit or Ctrl+D."""
    while True:
        try:
            message = input(f"{BOLD}User > {RESET}").strip()
        except EOFError:
            print()
            return
        except KeyboardInterrupt:
            print()
            continue
        if not message:
            continue
        if message.lower() in EXIT_COMMANDS:
            return
        if message == "/reset":
            session.reset()
            print(f"{DIM}conversation cleared{RESET}")
            continue
        try:
            answer = session.ask(message, on_step=show_step)
        except KeyboardInterrupt:
            print(f"\n{RED}interrupted{RESET}")
            if session.sandbox_needs_restart:
                with Spinner("Redémarrage du sandbox"):
                    session.restart_sandbox()
                    wait_for_index(session)
            continue
        except ProviderRequestError as e:
            print(f"{RED}LLM error: {e}{RESET}")
            continue
        print(f"\n{answer.text}\n")


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    load_dotenv()
    args = parse_args(argv)
    if not Path(args.root).is_dir():
        print(f"{RED}--root is not a folder: {args.root}{RESET}",
              file=sys.stderr)
        return 1
    try:
        provider = get_provider(args.provider, args.model)
    except ValueError as e:
        print(f"{RED}{e}{RESET}", file=sys.stderr)
        return 1
    try:
        with Spinner(LOADING_TEXT):
            llm = connect_llm(provider)
            session = start_session(llm, provider, args)
    except KeyboardInterrupt:
        print()
        return 130
    except (ValueError, ProviderRequestError) as e:
        print(f"{RED}{e}{RESET}", file=sys.stderr)
        return 1
    except (McpConnectionError, RuntimeError) as e:
        print(f"{RED}cannot start the sandbox: {e}\n"
              f"see {MCP_LOG_FILE}{RESET}", file=sys.stderr)
        return 1
    print(f"{DIM}{provider.name}: {llm.model} — project {args.root}"
          f"{' (read-only)' if args.read_only else ''}{RESET}")
    if session.index_error:
        print(f"{RED}code search unavailable ({session.index_error}), "
              f"see {MCP_LOG_FILE}{RESET}")
    try:
        chat(session)
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
