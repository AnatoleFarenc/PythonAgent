"""Minimal interactive prompt: type a question, press Enter, get the answer.

Run with ``python -m src.interface``. Quit with ``exit``, Ctrl-D, Ctrl-C
or SIGTERM. Library logs and progress bars are sent to
``INTERFACE_LOG_FILE`` (or discarded when it is empty); the terminal only
shows a spinner, the prompt and the answers.
"""

import itertools
import multiprocessing
import os
import readline  # noqa: F401  (arrow keys + history in input())
import shutil
import signal
import sys
import threading
from contextlib import contextmanager
from types import FrameType
from typing import Generator, Optional

from .config import AMBEDING_MODEL, TEST_RECALL_FOLDER, INTERFACE_LOG_FILE

COLOR = sys.stdout.isatty()
DIM = "\033[2m" if COLOR else ""
CYAN = "\033[36m" if COLOR else ""
RESET = "\033[0m" if COLOR else ""

LOG_PATH = INTERFACE_LOG_FILE or os.devnull


class Spinner:
    """Spinner drawn on *term* (a file descriptor) from a thread."""

    def __init__(self, term: int, message: str) -> None:
        """Start the animation thread.

        Args:
            term: File descriptor of the real terminal.
            message: Text shown next to the spinner.
        """
        self.term = term
        self.message = message
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def _spin(self) -> None:
        for frame in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
            line = f"\r{CYAN}{frame}{RESET} {self.message}"
            os.write(self.term, line.encode())
            if self.done.wait(0.1):
                break
        os.write(self.term, b"\r\033[K" if COLOR else b"\n")

    def stop(self) -> None:
        """Stop the animation and erase the spinner line."""
        self.done.set()
        self.thread.join()


current: Optional[Spinner] = None


@contextmanager
def quiet(message: str) -> Generator[None]:
    """Show a spinner while sending everything printed to ``LOG_PATH``.

    File descriptors 1 and 2 are redirected (not just ``sys.stdout``), so
    output from C code and from vLLM's engine subprocess is hidden too.
    """
    global current
    if os.path.dirname(LOG_PATH):
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    term = os.dup(1)
    saved_err = os.dup(2)
    log = os.open(LOG_PATH, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    os.dup2(log, 1)
    os.dup2(log, 2)
    current = Spinner(term, message)
    try:
        yield
    finally:
        current.stop()
        current = None
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(term, 1)
        os.dup2(saved_err, 2)
        for fd in (term, saved_err, log):
            os.close(fd)


def prompt() -> str:
    """Draw a separator line, then read one line from the user."""
    width = shutil.get_terminal_size().columns
    print(f"{DIM}{'─' * width}{RESET}")
    return input(f"\001{CYAN}\002> \001{RESET}\002").strip()


def stop_children() -> None:
    """Kill child processes (vLLM's engine) so the exit never hangs."""
    for child in multiprocessing.active_children():
        child.kill()


def on_signal(signum: int, frame: Optional[FrameType]) -> None:
    """Exit cleanly on Ctrl-C (SIGINT), SIGTERM or SIGHUP.

    vLLM's engine runs in a child process: it is always killed first.
    While loading or generating, vLLM can swallow exceptions raised here
    (it retries its socket waits), so the process leaves at once with
    ``os._exit``. At the prompt, ``SystemExit`` lets readline restore the
    terminal before exiting.
    """
    stop_children()
    code = 128 + signum
    if current is not None:
        current.stop()
        os.write(current.term, f"{DIM}Interupt.{RESET}\n".encode())
        os._exit(code)
    print(f"\n{DIM}Interupt.{RESET}")
    raise SystemExit(code)


def main() -> None:
    """Load the index and the LLM, then answer questions until exit."""
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, on_signal)

    with quiet("Load index and model..."):
        from .generator import RagGenerator
        from .indexer.HybridIndex import HybridIndex

        index = HybridIndex.from_folder(TEST_RECALL_FOLDER, AMBEDING_MODEL)
        generator = RagGenerator(index)

    while True:
        try:
            query = prompt()
        except EOFError:
            print()
            break
        if query in ("exit", "quit"):
            break
        if not query:
            continue
        with quiet("Reflection..."):
            answer = generator.answer(query, k=5)
        print(f"\n{answer.text}\n")


if __name__ == "__main__":
    main()
