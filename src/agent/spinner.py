"""Loading spinner for the terminal."""
import sys
import threading
import time
from types import TracebackType
from typing import Optional, TextIO

FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
CLEAR_LINE = "\r\033[K"
HIDE_CURSOR, SHOW_CURSOR = "\033[?25l", "\033[?25h"


class Spinner:
    """Show ``⠋ text (12 s)`` on one line while a block runs.

    The line is erased when the block ends. Outside a terminal (pipe, log
    file) the text is printed once instead, without animation.
    """

    def __init__(self, text: str, stream: TextIO = sys.stderr,
                 interval: float = 0.1) -> None:
        """Prepare the spinner; it starts with the ``with`` block."""
        self.text = text
        self.stream = stream
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._animated = stream.isatty()

    def _run(self) -> None:
        """Thread body: redraw the line until stopped."""
        start = time.monotonic()
        frame = 0
        while not self._stop.is_set():
            elapsed = int(time.monotonic() - start)
            self.stream.write(f"{CLEAR_LINE}{FRAMES[frame % len(FRAMES)]} "
                              f"{self.text} ({elapsed} s)")
            self.stream.flush()
            frame += 1
            self._stop.wait(self.interval)

    def __enter__(self) -> "Spinner":
        """Start the animation."""
        if self._animated:
            self.stream.write(HIDE_CURSOR)
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        else:
            print(f"{self.text}...", file=self.stream)
        return self

    def __exit__(self, exc_type: Optional[type[BaseException]],
                 exc: Optional[BaseException],
                 tb: Optional[TracebackType]) -> None:
        """Stop the animation and erase its line (also on error/Ctrl+C)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self.stream.write(CLEAR_LINE + SHOW_CURSOR)
            self.stream.flush()
