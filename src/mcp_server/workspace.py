"""Read-only access to the files of one project folder.

Every path received from the model is resolved (symlinks included) and
must stay inside the workspace root: ``../``, absolute paths or a symlink
pointing outside are refused. Errors are raised as ``ValueError`` with a
message written for the model, the MCP server returns it as a tool error.
"""

import ast
import difflib
import fnmatch
import os
import re
from pathlib import Path
from typing import Iterator, List

from .config import (
    IGNORED_DIRS, MAX_DIFF_LINES, MAX_GREP_FILE_BYTES, MAX_GREP_MATCHES,
    MAX_LINE_CHARS, MAX_LIST_ENTRIES, MAX_OUTPUT_CHARS, MAX_READ_LINES,
)


def truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Cut *text* to *limit* characters, saying that it was cut."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[output truncated at {limit} characters]"


def clip_line(line: str, limit: int = MAX_LINE_CHARS) -> str:
    """Cut one line to *limit* characters (minified files, data blobs)."""
    return line if len(line) <= limit else line[:limit] + " [...]"


class Workspace:
    """A project folder the model may read.

    Attributes:
        root: Absolute, resolved path of the folder.
    """

    def __init__(self, root: str | Path) -> None:
        """Check that *root* is an existing folder.

        Raises:
            NotADirectoryError: If *root* is not a folder.
        """
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(f"workspace root is not a folder: {root}")

    def resolve(self, path: str) -> Path:
        """Turn a path given by the model into a safe absolute path.

        Args:
            path: Path relative to the root ("" or "." = the root itself).

        Raises:
            ValueError: If the path leaves the workspace or does not exist.
        """
        target = (self.root / path).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError(f"'{path}' is outside the workspace")
        if not target.exists():
            raise ValueError(f"'{path}' does not exist")
        return target

    def relative(self, path: str | Path) -> str:
        """Path relative to the root, as shown to the model."""
        resolved = Path(path).resolve()
        if resolved.is_relative_to(self.root):
            return resolved.relative_to(self.root).as_posix() or "."
        return str(path)

    def _is_ignored(self, path: Path) -> bool:
        """True if *path* is inside an ignored folder (.venv, .git...)."""
        parts = path.relative_to(self.root).parts
        return bool(IGNORED_DIRS.intersection(parts))

    def read_file(
            self, path: str, start_line: int = 1,
            end_line: int | None = None) -> str:
        """Return lines *start_line*..*end_line* (1-based, inclusive).

        Lines are prefixed with their number. At most MAX_READ_LINES lines
        are returned; a footer tells how to read the next part.

        Raises:
            ValueError: If the path is invalid, not a text file, or the
                line range is empty.
        """
        target = self.resolve(path)
        if not target.is_file():
            raise ValueError(f"'{path}' is a folder, use list_files")
        try:
            lines = target.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError as e:
            raise ValueError(f"'{path}' is not a UTF-8 text file") from e

        total = len(lines)
        if total == 0:
            return f"[{self.relative(target)} is empty]"
        start = max(start_line, 1)
        end = total if end_line is None else min(end_line, total)
        end = min(end, start + MAX_READ_LINES - 1)
        if start > end:
            raise ValueError(
                f"empty line range {start_line}-{end_line}: "
                f"the file has {total} lines")

        width = len(str(end))
        body = "\n".join(
            f"{n:>{width}}| {clip_line(lines[n - 1])}"
            for n in range(start, end + 1))
        header = f"{self.relative(target)} (lines {start}-{end} of {total})"
        footer = ""
        if end < total:
            footer = (f"\n[{total - end} more lines: call again with "
                      f"start_line={end + 1}]")
        return truncate(f"{header}\n{body}{footer}")

    def list_files(self, path: str = ".", pattern: str = "*") -> str:
        """List entries of folder *path* matching the glob *pattern*.

        ``*`` lists one level, ``**/*.py`` searches recursively. Folders
        end with ``/``. Ignored folders (.venv, .git, caches) are hidden.

        Raises:
            ValueError: If the path is invalid or not a folder.
        """
        folder = self.resolve(path)
        if not folder.is_dir():
            raise ValueError(f"'{path}' is a file, use read_file")

        entries: List[str] = []
        more = False
        for entry in sorted(folder.glob(pattern)):
            resolved = entry.resolve()
            if (not resolved.is_relative_to(self.root)
                    or self._is_ignored(entry)):
                continue
            if len(entries) == MAX_LIST_ENTRIES:
                more = True
                break
            suffix = "/" if entry.is_dir() else ""
            entries.append(entry.relative_to(self.root).as_posix() + suffix)

        if not entries:
            return f"[no entry matches '{pattern}' in {self.relative(folder)}]"
        if more:
            entries.append(
                f"[more than {MAX_LIST_ENTRIES} entries: "
                "narrow the path or the pattern]")
        return "\n".join(entries)

    def _walk_files(self, folder: Path, glob: str) -> Iterator[Path]:
        """Yield files under *folder* whose name matches *glob*, sorted,
        without entering ignored folders or following symlinked folders."""
        for current, dirs, files in os.walk(folder):
            dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRS)
            for name in sorted(files):
                if fnmatch.fnmatch(name, glob):
                    yield Path(current) / name

    def grep(
            self, pattern: str, path: str = ".", glob: str = "*.py",
            ignore_case: bool = False) -> str:
        """Search the regex *pattern* in the files under *path*.

        Returns ``file:line: text`` lines, at most MAX_GREP_MATCHES.
        Binary, non-UTF-8 and files bigger than MAX_GREP_FILE_BYTES are
        skipped.

        Raises:
            ValueError: If the regex or the path is invalid.
        """
        try:
            regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as e:
            raise ValueError(f"invalid regex '{pattern}': {e}") from e
        start = self.resolve(path)
        files = [start] if start.is_file() else self._walk_files(start, glob)

        matches: List[str] = []
        for file in files:
            resolved = file.resolve()
            if (not resolved.is_relative_to(self.root)
                    or resolved.stat().st_size > MAX_GREP_FILE_BYTES):
                continue
            try:
                text = resolved.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            rel = self.relative(file)
            for n, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    text_line = clip_line(line.strip(), 200)
                    matches.append(f"{rel}:{n}: {text_line}")
                    if len(matches) == MAX_GREP_MATCHES:
                        matches.append(
                            f"[stopped at {MAX_GREP_MATCHES} matches: "
                            "narrow the pattern, the path or the glob]")
                        return truncate("\n".join(matches))

        if not matches:
            return (f"[no match for '{pattern}' in "
                    f"{self.relative(start)} ({glob})]")
        return truncate("\n".join(matches))

    # --- writing (only registered with --writable) --------------------------

    def _resolve_for_write(self, path: str) -> Path:
        """Like resolve(), but the target may not exist yet and must not be
        inside an ignored folder (.git, .venv...).

        Raises:
            ValueError: If the path leaves the workspace or is protected.
        """
        target = (self.root / path).resolve()
        if not target.is_relative_to(self.root) or target == self.root:
            raise ValueError(f"'{path}' is outside the workspace")
        if self._is_ignored(target):
            raise ValueError(f"'{path}' is in a protected folder")
        return target

    def edit_file(self, path: str, old_str: str, new_str: str) -> str:
        """Replace the unique occurrence of *old_str* by *new_str*.

        A Python file is only written if it still parses (when it parsed
        before), so a bad edit never breaks the user's code. Returns the
        diff of the change.

        Raises:
            ValueError: If the file is missing, *old_str* is empty, absent
                or ambiguous, or the edit breaks the Python syntax.
        """
        target = self._resolve_for_write(path)
        if not target.is_file():
            raise ValueError(f"'{path}' is not an existing file, "
                             "use create_file for a new file")
        if not old_str:
            raise ValueError("old_str is empty")
        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"'{path}' is not a UTF-8 text file") from e
        count = content.count(old_str)
        if count == 0:
            raise ValueError(
                f"old_str not found in '{path}': copy it exactly from "
                "read_file, without the line-number prefix, and check "
                "the indentation")
        if count > 1:
            raise ValueError(
                f"old_str appears {count} times in '{path}': add "
                "surrounding lines to make it unique")
        new_content = content.replace(old_str, new_str, 1)
        check_python_syntax(target, content, new_content)
        target.write_text(new_content, encoding="utf-8")
        return f"Edited {self.relative(target)}\n" + diff_text(
            self.relative(target), content, new_content)

    def create_file(self, path: str, content: str) -> str:
        """Create a new file (parent folders included); never overwrites.

        Raises:
            ValueError: If the file exists or the Python code is invalid.
        """
        target = self._resolve_for_write(path)
        if target.exists():
            raise ValueError(f"'{path}' already exists, use edit_file")
        check_python_syntax(target, None, content)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        lines = content.count("\n") + (not content.endswith("\n"))
        return f"Created {self.relative(target)} ({lines} lines)"


def check_python_syntax(target: Path, old: str | None, new: str) -> None:
    """Refuse a .py content that does not parse, unless the old one did
    not parse either (the model may be fixing a broken file).

    Raises:
        ValueError: With the line of the syntax error.
    """
    if target.suffix != ".py":
        return
    try:
        ast.parse(new)
    except SyntaxError as e:
        if old is not None:
            try:
                ast.parse(old)
            except SyntaxError:
                return
        raise ValueError(
            f"not written: the result has a syntax error line {e.lineno}: "
            f"{e.msg}") from e


def diff_text(name: str, old: str, new: str) -> str:
    """Unified diff of a change, cut to MAX_DIFF_LINES lines."""
    lines = list(difflib.unified_diff(
        old.splitlines(), new.splitlines(),
        f"a/{name}", f"b/{name}", lineterm="", n=2))
    if len(lines) > MAX_DIFF_LINES:
        lines = lines[:MAX_DIFF_LINES] + [
            f"[diff cut at {MAX_DIFF_LINES} lines]"]
    return "\n".join(lines)
