"""Code MCP server: workspace confinement, tool outputs, and a real
stdio round trip through McpBridge (BM25 only, no GPU needed)."""
from src.mcp_server.config import MAX_GREP_MATCHES, MAX_READ_LINES
from src.mcp_server.workspace import Workspace
from src.sandbox.McpBrige import McpBridge
from src.sandbox.errors import McpToolError

from pathlib import Path
import os
import sys
import time
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A tiny project, plus a secret file next to it (outside the root)."""
    root = tmp_path / "project"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "config.py").write_text(
        "def load_config(path):\n"
        "    \"\"\"Read the YAML settings file.\"\"\"\n"
        "    return open(path).read()\n")
    (root / "pkg" / "retry.py").write_text(
        "class RetryPolicy:\n"
        "    def backoff(self, attempt):\n"
        "        return 2 ** attempt\n")
    (root / "notes.txt").write_text("TODO: load_config cache\n")
    (root / ".venv" / "lib").mkdir(parents=True)
    (root / ".venv" / "lib" / "dep.py").write_text("def load_config(): ...\n")
    (tmp_path / "secret.txt").write_text("password\n")
    return root


@pytest.fixture
def ws(project: Path) -> Workspace:
    return Workspace(project)


# --- confinement ------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "../secret.txt", "/etc/passwd", "pkg/../../secret.txt"])
def test_paths_outside_root_are_refused(ws, path):
    with pytest.raises(ValueError, match="outside the workspace"):
        ws.read_file(path)


def test_symlink_escape_is_refused(ws, project):
    (project / "link.txt").symlink_to(project.parent / "secret.txt")
    with pytest.raises(ValueError, match="outside the workspace"):
        ws.read_file("link.txt")
    assert "no match" in ws.grep("password", glob="*")
    assert "link.txt" not in ws.list_files()


def test_symlinked_folder_escape_is_hidden(ws, project):
    (project / "out").symlink_to(project.parent)
    assert "secret" not in ws.list_files(".", "**/*")
    assert "no match" in ws.grep("password", glob="*")


def test_missing_path(ws):
    with pytest.raises(ValueError, match="does not exist"):
        ws.read_file("nope.py")


# --- read_file --------------------------------------------------------------

def test_read_file_numbers_lines(ws):
    out = ws.read_file("pkg/config.py")
    assert out.splitlines()[0] == "pkg/config.py (lines 1-3 of 3)"
    assert "1| def load_config(path):" in out


def test_read_file_range(ws):
    out = ws.read_file("pkg/config.py", start_line=2, end_line=2)
    assert "(lines 2-2 of 3)" in out
    assert "load_config" not in out


def test_read_file_caps_lines_and_says_how_to_continue(ws, project):
    (project / "big.py").write_text("x = 1\n" * (MAX_READ_LINES + 50))
    out = ws.read_file("big.py")
    assert f"(lines 1-{MAX_READ_LINES} of {MAX_READ_LINES + 50})" in out
    assert f"start_line={MAX_READ_LINES + 1}" in out


def test_read_file_rejects_binary_and_folders(ws, project):
    (project / "blob.bin").write_bytes(b"\xff\xfe\x00")
    with pytest.raises(ValueError, match="not a UTF-8"):
        ws.read_file("blob.bin")
    with pytest.raises(ValueError, match="folder"):
        ws.read_file("pkg")


def test_read_file_bad_range(ws):
    with pytest.raises(ValueError, match="has 3 lines"):
        ws.read_file("pkg/config.py", start_line=10)


# --- list_files / grep ------------------------------------------------------

def test_list_files_hides_ignored_dirs(ws):
    assert ws.list_files().splitlines() == ["notes.txt", "pkg/"]
    recursive = ws.list_files(".", "**/*.py")
    assert "pkg/config.py" in recursive
    assert ".venv" not in recursive


def test_grep_finds_matches_with_location(ws):
    out = ws.grep(r"def load_config")
    assert out == "pkg/config.py:1: def load_config(path):"


def test_grep_glob_and_case(ws):
    assert "notes.txt:1:" in ws.grep("LOAD_CONFIG", glob="*.txt",
                                     ignore_case=True)
    assert "no match" in ws.grep("LOAD_CONFIG")


def test_grep_invalid_regex(ws):
    with pytest.raises(ValueError, match="invalid regex"):
        ws.grep("(")


def test_grep_stops_at_max_matches(ws, project):
    (project / "many.py").write_text("hit\n" * (MAX_GREP_MATCHES + 10))
    out = ws.grep("hit", path="many.py")
    assert len(out.splitlines()) == MAX_GREP_MATCHES + 1
    assert "stopped at" in out


# --- real server over stdio -------------------------------------------------

@pytest.fixture
def bridge(project: Path):
    """The server started by McpBridge, exactly as the agent will."""
    b = McpBridge(
        f"{sys.executable} -m src.mcp_server --root {project} --no-embedding",
        env={**os.environ, "PYTHONPATH": str(ROOT)})
    b.connect()
    yield b
    b.close()


def test_server_lists_the_four_tools(bridge):
    tools = bridge.get_tools_list()
    assert set(tools) == {"search_code", "grep", "read_file", "list_files"}
    assert "query" in tools["search_code"].input_schema["properties"]


def test_server_tools_round_trip(bridge):
    assert "pkg/config.py:1:" in bridge.call_tool(
        "grep", {"pattern": "load_config"})
    assert "RetryPolicy" in bridge.call_tool(
        "read_file", {"path": "pkg/retry.py"})
    with pytest.raises(McpToolError, match="outside the workspace"):
        bridge.call_tool("read_file", {"path": "../secret.txt"})


def test_server_search_code(bridge):
    deadline = time.monotonic() + 30
    while True:
        try:
            out = bridge.call_tool(
                "search_code", {"query": "retry backoff attempt", "k": 1})
            break
        except McpToolError as e:
            if "still being built" not in str(e) or time.monotonic() > deadline:
                raise
    assert out.startswith("--- [1] pkg/retry.py:1-3 ---")
    assert "return 2 ** attempt" in out
    assert ".venv" not in out


# --- writing (--writable) ---------------------------------------------------

def test_edit_file_replaces_and_returns_diff(ws, project):
    out = ws.edit_file("pkg/retry.py", "2 ** attempt", "min(2 ** attempt, 60)")
    assert out.startswith("Edited pkg/retry.py")
    assert "-        return 2 ** attempt" in out
    assert "+        return min(2 ** attempt, 60)" in out
    assert "min(2 ** attempt, 60)" in (project / "pkg/retry.py").read_text()


@pytest.mark.parametrize("old, message", [
    ("does not exist", "not found"),
    ("", "empty"),
])
def test_edit_file_bad_old_str(ws, old, message):
    with pytest.raises(ValueError, match=message):
        ws.edit_file("pkg/retry.py", old, "x")


def test_edit_file_ambiguous(ws, project):
    (project / "dup.py").write_text("a = 1\na = 1\n")
    with pytest.raises(ValueError, match="2 times"):
        ws.edit_file("dup.py", "a = 1", "a = 2")


def test_edit_file_refuses_broken_python(ws, project):
    before = (project / "pkg/retry.py").read_text()
    with pytest.raises(ValueError, match="syntax error"):
        ws.edit_file("pkg/retry.py", "return 2 ** attempt", "return (2 **")
    assert (project / "pkg/retry.py").read_text() == before


def test_edit_file_may_fix_a_broken_file(ws, project):
    (project / "broken.py").write_text("def f(:\n    pass\n")
    ws.edit_file("broken.py", "def f(:", "def f(")  # parses after
    (project / "broken2.py").write_text("def f(:\n    x = 1\n")
    ws.edit_file("broken2.py", "x = 1", "x = 2")    # still broken: allowed


@pytest.mark.parametrize("path, message", [
    ("../secret.txt", "outside"),
    (".venv/lib/dep.py", "protected"),
    (".git/config", "protected"),
])
def test_writes_outside_or_protected_are_refused(ws, path, message):
    with pytest.raises(ValueError, match=message):
        ws.create_file(path, "x")


def test_create_file(ws, project):
    assert ws.create_file("pkg/new/mod.py", "X = 1\n") == \
        "Created pkg/new/mod.py (1 lines)"
    assert (project / "pkg/new/mod.py").read_text() == "X = 1\n"
    with pytest.raises(ValueError, match="already exists"):
        ws.create_file("pkg/new/mod.py", "Y = 2\n")
    with pytest.raises(ValueError, match="syntax error"):
        ws.create_file("bad.py", "def (")


@pytest.fixture
def writable_bridge(project: Path):
    b = McpBridge(
        f"{sys.executable} -m src.mcp_server --root {project} "
        "--no-embedding --writable",
        env={**os.environ, "PYTHONPATH": str(ROOT)})
    b.connect()
    yield b
    b.close()


def wait_search(bridge, query: str, expect: str, timeout: float = 30) -> str:
    """Call search_code until *expect* is in the result (index refresh)."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            out = bridge.call_tool("search_code", {"query": query, "k": 1})
            if expect in out or time.monotonic() > deadline:
                return out
        except McpToolError as e:
            if "still being built" not in str(e):
                raise
        time.sleep(0.2)


def test_read_only_server_has_no_write_tools(bridge):
    assert "edit_file" not in bridge.get_tools_list()


def test_writable_server_edits_and_reindexes(writable_bridge, project):
    assert {"edit_file", "create_file"} <= set(writable_bridge.get_tools_list())
    wait_search(writable_bridge, "retry backoff", "RetryPolicy")
    writable_bridge.call_tool("edit_file", {
        "path": "pkg/retry.py",
        "old_str": "class RetryPolicy:",
        "new_str": "class JitterRetryPolicy:"})
    out = wait_search(writable_bridge, "jitter retry", "JitterRetryPolicy")
    assert "class JitterRetryPolicy:" in out
    writable_bridge.call_tool("create_file", {
        "path": "pkg/telemetry.py",
        "content": "def send_metrics_to_collector():\n    pass\n"})
    out = wait_search(writable_bridge, "send metrics collector",
                      "send_metrics_to_collector")
    assert "pkg/telemetry.py" in out
