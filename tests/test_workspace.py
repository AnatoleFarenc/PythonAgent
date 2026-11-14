"""The workspace: visible as /workspace, nothing more of the host."""
from tests.conftest import run

from pathlib import Path
import os
import pytest

pytestmark = pytest.mark.isolation


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    (ws / "sub").mkdir(parents=True)
    (ws / "readme.txt").write_text("hello from host")
    for path in [ws, ws / "sub", ws / "readme.txt"]:
        path.chmod(0o755 if path.is_dir() else 0o644)
    return ws


def snapshot(path: Path) -> list:
    return sorted((str(p.relative_to(path)), p.stat().st_mtime_ns)
                  for p in path.rglob("*"))


def test_no_workspace_by_default(sb):
    assert run(sb, "print(os.path.exists('/workspace'))").stdout.strip() \
        == "False"
    assert "/workspace" not in sb.get_manual()


def test_read_only_workspace(make_sandbox, workspace):
    before = snapshot(workspace)
    sb = make_sandbox(workspace=workspace)
    assert "/workspace, /tmp/agent" in sb.get_manual()
    assert sb.execute("print(open('/workspace/readme.txt').read())"
                      ).stdout.strip() == "hello from host"
    r = sb.execute("open('/workspace/new.txt', 'w').write('x')")
    assert r.error
    # Even as the owner of the files, the mount itself is read-only.
    r = run(sb, "os.open('/workspace/new.txt', os.O_WRONLY | os.O_CREAT)")
    assert r.error
    assert snapshot(workspace) == before


def test_host_path_does_not_leak(make_sandbox, workspace):
    sb = make_sandbox(workspace=workspace)
    r = run(sb, "print(open('/proc/self/mountinfo').read())")
    assert "/workspace" in r.stdout
    assert str(workspace) not in r.stdout
    assert str(workspace.parent) not in r.stdout


def test_parent_of_workspace_is_the_container(make_sandbox, workspace):
    sb = make_sandbox(workspace=workspace)
    r = run(sb, f"print(os.path.exists({str(workspace.parent)!r}), "
                "sorted(os.listdir('/workspace/..')) == "
                "sorted(os.listdir('/')))")
    assert r.stdout.strip() == "False True"


def test_writable_workspace(make_sandbox, workspace):
    sb = make_sandbox(workspace=workspace, workspace_writable=True)
    r = sb.execute("f = open('/workspace/created.txt', 'w'); "
                   "f.write('agent'); f.close()")
    assert not r.error, r.error
    created = workspace / "created.txt"
    assert created.read_text() == "agent"
    assert created.stat().st_uid == os.getuid()


def test_symlink_does_not_escape(make_sandbox, workspace):
    sb = make_sandbox(workspace=workspace, workspace_writable=True)
    r = run(sb, "os.symlink('/', '/workspace/escape')\n"
                f"print(os.path.exists('/workspace/escape' + "
                f"{str(workspace)!r}))\n"
                "print(sorted(os.listdir('/workspace/escape')) == "
                "sorted(os.listdir('/')))")
    assert r.stdout.split() == ["False", "True"]
    # The link stays in the host workspace: it must not be followed by
    # host tools. Remove it so that pytest's cleanup does not follow it.
    (workspace / "escape").unlink()


def test_refused_workspace_starts_nothing(make_sandbox, tmp_path):
    (tmp_path / "home").symlink_to(Path.home())
    with pytest.raises(ValueError):
        make_sandbox(workspace=tmp_path / "home")
