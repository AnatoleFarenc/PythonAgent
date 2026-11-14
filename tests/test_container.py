"""The worker command keeps its hardening (no Docker needed)."""
from src.sandbox.container import (
    build_worker_command, check_workspace, visible_directories,
    SCRATCH_DIR, WORKSPACE_DIR
)
from src.models_agent.IsolationConfig import IsolationConfig

from pathlib import Path
from pydantic import ValidationError
from typing import List
import os
import pytest

HARDENING = ["--rm", "--init", "--network=none", "--cap-drop=ALL",
             "--read-only", "--pull=never", "--log-driver=none",
             "--runtime=runsc"]


def option(cmd: List[str], name: str) -> str:
    """Return the value that follows an option in a command line."""
    return cmd[cmd.index(name) + 1]


def test_hardening_options_present():
    cmd = build_worker_command("sandbox-test", IsolationConfig(), 512)
    for flag in HARDENING:
        assert flag in cmd, flag
    assert option(cmd, "--security-opt") == "no-new-privileges"
    assert option(cmd, "--user") == "65534:65534"
    assert option(cmd, "--name") == "sandbox-test"
    assert "--pids-limit=256" in cmd
    # A tty would translate bytes and corrupt the binary protocol.
    assert "-t" not in cmd and "--tty" not in cmd
    # Nothing after the image: the entrypoint is fixed by the image.
    assert cmd[-1] == IsolationConfig().image


def test_scratch_tmpfs_options():
    cmd = build_worker_command("x", IsolationConfig(), 512)
    tmpfs = option(cmd, "--tmpfs")
    assert tmpfs.startswith(f"{SCRATCH_DIR}:")
    for opt in ("size=64m", "noexec", "nosuid", "nodev"):
        assert opt in tmpfs, opt


def test_memory_limit_counts_tmpfs_and_margin():
    # Regression: the tmpfs is in RAM, without it a full tmpfs left no
    # margin and a fork bomb crashed the container.
    cmd = build_worker_command("x", IsolationConfig(), 512)
    assert f"--memory={512 + 64 + 128}m" in cmd
    assert f"--memory-swap={512 + 64 + 128}m" in cmd


def test_no_workspace_no_mount():
    cmd = build_worker_command("x", IsolationConfig(), 512)
    assert "--mount" not in cmd and "-v" not in cmd


def test_workspace_read_only_by_default(tmp_path):
    cmd = build_worker_command("x", IsolationConfig(workspace=tmp_path), 512)
    mount = option(cmd, "--mount")
    assert mount == f"type=bind,src={tmp_path.resolve()},dst={WORKSPACE_DIR}" \
                    ",readonly"
    assert "-v" not in cmd          # -v creates missing host dirs as root
    assert option(cmd, "--user") == "65534:65534"


def test_writable_workspace_runs_as_host_user(tmp_path):
    iso = IsolationConfig(workspace=tmp_path, workspace_writable=True)
    cmd = build_worker_command("x", iso, 512)
    assert "readonly" not in option(cmd, "--mount")
    assert option(cmd, "--user") == f"{os.getuid()}:{os.getgid()}"


def test_local_backend():
    cmd = build_worker_command("x", IsolationConfig(backend="local"), 512)
    assert cmd[1:] == ["-m", "src.sandbox.worker"]


@pytest.mark.parametrize("bad", [
    "/", "/etc", "/home", str(Path.home()), str(Path.home() / ".ssh"),
])
def test_sensitive_workspaces_refused(bad):
    if not Path(bad).exists():
        pytest.skip(f"{bad} does not exist on this machine")
    with pytest.raises(ValueError):
        check_workspace(Path(bad))


def test_missing_file_and_symlink_workspaces_refused(tmp_path):
    (tmp_path / "f.txt").write_text("x")
    (tmp_path / "home").symlink_to(Path.home())
    for bad in [tmp_path / "missing", tmp_path / "f.txt", tmp_path / "home"]:
        with pytest.raises(ValueError):
            check_workspace(bad)


def test_valid_workspace_is_resolved(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real")
    assert check_workspace(tmp_path / "link") == (tmp_path / "real").resolve()


def test_visible_directories_follow_the_mounts(tmp_path):
    assert visible_directories(IsolationConfig()) == [SCRATCH_DIR]
    assert visible_directories(IsolationConfig(workspace=tmp_path)) == [
        WORKSPACE_DIR, SCRATCH_DIR]


@pytest.mark.parametrize("bad", [
    {"pids_limit": 10},          # gVisor itself needs ~36 threads
    {"runtime": "runcs"},        # typo
    {"backend": "ssh"},
    {"worksapce": "/tmp"},       # typo: must not be silently ignored
])
def test_isolation_config_rejects_bad_values(bad):
    with pytest.raises(ValidationError):
        IsolationConfig(**bad)
