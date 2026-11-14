"""Command line of the worker and control of its container."""
from ..models_agent.IsolationConfig import IsolationConfig

from pathlib import Path
from typing import List
import os
import subprocess
import sys

WORKSPACE_DIR = "/workspace"
SCRATCH_DIR = "/tmp/agent"
SANDBOX_UID = "65534:65534"

# Directories that must never be the workspace, nor be inside it.
SENSITIVE_DIRS = [
    Path("/"), Path("/etc"), Path("/root"), Path("/run"), Path("/var/run"),
    Path.home(),
    Path.home() / ".ssh",
    Path.home() / ".gnupg",
    Path.home() / ".aws",
    Path.home() / ".config",
    Path.home() / ".docker",
]

DOCKER_CLI_ENV = {k: v for k, v in os.environ.items()
                  if k in ("PATH", "HOME", "DOCKER_HOST")}


def visible_directories(iso: IsolationConfig) -> List[str]:
    """Return the directories the code can use inside the worker.

    The same function decides what is mounted and what is announced to the
    worker (open() restriction) and to the LLM (manual), so that they never
    differ.
    """
    dirs = [SCRATCH_DIR]
    if iso.workspace is not None:
        dirs.insert(0, WORKSPACE_DIR)
    return dirs


def check_workspace(path: Path) -> Path:
    """Return the real path of a workspace, or raise ValueError.

    Symbolic links are resolved first. The workspace must be an existing
    directory, must not be a sensitive directory and must not contain one
    (e.g. the home directory contains ~/.ssh).

    Raises:
        ValueError: If the workspace cannot be mounted safely.
    """
    try:
        real = path.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as e:
        raise ValueError(f"workspace '{path}' does not exist") from e
    if not real.is_dir():
        raise ValueError(f"workspace '{real}' is not a directory")
    for sensitive in SENSITIVE_DIRS:
        sensitive = sensitive.resolve()
        if real == sensitive:
            raise ValueError(f"workspace '{real}' is a sensitive directory")
        if sensitive.is_relative_to(real):
            raise ValueError(
                f"workspace '{real}' contains the sensitive directory "
                f"'{sensitive}'"
            )
    return real


def build_worker_command(name: str, iso: IsolationConfig,
                         max_memory_mb: int) -> List[str]:
    """Return the command line that starts the worker.

    With the "local" backend the worker is a plain process (for tests, NOT
    isolated). With "docker" it runs in a hardened container named name.

    Args:
        name: Unique name of the container, used to kill it.
        iso: Isolation settings.
        max_memory_mb: Memory limit of the worker (RLIMIT_AS), the
            container also gets the tmpfs size and iso.memory_margin_mb.

    Raises:
        ValueError: If the workspace cannot be mounted safely.
    """
    if iso.backend == "local":
        return [sys.executable, "-m", "src.sandbox.worker"]

    # The tmpfs lives in RAM and counts in the container memory limit.
    memory = f"{max_memory_mb + iso.scratch_size_mb + iso.memory_margin_mb}m"
    user = SANDBOX_UID
    cmd = [
        "docker", "run", "--rm", "-i",
        "--name", name,
        "--init",
        "--log-driver=none",
        "--pull=never",
        f"--runtime={iso.runtime}",
        "--network=none",
        "--cap-drop=ALL",
        "--security-opt", "no-new-privileges",
        "--read-only",
        "--tmpfs",
        f"{SCRATCH_DIR}:rw,size={iso.scratch_size_mb}m,mode=1777,"
        "noexec,nosuid,nodev",
        f"--memory={memory}",
        f"--memory-swap={memory}",
        f"--cpus={iso.cpus}",
        f"--pids-limit={iso.pids_limit}",
    ]
    if iso.workspace is not None:
        src = check_workspace(iso.workspace)
        mount = f"type=bind,src={src},dst={WORKSPACE_DIR}"
        if iso.workspace_writable:
            # Files created by the agent must belong to the host user.
            user = f"{os.getuid()}:{os.getgid()}"
        else:
            mount += ",readonly"
        cmd += ["--mount", mount]
    cmd += ["--user", user, iso.image]
    return cmd


def kill_container(name: str) -> None:
    """Kill a container by its name (no error if it is already gone)."""
    subprocess.run(["docker", "kill", name], env=DOCKER_CLI_ENV,
                   capture_output=True, timeout=10)
