"""Shared fixtures of the sandbox tests.

The integration tests (marker "isolation") need Docker, the runsc runtime
(gVisor) and the sandbox image, which is (re)built once per session so that
the tests always run the current worker code. Without them the tests are
skipped, or fail with --require-isolation.
"""
from src.sandbox.api import Sandbox
from src.models_agent.SandboxConfig import SandboxConfig
from src.models_agent.IsolationConfig import IsolationConfig
from src.models_agent.ExecutionResult import ExecutionResult

from pathlib import Path
from typing import Callable, Iterator, List, Optional, Set
import shutil
import subprocess
import time
import pytest

ROOT = Path(__file__).resolve().parents[1]

# The tests assume the attacker already escaped the Python restrictions:
# every integration test starts with a real os module and the real open().
ESCAPE = (
    'os = print.__self__.__import__("os")\n'
    'open = print.__self__.open\n'
)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add --require-isolation."""
    parser.addoption(
        "--require-isolation", action="store_true",
        help="Fail instead of skip when Docker or gVisor is missing"
    )


def _prepare_isolation() -> Optional[str]:
    """Build the image and return why isolation tests cannot run, or None."""
    if shutil.which("docker") is None:
        return "docker is not installed"
    info = subprocess.run(
        ["docker", "info", "--format", "{{json .Runtimes}}"],
        capture_output=True, text=True
    )
    if info.returncode != 0:
        return "the docker daemon is not reachable"
    if "runsc" not in info.stdout:
        return "the runsc (gVisor) runtime is not installed"
    build = subprocess.run(
        ["docker", "build", "-q", "-f", "docker/sandbox.Dockerfile",
         "-t", IsolationConfig().image, "."],
        cwd=ROOT, capture_output=True, text=True
    )
    if build.returncode != 0:
        return f"cannot build the sandbox image: {build.stderr[-500:]}"
    return None


@pytest.fixture(scope="session")
def isolation_ready(request: pytest.FixtureRequest) -> None:
    """Skip (or fail with --require-isolation) if isolation is missing."""
    problem = _prepare_isolation()
    if problem is not None:
        if request.config.getoption("--require-isolation"):
            pytest.fail(problem)
        pytest.skip(problem)


def sandbox_containers() -> Set[str]:
    """Return the ids of the existing sandbox containers."""
    out = subprocess.run(
        ["docker", "ps", "-aq", "--filter", "name=sandbox-"],
        capture_output=True, text=True
    ).stdout
    return set(out.split())


SandboxFactory = Callable[..., Sandbox]


@pytest.fixture
def make_sandbox(isolation_ready: None) -> Iterator[SandboxFactory]:
    """Factory of sandboxes, closed after the test.

    After the test, no container created during the test may remain.
    """
    before = sandbox_containers()
    created: List[Sandbox] = []

    def factory(config: Optional[SandboxConfig] = None,
                **isolation: object) -> Sandbox:
        sb = Sandbox(
            sandbox_config=config or SandboxConfig(
                max_execution_time_seconds=5),
            isolation=IsolationConfig(**isolation),  # type: ignore[arg-type]
        )
        created.append(sb)
        return sb

    yield factory
    for sb in created:
        sb.close()
    # --rm removes the containers asynchronously.
    deadline = time.monotonic() + 10
    leaked = sandbox_containers() - before
    while leaked and time.monotonic() < deadline:
        time.sleep(0.2)
        leaked = sandbox_containers() - before
    assert not leaked, f"containers left behind: {leaked}"


@pytest.fixture
def sb(make_sandbox: SandboxFactory) -> Sandbox:
    """A sandbox with the default isolation (gVisor, no workspace)."""
    return make_sandbox()


def run(sb: Sandbox, code: str) -> ExecutionResult:
    """Execute code after the escape prelude."""
    return sb.execute(ESCAPE + code)
