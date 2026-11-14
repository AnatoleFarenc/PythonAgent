"""Resource limits: errors are clear and the environment survives."""
from tests.conftest import run

import pytest

pytestmark = pytest.mark.isolation

FORK_BOMB = (
    "n = 0\n"
    "try:\n"
    "    while True:\n"
    "        pid = os.fork()\n"
    "        if pid == 0:\n"
    "            print.__self__.__import__('time').sleep(30)\n"
    "            os._exit(0)\n"
    "        n += 1\n"
    "except OSError as e:\n"
    "    print('stopped after', n, 'errno', e.errno)"
)

FILL_SCRATCH = (
    "f = open('/tmp/agent/big', 'wb')\n"
    "try:\n"
    "    for _ in range(200):\n"
    "        f.write(b'0' * 1024 * 1024)\n"
    "except OSError as e:\n"
    "    print('errno', e.errno)\n"
    "f.close()"
)


def survived(sb) -> bool:
    """True if the variable "kept" defined before is still there."""
    return sb.execute("print(kept)").stdout.strip() == "42"


def test_memory_limit_is_a_clear_error(sb):
    sb.execute("kept = 42")
    r = sb.execute("b = bytearray(2 * 1024 ** 3)")
    assert "MemoryError" in r.error
    assert survived(sb)


def test_fork_bomb_is_stopped(sb):
    sb.execute("kept = 42")
    r = run(sb, FORK_BOMB)
    assert "errno 11" in r.stdout, r.stdout + r.error     # EAGAIN
    assert survived(sb)


def test_fork_bomb_with_full_scratch(sb):
    # Regression: the tmpfs counts in the container memory; it used to eat
    # the margin and the fork bomb crashed the container.
    sb.execute("kept = 42")
    assert "errno 28" in run(sb, FILL_SCRATCH).stdout
    r = run(sb, FORK_BOMB)
    assert "errno 11" in r.stdout, r.stdout + r.error
    assert survived(sb)


def test_scratch_size_limit(sb):
    assert "errno 28" in run(sb, FILL_SCRATCH).stdout      # ENOSPC


def test_scratch_is_writable(sb):
    r = sb.execute("f = open('/tmp/agent/a.txt', 'w'); f.write('hi'); "
                   "f.close(); print(open('/tmp/agent/a.txt').read())")
    assert r.stdout.strip() == "hi"


def test_timeout_restarts_environment(sb):
    sb.execute("kept = 42")
    first = sb._container
    r = sb.execute("while True: pass")
    assert r.timed_out
    assert sb._container != first
    assert sb.execute("print('ok')").stdout.strip() == "ok"
    assert not survived(sb)


def test_crash_restarts_environment(sb):
    r = run(sb, "os._exit(1)")
    assert "crashed" in r.error
    assert sb.execute("print('ok')").stdout.strip() == "ok"


def test_variables_persist(sb):
    sb.execute("x = 2")
    assert sb.execute("print(x * 21)").stdout.strip() == "42"
