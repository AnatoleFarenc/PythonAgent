"""An attacker who escaped the Python restrictions cannot reach the host."""
from tests.conftest import run

import pytest

pytestmark = pytest.mark.isolation


def test_python_restrictions_are_not_the_boundary(sb):
    # The known escapes still work: security relies on the container.
    r = run(sb, "print(type(os).__name__, os.getpid() > 0)")
    assert r.stdout.strip() == "module True"


def test_runs_under_gvisor(sb):
    assert "gvisor" in run(sb, "print(os.uname().release)").stdout


def test_no_privileges(sb):
    status = run(sb, "print(open('/proc/self/status').read())").stdout
    assert "CapEff:\t0000000000000000" in status
    assert "CapPrm:\t0000000000000000" in status
    assert "NoNewPrivs:\t1" in status
    assert run(sb, "print(os.getuid(), os.getgid())").stdout.split() == [
        "65534", "65534"]


def test_host_environment_does_not_leak(make_sandbox, monkeypatch):
    monkeypatch.setenv("SANDBOX_CANARY", "c4n4ry-s3cr3t")
    sb = make_sandbox()
    r = run(sb, "print(dict(os.environ))\n"
                "print(open('/proc/self/environ', 'rb').read())\n"
                "print(open('/proc/1/environ', 'rb').read())")
    assert not r.error
    assert "c4n4ry-s3cr3t" not in r.stdout


def test_host_files_are_unreachable(sb, tmp_path):
    canary = tmp_path / "canary.txt"
    canary.write_text("c4n4ry")
    r = run(sb, f"print(os.path.exists({str(canary)!r}))\n"
                "print(os.path.exists('/home/' + 'x'), "
                "sorted(os.listdir('/home')))\n"
                "print(sorted(os.listdir('/app/src')))")
    lines = r.stdout.splitlines()
    assert lines[0] == "False"
    assert lines[1] == "False []"
    assert lines[2] == "['__init__.py', 'models_agent', 'sandbox']"


def test_no_network(sb):
    r = run(sb, "s = print.__self__.__import__('_socket').socket()\n"
                "s.settimeout(3)\n"
                "try:\n"
                "    s.connect(('1.1.1.1', 80)); print('CONNECTED')\n"
                "except OSError as e:\n"
                "    print('blocked', e.errno)")
    assert r.stdout.startswith("blocked"), r.stdout


def test_no_dns(sb):
    r = run(sb, "sock = print.__self__.__import__('socket')\n"
                "try:\n"
                "    print(sock.getaddrinfo('example.com', 80))\n"
                "except OSError as e:\n"
                "    print('blocked')")
    assert r.stdout.strip() == "blocked", r.stdout


def test_root_filesystem_read_only(sb):
    # /tmp is world-writable in the image: only --read-only stops this.
    r = run(sb, "os.open('/tmp/x', os.O_WRONLY | os.O_CREAT)")
    assert "Read-only file system" in r.error


def test_worker_code_cannot_be_modified(sb):
    r = run(sb, "os.open('/app/src/sandbox/worker.py', os.O_WRONLY)")
    assert r.error


def test_scratch_is_noexec(sb):
    r = run(sb, "import_ = print.__self__.__import__\n"
                "import_('shutil').copy('/bin/true', '/tmp/agent/t')\n"
                "os.chmod('/tmp/agent/t', 0o755)\n"
                "os.execv('/tmp/agent/t', ['t'])")
    assert "Permission denied" in r.error


def test_setuid_binaries_give_nothing(sb):
    # su is setuid root in the image. With runc and without
    # no-new-privileges its effective uid becomes 0 while it waits for the
    # password; here it must stay 65534 (no-new-privileges, and gVisor).
    r = run(sb, "sp = print.__self__.__import__('subprocess')\n"
                "time = print.__self__.__import__('time')\n"
                "p = sp.Popen(['su', 'root', '-c', 'true'], stdin=sp.PIPE,"
                " stdout=sp.PIPE, stderr=sp.PIPE)\n"
                "time.sleep(1)\n"
                "try:\n"
                "    status = open(f'/proc/{p.pid}/status').read()\n"
                "    print([l for l in status.splitlines()"
                " if l.startswith('Uid:')][0].split()[1:])\n"
                "finally:\n"
                "    p.kill()")
    assert r.stdout.strip() == "['65534', '65534', '65534', '65534']", \
        r.stdout + r.error


def test_garbage_on_protocol_does_not_break_host(sb):
    sb.execute("x = 1")
    run(sb, "for fd in range(3, 16):\n"
            "    try:\n"
            "        os.write(fd, b'\\xff\\xff\\xff\\xff')\n"
            "    except OSError:\n"
            "        pass")
    assert sb.execute("print('alive')").stdout.strip() == "alive"
