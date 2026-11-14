"""StreamChannel: length-prefixed messages over raw file descriptors."""
from src.sandbox.channel import StreamChannel

from typing import Iterator, Tuple
import os
import threading
import time
import pytest


@pytest.fixture
def pair() -> Iterator[Tuple[StreamChannel, int]]:
    """A channel reading its own writes, and the raw write fd."""
    r, w = os.pipe()
    ch = StreamChannel(r, w, read_timeout=0.5)
    yield ch, w
    ch.close()


def test_round_trip(pair):
    ch, _ = pair
    ch.send_bytes(b"hello")
    assert ch.recv_bytes() == b"hello"


def test_empty_message(pair):
    ch, _ = pair
    ch.send_bytes(b"")
    assert ch.recv_bytes() == b""


def test_message_larger_than_pipe_buffer(pair):
    ch, _ = pair
    big = os.urandom(5_000_000)
    # The writer blocks when the 64 KB pipe buffer is full: it needs a
    # reader at the same time.
    writer = threading.Thread(target=ch.send_bytes, args=(big,))
    writer.start()
    assert ch.recv_bytes() == big
    writer.join()


def test_announced_length_over_limit(pair):
    ch, _ = pair
    ch.send_bytes(b"y" * 100)
    with pytest.raises(OSError, match="exceeds the limit"):
        ch.recv_bytes(10)


def test_end_of_stream():
    r, w = os.pipe()
    ch = StreamChannel(r, -1)
    os.write(w, b"\x00\x00")
    os.close(w)
    with pytest.raises(EOFError):
        ch.recv_bytes()
    os.close(r)


def test_started_message_never_finished(pair):
    ch, w = pair
    os.write(w, b"\x00\x00")       # half a header, then nothing
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        ch.recv_bytes()
    assert time.monotonic() - start < 2


def test_poll_empty_channel(pair):
    ch, _ = pair
    start = time.monotonic()
    assert ch.poll(0.1) is False
    assert time.monotonic() - start >= 0.09


def test_close_twice_and_use_after_close(pair):
    ch, _ = pair
    ch.close()
    ch.close()
    with pytest.raises(OSError):
        ch.send_bytes(b"x")
