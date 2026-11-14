"""Transport used between the main process and the worker.

Messages are sent on a byte stream (pipes, stdin/stdout of a container)
as a 4 bytes big-endian length followed by the payload.
"""
from typing import Optional, Protocol
import os
import select
import struct
import time

HEADER = struct.Struct("!I")
MAX_FRAME = 2 ** 32 - 1


class Channel(Protocol):
    """Methods needed by the protocol.

    multiprocessing.connection.Connection already matches this interface.
    """

    def send_bytes(self, buf: bytes) -> None: ...
    def recv_bytes(self, maxlength: Optional[int] = None) -> bytes: ...
    def poll(self, timeout: Optional[float] = 0.0) -> bool: ...
    def close(self) -> None: ...


class StreamChannel:
    """Length-prefixed messages over two raw file descriptors.

    The channel owns the descriptors: close() closes them.
    """

    def __init__(self, read_fd: int, write_fd: int,
                 read_timeout: Optional[float] = None) -> None:
        """Store the descriptors.

        Args:
            read_fd: Descriptor the messages are read from.
            write_fd: Descriptor the messages are written to.
            read_timeout: Max time in seconds to receive the rest of a
                message once its first byte has arrived (None: no limit).
        """
        self._read_fd = read_fd
        self._write_fd = write_fd
        self._read_timeout = read_timeout
        self._closed = False

    def send_bytes(self, buf: bytes) -> None:
        """Send one message.

        Raises:
            ValueError: If the message is longer than MAX_FRAME.
            OSError: If the channel or the other end is closed.
        """
        self._check_open()
        data = bytes(buf)
        if len(data) > MAX_FRAME:
            raise ValueError("message too long")
        self._write_all(HEADER.pack(len(data)) + data)

    def recv_bytes(self, maxlength: Optional[int] = None) -> bytes:
        """Receive one message.

        Wait without limit for the first byte, then read_timeout applies
        to the rest of the message.

        Raises:
            EOFError: If the other end closed the stream.
            OSError: If the announced length is greater than maxlength.
            TimeoutError: If the message is not complete in time.
        """
        self._check_open()
        first = self._read_exact(1, None)
        deadline = (None if self._read_timeout is None
                    else time.monotonic() + self._read_timeout)
        (size,) = HEADER.unpack(first + self._read_exact(3, deadline))
        if maxlength is not None and size > maxlength:
            raise OSError(
                f"message of {size} bytes exceeds the limit of "
                f"{maxlength} bytes"
            )
        return self._read_exact(size, deadline)

    def poll(self, timeout: Optional[float] = 0.0) -> bool:
        """Return True if data can be read within timeout seconds.

        timeout=None waits forever.
        """
        self._check_open()
        ready, _, _ = select.select([self._read_fd], [], [], timeout)
        return bool(ready)

    def close(self) -> None:
        """Close the descriptors. Calling it twice does nothing."""
        if self._closed:
            return
        self._closed = True
        for fd in {self._read_fd, self._write_fd}:
            try:
                os.close(fd)
            except OSError:
                pass

    def _check_open(self) -> None:
        """Raise OSError if the channel is closed."""
        if self._closed:
            raise OSError("channel is closed")

    def _write_all(self, data: bytes) -> None:
        """Write all bytes, os.write may write only a part of them."""
        view = memoryview(data)
        while view:
            written = os.write(self._write_fd, view)
            view = view[written:]

    def _read_exact(self, n: int, deadline: Optional[float]) -> bytes:
        """Read exactly n bytes, os.read may return less.

        Raises:
            EOFError: If the stream ends before n bytes.
            TimeoutError: If deadline is passed.
        """
        buf = bytearray()
        while len(buf) < n:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("incomplete message")
                ready, _, _ = select.select([self._read_fd], [], [],
                                            remaining)
                if not ready:
                    raise TimeoutError("incomplete message")
            chunk = os.read(self._read_fd, n - len(buf))
            if not chunk:
                raise EOFError("stream closed")
            buf += chunk
        return bytes(buf)
