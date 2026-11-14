"""Messages between host and worker: what the host accepts."""
from src.sandbox.channel import StreamChannel
from src.sandbox.protocol import (
    InitMsg, CodeMsg, MAX_MESSAGE_BYTES,
    recv_init_message, recv_worker_message, send_message
)
from src.models_agent.SandboxConfig import SandboxConfig
from src.models_agent.ToolSpec import ToolSpec

from pydantic import ValidationError
from typing import Iterator
import os
import pytest


@pytest.fixture
def ch() -> Iterator[StreamChannel]:
    r, w = os.pipe()
    channel = StreamChannel(r, w, read_timeout=0.5)
    yield channel
    channel.close()


def test_init_message_round_trip(ch):
    msg = InitMsg(config=SandboxConfig(),
                  tools={"add": ToolSpec(name="add")}, echo=True)
    send_message(ch, msg)
    assert recv_init_message(ch) == msg


def test_init_expected_first(ch):
    send_message(ch, CodeMsg(code="print(1)"))
    with pytest.raises(ValidationError):
        recv_init_message(ch)


@pytest.mark.parametrize("raw", [
    b"not json",
    b'{"type": "unknown"}',
    b'{"type": "code", "code": "x"}',          # a host message
    b'{"type": "ready", "extra": 1}',           # unknown field
    b'{"type": "stdout"}',                      # missing field
])
def test_invalid_worker_messages_refused(ch, raw):
    ch.send_bytes(raw)
    with pytest.raises(ValidationError):
        recv_worker_message(ch)


def test_oversized_worker_message_refused(ch):
    # Only the header: the host must refuse before reading the payload.
    os.write(ch._write_fd, (MAX_MESSAGE_BYTES + 1).to_bytes(4, "big"))
    with pytest.raises(OSError):
        recv_worker_message(ch)
