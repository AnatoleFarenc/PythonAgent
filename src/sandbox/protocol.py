"""Messages exchanged between the main process and the worker process.

Every message is a pydantic model sent as UTF-8 JSON bytes. JSON can only
carry data, never code: reading a message sent by the worker (which runs
untrusted code) cannot execute anything, unlike pickle, which is what
Connection.send() and Connection.recv() use.

Worker -> main:  ReadyMsg, StdoutMsg, ToolCallMsg, ResultMsg
Main -> worker:  CodeMsg, StopMsg, ToolResultMsg
"""
from .channel import Channel
from typing import Annotated, Any, Dict, Literal, Optional, Union
from ..models_agent.SandboxConfig import SandboxConfig
from ..models_agent.ToolSpec import ToolSpec

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

MAX_MESSAGE_BYTES = 16 * 1024 * 1024


class _Message(BaseModel):
    """Base class of all messages: unknown fields are refused."""

    model_config = ConfigDict(extra="forbid")


class ReadyMsg(_Message):
    """The worker is started and waits for code."""

    type: Literal["ready"] = "ready"


class StdoutMsg(_Message):
    """A piece of text printed by the code."""

    type: Literal["stdout"] = "stdout"
    text: str


class ToolCallMsg(_Message):
    """The code called an MCP tool: the main process must call it."""

    type: Literal["tool_call"] = "tool_call"
    name: str
    args: Dict[str, Any] = Field(default_factory=dict)


class ResultMsg(_Message):
    """End of the execution of one code block (stdout is sent apart)."""

    type: Literal["result"] = "result"
    error: str = ""
    final_answer: Optional[str] = None


class CodeMsg(_Message):
    """A block of code to execute."""

    type: Literal["code"] = "code"
    code: str


class StopMsg(_Message):
    """Ask the worker to stop."""

    type: Literal["stop"] = "stop"


class ToolResultMsg(_Message):
    """Answer to a ToolCallMsg."""

    type: Literal["tool_result"] = "tool_result"
    error: bool = False
    output: str = ""


WorkerMessage = Annotated[
    Union[ReadyMsg, StdoutMsg, ToolCallMsg, ResultMsg],
    Field(discriminator="type"),
]
MainMessage = Annotated[
    Union[CodeMsg, StopMsg, ToolResultMsg],
    Field(discriminator="type"),
]


class InitMsg(_Message):
    """First message sent to the worker: what it needs to start.

    Replaces the arguments that multiprocessing passed with pickle.
    """

    type: Literal["init"] = "init"
    config: SandboxConfig
    tools: Dict[str, ToolSpec] = Field(default_factory=dict)
    echo: bool = False


_WORKER_ADAPTER: TypeAdapter[WorkerMessage] = TypeAdapter(WorkerMessage)
_MAIN_ADAPTER: TypeAdapter[MainMessage] = TypeAdapter(MainMessage)


def send_message(conn: Channel, message: BaseModel) -> None:
    """Send a message as JSON bytes.

    Raises:
        TypeError, ValueError: If a value cannot be converted to JSON (for
            example a tool argument that is a function).
        OSError: If the pipe is closed.
    """
    conn.send_bytes(message.model_dump_json().encode("utf-8"))


def recv_worker_message(conn: Channel) -> WorkerMessage:
    """Read one message sent by the worker (untrusted).

    The length is limited and the content is validated.

    Raises:
        pydantic.ValidationError: If the bytes are not valid JSON or do not
            match one of the messages of the worker.
        EOFError: If the pipe is closed.
        OSError: If the message is longer than MAX_MESSAGE_BYTES (the
            connection can then no longer be read).
    """
    return _WORKER_ADAPTER.validate_json(conn.recv_bytes(MAX_MESSAGE_BYTES))


def recv_main_message(conn: Channel) -> MainMessage:
    """Read one message sent by the main process (trusted).

    Raises:
        pydantic.ValidationError: If the message is invalid.
        EOFError: If the pipe is closed.
    """
    return _MAIN_ADAPTER.validate_json(conn.recv_bytes())


def recv_init_message(conn: Channel) -> InitMsg:
    """Read the first message sent by the main process (trusted).

    It carries what the worker needs to start: configuration, tools and
    echo mode.

    Raises:
        pydantic.ValidationError: If the message is not a valid InitMsg.
        EOFError: If the pipe is closed.
    """
    return InitMsg.model_validate_json(conn.recv_bytes())
