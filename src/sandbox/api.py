"""Public sandbox API: run LLM generated code in a worker process."""
from ..models_agent.ExecutionResult import ExecutionResult
from ..models_agent.SandboxConfig import SandboxConfig
from ..models_agent.IsolationConfig import IsolationConfig
from .restricted_sandbox import truncke_output
from .McpBrige import McpBridge
from ..models_agent.ToolSpec import ToolSpec
from .manual_manager import build_manual
from .config import (
    START_WORK_TIMEOUT, READ_MESSAGE_TIMEOUT,
    PROJECT_ROOT, RED, RESET
)
from .container import (
    DOCKER_CLI_ENV, build_worker_command, check_workspace,
    kill_container, visible_directories
)
from .protocol import (
    CodeMsg, StdoutMsg, ToolCallMsg,
    ResultMsg, ToolResultMsg, ReadyMsg,
    StopMsg, InitMsg,
    recv_worker_message, send_message
)
from .channel import StreamChannel

from typing import Any, Dict, List
from pydantic import ValidationError
import time
import concurrent.futures
import os
import subprocess
import sys
import uuid


class Sandbox:
    """Execute LLM generated code in a restricted environment.

    The code runs in a separate worker process. Variables persist between
    calls to execute() until the worker is restarted (timeout or crash).
    MCP tool calls made by the code are forwarded by the main process to
    the MCP server.
    """

    def __init__(self,
                 mcp_command: str | None = None,
                 sandbox_config: SandboxConfig | None = None,
                 echo: bool = False,
                 mcp_env: Dict[str, str] | None = None,
                 isolation: IsolationConfig | None = None
                 ) -> None:
        """Connect to the MCP server and start the worker process.

        Args:
            mcp_command: Command line that starts the MCP server.
            sandbox_config: Sandbox configuration, the default one if None.
            echo: True to print the value of expression statements.
            mcp_env: Environment of the MCP server.
            isolation: Isolation of the worker, the default one (gVisor
                container, no workspace) if None.

        Raises:
            ValueError: If the workspace cannot be mounted safely.
            McpConnectionError: If the MCP server cannot be reached.
            RuntimeError: If the worker process fails to start.
        """
        self._echo = echo
        self._isolation = isolation or IsolationConfig()
        if self._isolation.workspace is not None:
            # Fail early, before connecting to MCP or starting anything.
            self._isolation = self._isolation.model_copy(update={
                "workspace": check_workspace(self._isolation.workspace)
            })
        if self._isolation.backend == "local":
            print(f"{RED}WARNING: local backend, the code is NOT isolated "
                  f"from this machine{RESET}", file=sys.stderr)
        # allowed_directories always matches what is really mounted.
        self.__config = (sandbox_config or SandboxConfig()).model_copy(
            update={"allowed_directories":
                    visible_directories(self._isolation)}
        )
        self._tools_list: Dict[str, ToolSpec] = {}
        self._mcp_bridge: McpBridge | None = None

        if mcp_command is not None:
            self._mcp_bridge = McpBridge(
                mcp_command,
                self.__config.max_tool_time_seconds,
                mcp_env
            )
            try:
                self._mcp_bridge.connect()
                self._tools_list = self._mcp_bridge.get_tools_list()
            except BaseException:
                self._mcp_bridge.close()
                raise
        try:
            self._start_work_process()
        except BaseException:
            if self._mcp_bridge is not None:
                self._mcp_bridge.close()
            raise

    def _start_work_process(self) -> None:
        """Start the worker program and wait for its "ready" message.

        Two pipes are created: the worker gets one end of each as its
        stdin/stdout, the host keeps the other ends in a StreamChannel.
        The worker starts with an empty environment, in its own session,
        and receives its configuration in an InitMsg.

        Raises:
            RuntimeError: If the worker does not answer within
                START_WORK_TIMEOUT seconds or dies during startup.
        """
        to_worker_r, to_worker_w = os.pipe()        # hôte -> worker
        from_worker_r, from_worker_w = os.pipe()    # worker -> hôte
        self._container = f"sandbox-{uuid.uuid4().hex[:12]}"
        docker = self._isolation.backend == "docker"
        try:
            self._process = subprocess.Popen(
                build_worker_command(self._container, self._isolation,
                                     self.__config.max_memory_mb),
                stdin=to_worker_r,
                stdout=from_worker_w,
                stderr=None,
                cwd=PROJECT_ROOT,
                env=DOCKER_CLI_ENV if docker else {},
                close_fds=True,
                start_new_session=True,
            )
        finally:
            os.close(to_worker_r)
            os.close(from_worker_w)
        self._conn = StreamChannel(
            from_worker_r, to_worker_w,
            read_timeout=READ_MESSAGE_TIMEOUT
        )
        try:
            send_message(self._conn, InitMsg(
                            config=self.__config,
                            tools=self._tools_list,
                            echo=self._echo
                        ))
            if not self._conn.poll(START_WORK_TIMEOUT):
                raise RuntimeError("The worker process failed to start")
            message = recv_worker_message(self._conn)
            if not isinstance(message, ReadyMsg):
                raise RuntimeError("unexpected first message")
        except (EOFError, OSError) as e:
            self._kill_worker()
            raise RuntimeError(
                "The worker process died during startup "
                f"(exit code {self._process.returncode}), "
                "see its traceback above"
            ) from e
        except BaseException:
            self._kill_worker()
            raise

    def _kill_worker(self) -> None:
        """Kill the worker at once and close the channel.

        With Docker, the process started by Popen is only the docker
        client: the container itself is killed by its name.
        """
        if self._isolation.backend == "docker":
            kill_container(self._container)
        self._process.kill()
        self._process.wait()
        self._conn.close()

    def _stop_worker(self) -> None:
        """Ask the worker to stop, kill it if needed and close the channel."""
        try:
            send_message(self._conn, StopMsg())
        except (EOFError, OSError):
            pass
        try:
            self._process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            self._kill_worker()
        self._conn.close()

    def _restart_work_process(self) -> None:
        """Replace the worker by a new one with an empty namespace."""
        self._stop_worker()
        self._start_work_process()

    def execute(self, code: str) -> ExecutionResult:
        """Execute code in the worker and return its result.

        Stdout is collected while the code runs, and tool calls are
        forwarded to the MCP server. Time spent in tools is not counted in
        the timeout. On timeout or crash the worker is restarted and all
        variables are lost.

        Args:
            code: Python source code to execute.

        Returns:
            The execution result, with stdout truncated to max_len_output.
        """
        timed_out = False
        chunks: List[str] = []
        deadline = time.monotonic() + self.__config.max_execution_time_seconds
        tool_truncated = False
        protocol_error = False
        try:
            send_message(self._conn, CodeMsg(code=code))
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._conn.poll(remaining):
                    timed_out = True
                    break
                message = recv_worker_message(self._conn)
                if isinstance(message, StdoutMsg):
                    chunks.append(message.text)
                elif isinstance(message, ToolCallMsg):
                    start_time = time.monotonic()
                    response = ToolResultMsg()
                    try:
                        if self._mcp_bridge is None:
                            raise RuntimeError(
                                "No MCP server connected"
                            )
                        response.output = self._mcp_bridge.call_tool(
                            message.name,
                            message.args
                        )
                        response.output, cut = truncke_output(
                            response.output,
                            self.__config.max_len_tool_output
                        )
                        tool_truncated = tool_truncated or cut
                    except concurrent.futures.TimeoutError:
                        response.error = True
                        response.output = (
                            f"Tool '{message.name}' did not answer within "
                            f"{self.__config.max_tool_time_seconds} s"
                        )
                    except Exception as e:
                        response.error = True
                        response.output = (
                            f"{type(e).__name__}: {e}"
                        )
                    send_message(self._conn, response)
                    deadline += time.monotonic() - start_time

                elif isinstance(message, ResultMsg):
                    output = "".join(chunks)
                    output, is_truncate = truncke_output(
                        output, self.__config.max_len_output
                    )
                    return (
                        ExecutionResult(
                            stdout=output,
                            error=message.error,
                            final_answer=message.final_answer,
                            truncated=is_truncate or tool_truncated
                        )
                    )
                else:
                    protocol_error = True
                    break

        except ValidationError:
            protocol_error = True
        except (EOFError, OSError):
            pass
        except KeyboardInterrupt:
            self.close()
            raise
        self._restart_work_process()
        output = "".join(chunks)
        output, is_truncate = truncke_output(
            output, self.__config.max_len_output
        )
        if protocol_error:
            return ExecutionResult(
                stdout=output,
                error=("The execution environment sent an invalid message "
                       "and was restarted: all variables are lost."),
                truncated=is_truncate or tool_truncated
            )
        if timed_out:
            return ExecutionResult(
                stdout=output,
                error=(
                        f"TimeoutError: your code did not finish within "
                        f"{self.__config.max_execution_time_seconds} "
                        "seconds and was stopped "
                        "(possible infinite loop or too heavy computation). "
                        "The execution environment was restarted: all "
                        "variables and imports "
                        "are lost, you must redefine them. "
                    ),
                timed_out=True,
                truncated=is_truncate or tool_truncated
                    )
        return ExecutionResult(
                stdout=output,
                error=(
                        "The execution environment crashed "
                        "(e.g. os._exit). "
                        "A new empty environment was "
                        "started: all variables are lost."
                ),
                truncated=is_truncate or tool_truncated
            )

    def get_manual(self) -> str:
        """Return the sandbox manual"""
        return build_manual(self.__config, self._tools_list)

    def close(self) -> None:
        """Stop the worker process and close the MCP connection."""
        self._stop_worker()
        if self._mcp_bridge is not None:
            self._mcp_bridge.close()

    def __enter__(self) -> "Sandbox":
        """Return the sandbox itself for use in a with statement."""
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Close the sandbox when leaving the with statement."""
        self.close()
