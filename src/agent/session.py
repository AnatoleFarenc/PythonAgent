"""Conversation with the coding assistant.

Agent-Smith's ``complete_task`` ran one graded task in a fresh sandbox
and returned metrics. Here a ``Session`` keeps the sandbox and the
conversation open: each ``ask()`` runs the same inner loop

    LLM -> extract code -> sandbox.execute -> observation -> LLM ...

until the model calls ``final_answer``, whose text is the reply to the
user. The conversation then goes on with the next user message.
"""
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol, Sequence

from openai.types.chat import ChatCompletionMessageParam

from ..models_agent.ExecutionResult import ExecutionResult
from .exceptions import ProviderRequestError, RequestTooLargeError
from .extractor import extract_code, keep_first_block
from .llm import LLMReply
from .observation import NO_CODE_MESSAGE, format_observation
from .prompts import build_system_prompt

DEFAULT_MAX_STEPS = 15
# Characters kept from an old observation when the context is trimmed.
KEPT_OBSERVATION_CHARS = 300
TRIM_NOTICE = "\n[... old output removed to fit the context ...]"
# Rough token estimate: code and logs average ~3 characters per token.
CHARS_PER_TOKEN = 3
# Tokens kept free on top of the answer budget (estimate error, chat
# template overhead).
CONTEXT_MARGIN_TOKENS = 512
INTERRUPTED_NOTICE = "[interrupted by the user]"
REPEAT_MESSAGE = (
    "Not executed: this is exactly the code of your previous step, its "
    "result is above. If you have the answer, call final_answer(...) now; "
    "otherwise do something different.")
LAST_STEP_NOTICE = (
    "\n\n[Next step is your last one: call final_answer(...) with what you "
    "found, even if incomplete.]")

Message = ChatCompletionMessageParam


class ChatModel(Protocol):
    """What the session needs from the LLM (LLMClient, or a fake)."""

    def complete(self, messages: Sequence[Message]) -> LLMReply: ...


class SandboxLike(Protocol):
    """What the session needs from the sandbox (Sandbox, or a fake)."""

    def execute(self, code: str) -> ExecutionResult: ...
    def get_manual(self) -> str: ...
    def close(self) -> None: ...


@dataclass(slots=True)
class Step:
    """One LLM -> sandbox cycle, for display and debugging."""

    number: int
    llm_output: str
    code: str
    observation: str
    reply: LLMReply


@dataclass(slots=True)
class Answer:
    """Result of one ask().

    Attributes:
        text: Reply to the user (final_answer), or why the loop stopped.
        finished: False if the step limit was hit before final_answer.
        steps: Every step of this request.
    """

    text: str
    finished: bool
    steps: List[Step] = field(default_factory=list)


def is_request(message: Message) -> bool:
    """True for a message typed by the user (not a sandbox observation)."""
    return (message["role"] == "user"
            and not str(message.get("content") or "")
            .startswith("Observation:"))


def estimate_tokens(messages: Sequence[Message]) -> int:
    """Cheap token estimate of a message list."""
    return sum(len(str(m.get("content") or "")) for m in messages) \
        // CHARS_PER_TOKEN


class Session:
    """A conversation with the assistant over one sandbox.

    Attributes:
        llm: Model used.
        sandbox: Current sandbox (replaced by restart_sandbox()).
        messages: Full chat history sent to the model, system prompt first.
        max_steps: LLM calls allowed per ask().
        context_budget: Prompt tokens allowed (context minus answer).
    """

    def __init__(self, llm: ChatModel,
                 sandbox_factory: Callable[[], SandboxLike],
                 context_window: int, max_output_tokens: int,
                 max_steps: int = DEFAULT_MAX_STEPS) -> None:
        """Start the sandbox and build the system prompt from its manual."""
        self.llm = llm
        self._sandbox_factory = sandbox_factory
        self.sandbox = sandbox_factory()
        self.max_steps = max_steps
        self.context_budget = (context_window - max_output_tokens
                               - CONTEXT_MARGIN_TOKENS)
        self.system_prompt = build_system_prompt(self.sandbox.get_manual())
        self.messages: List[Message] = [
            {"role": "system", "content": self.system_prompt}]
        # Index of the user message of the request in progress: messages
        # before it belong to finished requests and may be dropped.
        self._request_start = 1
        # Set when Ctrl+C hit during execute(): Sandbox.execute() closes
        # the sandbox on KeyboardInterrupt, it must be restarted.
        self.sandbox_needs_restart = False
        # Set by the CLI when the code index could not be built.
        self.index_error: Optional[str] = None

    # --- conversation ----------------------------------------------------

    def ask(self, user_message: str,
            on_step: Optional[Callable[[Step], None]] = None) -> Answer:
        """Run the agent loop on one user message.

        Args:
            user_message: What the user typed.
            on_step: Called after each step (live display).

        Returns:
            The answer; when ``finished`` is False the intermediate steps
            stay in the history, so "continue" resumes the work.

        Raises:
            ProviderRequestError: The LLM could not be reached / used.
            KeyboardInterrupt: Re-raised after noting it in the history.
        """
        self._request_start = len(self.messages)
        self.messages.append({"role": "user", "content": user_message})
        steps: List[Step] = []
        try:
            for number in range(1, self.max_steps + 1):
                reply = self._query()
                # Only the executed part stays in the context: anything
                # after the first block is a hallucinated next step.
                kept = keep_first_block(reply.text)
                self.messages.append({"role": "assistant", "content": kept})
                code = extract_code(reply.text)
                final: Optional[str] = None
                if code is None:
                    code = ""
                    observation = NO_CODE_MESSAGE
                elif steps and code.strip() == steps[-1].code.strip():
                    # Small models loop on the same call: don't re-run it,
                    # push them to conclude or change approach.
                    observation = REPEAT_MESSAGE
                else:
                    try:
                        result = self.sandbox.execute(code)
                    except KeyboardInterrupt:
                        self.sandbox_needs_restart = True
                        raise
                    final = result.final_answer
                    observation = ("Final answer submitted."
                                   if final is not None
                                   else format_observation(result))
                step = Step(number, reply.text, code, observation, reply)
                steps.append(step)
                if on_step is not None:
                    on_step(step)
                if final is not None:
                    self._compact_request(final_step=kept)
                    return Answer(final, True, steps)
                if number == self.max_steps - 1:
                    observation += LAST_STEP_NOTICE
                self.messages.append(
                    {"role": "user",
                     "content": f"Observation:\n{observation}"})
        except KeyboardInterrupt:
            self._close_dangling_step(INTERRUPTED_NOTICE)
            raise
        except ProviderRequestError:
            if len(self.messages) - 1 == self._request_start:
                # Nothing happened yet: forget the question, the user can
                # simply ask again.
                self.messages.pop()
            raise
        return Answer(
            f"Stopped after {self.max_steps} steps without a final answer. "
            "Say \"continue\" to let the assistant go on.", False, steps)

    def reset(self) -> None:
        """Forget the conversation (the sandbox and its variables stay)."""
        del self.messages[1:]
        self._request_start = 1

    def restart_sandbox(self) -> None:
        """Replace the sandbox, e.g. after Ctrl+C killed it mid-execution."""
        try:
            self.sandbox.close()
        except Exception:
            pass
        self.sandbox = self._sandbox_factory()
        self.sandbox_needs_restart = False

    def close(self) -> None:
        """Close the sandbox (and the MCP server it started)."""
        self.sandbox.close()

    # --- LLM call and context --------------------------------------------

    def _query(self) -> LLMReply:
        """Call the LLM, shrinking the context while it does not fit."""
        self._fit_context()
        while True:
            try:
                return self.llm.complete(self.messages)
            except RequestTooLargeError:
                if not self._trim_once():
                    raise

    def _fit_context(self) -> None:
        """Trim the history until its estimate fits the budget."""
        while (estimate_tokens(self.messages) > self.context_budget
               and self._trim_once()):
            pass

    def _trim_once(self) -> bool:
        """Free some context; False when nothing more can go.

        1. Shorten the oldest long observation (Agent-Smith's
           trim_context), never the latest one.
        2. Then drop the oldest finished request (user message + reply).
        The system prompt and the current user message always stay.
        """
        trimmed_size = KEPT_OBSERVATION_CHARS + len(TRIM_NOTICE)
        for i in range(1, len(self.messages) - 1):
            message = self.messages[i]
            content = message.get("content")
            if (message["role"] == "user" and isinstance(content, str)
                    and content.startswith("Observation:")
                    and len(content) > trimmed_size):
                self.messages[i] = {
                    "role": "user",
                    "content": content[:KEPT_OBSERVATION_CHARS] + TRIM_NOTICE,
                }
                return True
        if self._request_start > 1:
            # The oldest request runs from index 1 to the next user
            # message that is a request (not an observation).
            end = next((i for i in range(2, self._request_start)
                        if is_request(self.messages[i])),
                       self._request_start)
            del self.messages[1:end]
            self._request_start -= end - 1
            return True
        return False

    def _compact_request(self, final_step: str) -> None:
        """Replace the steps of a finished request by its final step.

        The next requests then see "user question -> assistant answer"
        instead of every search and read, which saves most of the context.
        The sandbox variables are kept, so nothing computed is lost.
        """
        del self.messages[self._request_start + 1:]
        self.messages.append({"role": "assistant", "content": final_step})

    def _close_dangling_step(self, notice: str) -> None:
        """Keep the history valid after an interruption (user/assistant
        alternation): answer a pending assistant step with *notice*."""
        if self.messages[-1]["role"] == "assistant":
            self.messages.append({"role": "user",
                                  "content": f"Observation:\n{notice}"})
