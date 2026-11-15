"""Session: the agent loop with a scripted LLM and a fake sandbox."""
from src.agent.exceptions import RequestTooLargeError, ServerError
from src.agent.llm import LLMReply
from src.agent.observation import NO_CODE_MESSAGE
from src.agent.session import (KEPT_OBSERVATION_CHARS, Session,
                               estimate_tokens)
from src.models_agent.ExecutionResult import ExecutionResult

from typing import Callable, List
import pytest


def block(code: str) -> str:
    return f"Thought: step\nCode:\n```python\n{code}\n```"


class ScriptedLLM:
    """Returns the scripted answers in order, records what it was sent."""

    def __init__(self, answers: List[object]) -> None:
        self.answers = list(answers)
        self.calls: List[list] = []

    def complete(self, messages):
        self.calls.append([dict(m) for m in messages])
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return LLMReply(text=str(answer))


class FakeSandbox:
    """Runs nothing: maps code to results with *run*."""

    def __init__(self, run: Callable[[str], ExecutionResult]) -> None:
        self.run = run
        self.codes: List[str] = []
        self.closed = False

    def execute(self, code: str) -> ExecutionResult:
        self.codes.append(code)
        return self.run(code)

    def get_manual(self) -> str:
        return "MANUAL"

    def close(self) -> None:
        self.closed = True


def default_run(code: str) -> ExecutionResult:
    if code.startswith("final_answer("):
        return ExecutionResult(final_answer=code[len("final_answer('"):-2])
    return ExecutionResult(stdout=f"ran {code}")


def make_session(answers, run=default_run, context_window=100_000,
                 max_steps=5):
    sandboxes: List[FakeSandbox] = []

    def factory():
        sandboxes.append(FakeSandbox(run))
        return sandboxes[-1]
    llm = ScriptedLLM(answers)
    session = Session(llm, factory, context_window, 100, max_steps)
    return session, llm, sandboxes


def test_loop_runs_until_final_answer():
    session, llm, boxes = make_session([
        block("print(grep('x'))"), block("final_answer('done')")])
    answer = session.ask("where is x?")
    assert answer.finished and answer.text == "done"
    assert boxes[0].codes == ["print(grep('x'))", "final_answer('done')"]
    # Second call saw the first step and its observation.
    second = llm.calls[1]
    assert second[-1] == {"role": "user",
                          "content": "Observation:\nran print(grep('x'))"}
    assert "MANUAL" in second[0]["content"]


def test_finished_request_is_compacted():
    session, _, _ = make_session([
        block("print(1)"), block("final_answer('one')"),
        block("final_answer('two')")])
    session.ask("q1")
    roles = [m["role"] for m in session.messages]
    assert roles == ["system", "user", "assistant"]
    assert "final_answer('one')" in session.messages[-1]["content"]
    session.ask("q2")
    assert [m["content"] for m in session.messages[1:4:2]] == ["q1", "q2"]


def test_answer_without_code_gets_a_correction():
    session, llm, boxes = make_session([
        "I think it is fine.", block("final_answer('ok')")])
    assert session.ask("q").text == "ok"
    assert llm.calls[1][-1]["content"] == f"Observation:\n{NO_CODE_MESSAGE}"
    assert boxes[0].codes == ["final_answer('ok')"]


def test_only_first_block_is_kept_in_history():
    two_blocks = block("print(1)") + "\n" + block("final_answer('cheat')")
    session, llm, boxes = make_session([two_blocks, block("final_answer('ok')")])
    session.ask("q")
    assert boxes[0].codes[0] == "print(1)"
    assert "cheat" not in llm.calls[1][-2]["content"]


def test_step_limit_keeps_steps_for_continue():
    session, _, _ = make_session([block("print(1)")] * 2, max_steps=2)
    answer = session.ask("q")
    assert not answer.finished and "continue" in answer.text
    assert len(answer.steps) == 2
    assert len(session.messages) == 1 + 1 + 2 * 2


def test_on_step_is_called():
    seen = []
    session, _, _ = make_session([block("print(1)"), block("final_answer('x')")])
    session.ask("q", on_step=lambda s: seen.append(s.number))
    assert seen == [1, 2]


def test_provider_error_before_any_step_forgets_the_question():
    session, _, _ = make_session([ServerError("down")])
    with pytest.raises(ServerError):
        session.ask("q")
    assert len(session.messages) == 1


def test_request_too_large_trims_old_observations_then_retries():
    big = "x" * 5000
    session, llm, _ = make_session(
        [block("print(1)"), block("print(2)"),
         RequestTooLargeError("too big"), block("final_answer('ok')")],
        run=lambda c: (default_run(c) if c.startswith("final")
                       else ExecutionResult(stdout=big)))
    assert session.ask("q").text == "ok"
    retried = llm.calls[-1]
    first_obs = retried[3]["content"]
    assert len(first_obs) < KEPT_OBSERVATION_CHARS + 100
    assert retried[-1]["content"].endswith(big)  # latest kept whole


def test_context_budget_drops_oldest_requests():
    long_q = "q" * 3000
    session, llm, _ = make_session(
        [block("final_answer('a')"), block("final_answer('b')"),
         block("final_answer('c')")],
        context_window=100 + 512 + 2500)
    for _ in range(3):
        session.ask(long_q)
    last_call = llm.calls[-1]
    assert estimate_tokens(last_call) <= session.context_budget
    assert last_call[-1]["content"] == long_q  # current request kept


def test_interrupt_during_execute_marks_sandbox_for_restart():
    def run(code):
        raise KeyboardInterrupt
    session, _, boxes = make_session([block("while True: pass")], run=run)
    with pytest.raises(KeyboardInterrupt):
        session.ask("q")
    assert session.sandbox_needs_restart
    # History stays valid: the pending step got an observation.
    assert session.messages[-1]["role"] == "user"
    session.restart_sandbox()
    assert boxes[0].closed and len(boxes) == 2
    assert not session.sandbox_needs_restart


def test_reset_keeps_only_system_prompt():
    session, _, _ = make_session([block("final_answer('a')")])
    session.ask("q")
    session.reset()
    assert [m["role"] for m in session.messages] == ["system"]


def test_repeated_code_is_not_rerun():
    session, llm, boxes = make_session([
        block("print(1)"), block("print(1)"), block("final_answer('ok')")])
    answer = session.ask("q")
    assert boxes[0].codes == ["print(1)", "final_answer('ok')"]
    assert "Not executed" in answer.steps[1].observation


def test_last_step_notice():
    session, llm, _ = make_session(
        [block("print(1)"), block("print(2)"), block("print(3)")],
        max_steps=3)
    session.ask("q")
    assert "last one" in llm.calls[2][-1]["content"]
    assert "last one" not in llm.calls[1][-1]["content"]
