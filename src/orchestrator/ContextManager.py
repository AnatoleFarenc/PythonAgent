from .Prompt import Prompt
from .config import PATH_SYS_PROMPT
from ..models_agent.ExecutionResult import ExecutionResult

from openai.types.chat import ChatCompletionMessageParam
from typing import Optional, List
import re


class ContextManager:
    def __init__(self, sys_promt_template_path: Optional[str] = None) -> None:
        self.sys_path = (
            sys_promt_template_path if sys_promt_template_path
            else PATH_SYS_PROMPT
        )
        self.__sys_prompt = Prompt()

    def build_context(
            self,
            manual: str,
            context: List[ChatCompletionMessageParam],
            responce: Optional[str],
            querry: Optional[str]
    ) -> List[ChatCompletionMessageParam]:
        if not context:
            self.__sys_prompt = Prompt(
                self.sys_path,
                {"tools_manual": manual}
            )
            sys: ChatCompletionMessageParam = {
                "role": "system",
                "content": self.sys_prompt
            }
            context.append(sys)
        if querry:
            user: ChatCompletionMessageParam = {
                "role": "user",
                "content": querry
            }
            context.append(user)
        if responce:
            responce = keep_first_block(responce)
            assistant: ChatCompletionMessageParam = {
                "role": "assistant",
                "content": responce
            }
            context.append(assistant)
        return context

    @staticmethod
    def sandbox_output_analyse(sandbox_output: ExecutionResult) -> str:
        """Format raw sandbox execution results into agent observation text."""
        if sandbox_output.timed_out:
            if sandbox_output.stdout:
                return (f"Execution timed out (infinite loop/slow algorithm). "
                        f"Current output:\n{sandbox_output.stdout}")
            return "Execution timed out (infinite loop/slow algorithm)."
        if sandbox_output.error:
            if sandbox_output.stdout:
                return (f"Error:\n{sandbox_output.error}\n"
                        f"Current output:\n{sandbox_output.stdout}")
            return f"Error:\n{sandbox_output.error}"
        if sandbox_output.final_answer is not None:
            return f"Final answer submitted:\n{sandbox_output.final_answer}"
        if sandbox_output.truncated:
            return (f"The output was truncated.\n"
                    f"Current output:\n{sandbox_output.stdout}")
        if sandbox_output.stdout:
            return sandbox_output.stdout
        return ("Code executed successfully with no output return. "
                "Remember to use print() to see results.")

    @property
    def sys_prompt(self) -> str:
        return self.__sys_prompt.prompt

    @sys_prompt.setter
    def sys_prompt(self, n_sys_prompt) -> None:
        self.__sys_prompt.prompt = n_sys_prompt


def keep_first_block(answer: str) -> str:
    """Cut an LLM answer right after its first fenced code block."""
    # Only the first block is executed: any hallucinated follow-up steps
    # would bloat the context and mislead the model on the next turn
    match = re.search(r"```[^\n]*\n.*?```", answer, re.DOTALL)
    if not match:
        return answer
    return answer[:match.end()]
