"""Turn a sandbox result into the observation shown to the model.

From Agent-Smith's ``Orchestrator.sandbox_analysis``.
"""
from ..models_agent.ExecutionResult import ExecutionResult

NO_CODE_MESSAGE = (
    "Error: No valid Python code block found. "
    "You must wrap your code in ```python ... ```."
)


def format_observation(result: ExecutionResult) -> str:
    """Describe what the code printed, raised or hit (timeout, size)."""
    if result.timed_out:
        if result.stdout:
            return (f"Execution timed out (infinite loop/slow algorithm). "
                    f"Current output:\n{result.stdout}")
        return "Execution timed out (infinite loop/slow algorithm)."
    if result.error:
        if result.stdout:
            return (f"Error:\n{result.error}\n"
                    f"Current output:\n{result.stdout}")
        return f"Error:\n{result.error}"
    if result.truncated:
        return (f"The output was truncated.\n"
                f"Current output:\n{result.stdout}")
    if result.stdout:
        return result.stdout
    # The model forgot print(): tools return strings, nothing is shown.
    return ("Code executed successfully with no output. "
            "Remember to print() tool results to see them.")
