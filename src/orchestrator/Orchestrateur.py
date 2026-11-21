from .ContextManager import ContextManager
from .config import (
    STOP_PROMPT, MAX_SERVER_RETRIES,
    LIMITS_ITERATION, PATH_SANDBOX_CONFIG, MCP_CMD,
    MAX_INPUT_TOKEN, MAX_OUTPUT_TOKEN, TIMEOUT_SEC
)
from .OrchestratorError import (
    OrchestratorError, KeyQuotaError,
    ServerError, RequestTooLargeError,
    FatalError
)
from ..sandbox.api import Sandbox
from .extractor import extract_code
from .KeyManager import KeyManager
from ..models_agent.StepMetrics import StepMetrics
from ..models_agent.SandboxConfig import load_config
from ..models_agent.ExecutionResult import ExecutionResult
from ..models_agent.SolutionOutput import SolutionOutput


from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from typing import Optional, List
import time


class Orchestrateur:
    def __init__(self, model_name: str, provider_url: str,
                 sys_prompt_template_path: Optional[str] = None) -> None:
        self.model_name = model_name
        self.provider_url = provider_url
        self.keys_manager = KeyManager()
        self.key = self.keys_manager.get_new_key()
        self.context_manager = ContextManager(sys_prompt_template_path)
        self.context: List[ChatCompletionMessageParam] = []
        self.sandbox_manual = ""

        self.step_retries = 0
        self.step_input_tokens = 0
        self.step_output_tokens = 0
        self.step_time = 0.0
        self.all_steps: list[StepMetrics] = []

    def call_api(self) -> str:
        try:
            with OpenAI(
                base_url=self.provider_url,
                api_key=self.key
            ) as client:
                start_time = time.perf_counter()
                responce = client.chat.completions.create(
                    model=self.model_name,
                    messages=self.context,
                    stop=STOP_PROMPT,
                )
                end_time = time.perf_counter()
                self.step_time = (end_time - start_time) * 1000

                if responce.usage:
                    self.step_input_tokens = responce.usage.prompt_tokens
                    self.step_output_tokens = responce.usage.completion_tokens

                content = responce.choices[0].message.content or ""
                if content.count("```") % 2 == 1:
                    content = f"{content.rstrip()}\n```"
                return content
        except Exception as e:
            raise OrchestratorError.from_exeption(e) from e

    def query_llm(self) -> str:
        self.step_retries = 0
        self.step_input_tokens = 0
        self.step_output_tokens = 0
        self.step_time = 0
        server_retries = 0

        while True:
            try:
                content = self.call_api()
                return content
            except KeyQuotaError as e:
                self.step_retries += 1
                if e.retry_after:
                    time.sleep(e.retry_after)
                else:
                    self.key = self.keys_manager.get_new_key()
            except ServerError:
                server_retries += 1
                self.step_retries += 1
                if server_retries > MAX_SERVER_RETRIES:
                    raise
            except RequestTooLargeError:
                self.step_retries += 1
                # reduct context
                pass
            except FatalError:
                raise

    def complete_task(self, task: str):
        solved = False
        start_time = time.time()
        current_iteration = 0
        solution = ""
        error = None
        config = (load_config(PATH_SANDBOX_CONFIG).sandbox
                  if PATH_SANDBOX_CONFIG else None)

        with Sandbox(MCP_CMD, config) as sandbox:
            self.sandbox_manual = sandbox.get_manual()
            self.context = self.context_manager.build_context(
                manual=self.sandbox_manual,
                context=self.context,
                responce=None,
                querry=task
            )
            while current_iteration < LIMITS_ITERATION and not solved:
                current_iteration += 1
                try:
                    responce = self.query_llm()
                except OrchestratorError as e:
                    error = str(e)
                    break
                self.context = self.context_manager.build_context(
                    self.sandbox_manual,
                    context=self.context,
                    responce=responce,
                    querry=None
                )
                sandbox_input = extract_code(responce)
                if not sandbox_input:
                    sandbox_input = ""
                    sandbox_output = (
                        "Error: No valid Python code block found. "
                        "You must wrap your code in ```python ... ```."
                    )
                else:
                    sandbox_result: ExecutionResult = (
                        sandbox.execute(sandbox_input)
                    )
                if sandbox_result.final_answer is not None:
                    solved = True
                    solution = sandbox_result.final_answer
                else:
                    sandbox_output = (
                        self.context_manager.sandbox_output_analyse(
                            sandbox_result
                        )
                    )
                    self.context = self.context_manager.build_context(
                        self.sandbox_manual,
                        context=self.context,
                        responce=None,
                        querry=sandbox_output
                    )
                step = StepMetrics(
                    step=current_iteration,
                    input_tokens=self.step_input_tokens,
                    output_tokens=self.step_output_tokens,
                    request_time_ms=self.step_time,
                    api_url=self.provider_url,
                    model_name=self.model_name,
                    llm_output=responce,
                    sandbox_input=sandbox_input,
                    sandbox_output=sandbox_output,
                    retries=self.step_retries
                )
                self.all_steps.append(step)

                stop, error = self.stop_loop(
                    iterations=len(self.all_steps),
                    input_tok=sum(s.input_tokens for s in self.all_steps),
                    output_tok=sum(s.output_tokens for s in self.all_steps),
                    time=time.time() - start_time,
                    solved=solved
                )
                if stop:
                    break

            total_time = time.time() - start_time
            return SolutionOutput(
                success=solved,
                solution=solution,
                iterations=len(self.all_steps),
                total_requests=len(self.all_steps)
                + sum(s.retries for s in self.all_steps),
                total_input_tokens=sum(
                    s.input_tokens for s in self.all_steps
                ),
                total_output_tokens=sum(
                    s.output_tokens for s in self.all_steps
                ),
                total_time_seconds=total_time,
                steps=self.all_steps,
                system_prompt=self.context_manager.sys_prompt,
                error=error if not solved else None,
            )

    def stop_loop(self, iterations: int,
                  input_tok: int,
                  output_tok: int,
                  time: float,
                  solved: bool) -> tuple[bool, Optional[str]]:
        """Evaluate task termination conditions against configured limits."""
        if solved:
            return True, None
        if iterations >= LIMITS_ITERATION:
            return True, "Max iterations limit reached"
        if input_tok >= MAX_INPUT_TOKEN:
            return True, "Max input tokens exceeded"
        if output_tok >= MAX_OUTPUT_TOKEN:
            return True, "Max output tokens exceeded"
        if time >= TIMEOUT_SEC:
            return True, "Execution timeout exceeded"

        return False, None
