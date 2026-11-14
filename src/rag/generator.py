"""Answer generation: retrieved chunks + question -> LLM answer (vLLM).

The ``HybridIndex`` finds the relevant chunks, they are packed into a
chat prompt (within the model's context budget) and a small instruct
model served by vLLM writes the answer. Questions are generated in one
batched ``LLM.generate`` call, which is where vLLM is fastest.
"""

import os
from dataclasses import dataclass
from typing import Any, List

from model import ChunkModel
from .indexer.HybridIndex import HybridIndex
from .chunking.chunking_tools_python import read_chunk
from .config import (
    LLM_ENFORCE_EAGER, LLM_GPU_MEMORY_UTILIZATION, LLM_MAX_MODEL_LEN,
    LLM_MAX_TOKENS, LLM_MODEL_NAME, LLM_TEMPERATURE, LLM_TOP_P,
)

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams  # noqa: E402


SYSTEM_PROMPT = (
    "You are an expert on the vLLM code base. Answer the user's question "
    "using only the code chunks below. Be concise and precise, cite the "
    "relevant function, class or file names. If the chunks do not contain "
    "the answer, say so instead of guessing."
)


@dataclass(slots=True)
class Answer:
    """Generated answer and the chunks it was based on.

    Attributes:
        question: The question asked.
        text: The model's answer.
        chunks: Chunks actually placed in the prompt, most relevant first.
    """

    question: str
    text: str
    chunks: List[ChunkModel]


def format_chunk(i: int, chunk: ChunkModel) -> str:
    """Render one chunk for the prompt, with its location as a header."""
    where = f"{chunk.file}:{chunk.start_line}-{chunk.end_line}"
    scope = f" ({chunk.paths})" if chunk.paths else ""
    return f"--- Chunk {i} | {where}{scope} ---\n{read_chunk(chunk)}"


class RagGenerator:
    """Retrieval-augmented answer generator backed by vLLM.

    Attributes:
        index: Index used to retrieve the chunks.
        llm: vLLM engine.
        params: Sampling parameters used for every generation.
    """

    def __init__(
            self, index: HybridIndex,
            model_name: str = LLM_MODEL_NAME) -> None:
        """Load the LLM on the GPU.

        Args:
            index: Index used to retrieve the chunks.
            model_name: Hugging Face name of the instruct model.
        """
        self.index = index
        self.llm = LLM(
            model=model_name,
            max_model_len=LLM_MAX_MODEL_LEN,
            gpu_memory_utilization=LLM_GPU_MEMORY_UTILIZATION,
            enforce_eager=LLM_ENFORCE_EAGER,
            disable_log_stats=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.params = SamplingParams(
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            max_tokens=LLM_MAX_TOKENS,
        )

    def _n_tokens(self, text: str) -> int:
        """Number of tokens of *text* for the loaded model."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _render(self, question: str, context: str) -> str:
        """Apply the model's chat template (thinking disabled for Qwen3)."""
        messages: List[Any] = [
            {"role": "system",
             "content": f"{SYSTEM_PROMPT}\n\nChunks:\n{context}"},
            {"role": "user", "content": question},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        assert isinstance(prompt, str)
        return prompt

    def build_prompt(
            self, question: str,
            chunks: List[ChunkModel]) -> tuple[str, List[ChunkModel]]:
        """Build the prompt, keeping as many chunks as the context allows.

        Chunks are added in relevance order until the prompt would leave
        less than ``LLM_MAX_TOKENS`` tokens for the answer; the remaining
        (less relevant) chunks are dropped instead of failing.

        Args:
            question: The user's question.
            chunks: Retrieved chunks, most relevant first.

        Returns:
            (prompt, chunks kept in the prompt).
        """
        budget = LLM_MAX_MODEL_LEN - LLM_MAX_TOKENS
        used = self._n_tokens(self._render(question, ""))
        parts: List[str] = []
        kept: List[ChunkModel] = []
        for chunk in chunks:
            part = format_chunk(len(kept) + 1, chunk)
            cost = self._n_tokens(part) + 2
            if used + cost > budget:
                continue
            parts.append(part)
            kept.append(chunk)
            used += cost
        return self._render(question, "\n\n".join(parts)), kept

    def answer_many(self, questions: List[str], k: int = 5) -> List[Answer]:
        """Retrieve and answer several questions in one vLLM batch.

        Args:
            questions: Questions to answer.
            k: Number of chunks retrieved per question.

        Returns:
            One ``Answer`` per question, in the same order.
        """
        prompts: List[str] = []
        kept_lst: List[List[ChunkModel]] = []
        for question in questions:
            prompt, kept = self.build_prompt(
                question, self.index.search(question, k=k))
            prompts.append(prompt)
            kept_lst.append(kept)

        outputs = self.llm.generate(prompts, self.params)
        return [
            Answer(question=q, text=out.outputs[0].text.strip(), chunks=kept)
            for q, out, kept in zip(questions, outputs, kept_lst)
        ]

    def answer(self, question: str, k: int = 5) -> Answer:
        """Retrieve the *k* best chunks and answer *question*."""
        return self.answer_many([question], k)[0]
