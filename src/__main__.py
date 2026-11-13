"""Entry point: build the index and answer one test question."""

from .HybridIndex import HybridIndex
from .generator import RagGenerator


def main() -> None:
    """Index the code base and answer a test question with the LLM."""
    index = HybridIndex.from_folder("vllm-0.10.1", "BAAI/bge-m3")
    answer = RagGenerator(index).answer(
        "What activation formats does the fused batched MoE layer return "
        "in vLLM?", k=5)
    print("\n\n\n")
    print(answer.text)


if __name__ == "__main__":
    main()
