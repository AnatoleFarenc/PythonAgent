"""Entry point: build the index and answer one test question."""

from .rag.config import AMBEDING_MODEL, TEST_RECALL_FOLDER
from .rag.indexer.HybridIndex import HybridIndex
from .rag.generator import RagGenerator


def main() -> None:
    """Index the code base and answer a test question with the LLM."""
    index = HybridIndex.from_folder(TEST_RECALL_FOLDER, AMBEDING_MODEL)
    answer = RagGenerator(index).answer(
        "What activation formats does the fused batched MoE layer return "
        "in vLLM?", k=5)
    print("\n\n\n")
    print(answer.text)


if __name__ == "__main__":
    main()
