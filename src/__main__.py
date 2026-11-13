"""Entry point: build the index and print the recall@k on the dataset."""

from .HybridIndex import HybridIndex
from .recall import find_recall_metric


DATA_SET_PATH = "data/datasets/dataset_code_public.json"


def main() -> None:
    """Index the code base and evaluate recall@3, @5 and @10."""
    index = HybridIndex.from_folder("vllm-0.10.1", "BAAI/bge-m3")

    k3 = find_recall_metric(index, DATA_SET_PATH, 3)
    k5 = find_recall_metric(index, DATA_SET_PATH, 5)
    k10 = find_recall_metric(index, DATA_SET_PATH, 10)
    print(f"k3: {k3} -> {k3 * 100}")
    print(f"k5: {k5} -> {k5 * 100}")
    print(f"k10: {k10} -> {k10 * 100}")


if __name__ == "__main__":
    main()
