"""Recall metric evaluation module.

Evaluates retrieval quality of a ``HybridIndex`` against a dataset in the
official format (``rag_questions`` with ``sources`` given as file path +
character indices).
"""

import json
from functools import lru_cache
from pathlib import PurePosixPath
from typing import List, Tuple, TypedDict

from model import ChunkModel
from .HybridIndex import HybridIndex


class Source(TypedDict):
    """Expected answer location: file + character range."""

    file_path: str
    first_character_index: int
    last_character_index: int


class Question(TypedDict):
    """One entry of the ``rag_questions`` list of the dataset."""

    question_id: str
    question: str
    answer: str
    sources: List[Source]
    difficulty: str
    is_valid: bool


def check_path(answer_path: str, check_path: str) -> bool:
    """Check if two file paths point to the same file.

    The dataset may use a longer prefix than the chunks
    (e.g. ``data/raw/vllm-0.10.1/vllm/x.py`` vs ``vllm-0.10.1/vllm/x.py``),
    so paths match when one ends with the other on a path boundary.

    Args:
        answer_path: First file path.
        check_path: Second file path.

    Returns:
        True if paths refer to the same file, False otherwise.
    """
    a = PurePosixPath(answer_path).parts
    b = PurePosixPath(check_path).parts
    n = min(len(a), len(b))
    return n > 0 and a[-n:] == b[-n:]


def check_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """Calculate Intersection over Union (IoU) of two intervals.

    Args:
        a_start: Start of first interval.
        a_end: End of first interval.
        b_start: Start of second interval.
        b_end: End of second interval.

    Returns:
        IoU score between 0 and 1.
    """
    intersection = max(0, min(a_end, b_end) - max(a_start, b_start))
    if intersection == 0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start)
    return intersection / union


@lru_cache(maxsize=256)
def _read_bytes(path: str) -> bytes:
    """Read a file as bytes (cached: the same files are read many times)."""
    with open(path, "rb") as f:
        return f.read()


def chunk_char_range(chunk: ChunkModel) -> Tuple[int, int]:
    """Convert a chunk's byte offsets into character offsets.

    Chunks store byte offsets, the dataset uses character offsets; they
    differ as soon as the file contains non-ASCII characters.

    Args:
        chunk: Chunk whose range is converted.

    Returns:
        ``(start, end)`` character offsets, end exclusive.
    """
    source = _read_bytes(chunk.file)
    start = len(source[:chunk.start_byte].decode("utf-8", errors="ignore"))
    length = len(
        source[chunk.start_byte:chunk.end_byte].decode(
            "utf-8", errors="ignore")
    )
    return start, start + length


def from_root(path: str, root: str) -> str:
    """Return *path* starting at the name of the indexed folder *root*.

    ``src/rag/vllm-0.10.1/vllm/x.py`` with root ``src/rag/vllm-0.10.1``
    gives ``vllm-0.10.1/vllm/x.py``, so it can be suffix-matched against
    the dataset paths whatever the location of the indexed folder.

    Args:
        path: Path of a chunk's file, inside *root*.
        root: Indexed folder ("" leaves *path* unchanged).

    Returns:
        The trimmed path.
    """
    skip = len(PurePosixPath(root).parts) - 1
    return str(PurePosixPath(*PurePosixPath(path).parts[max(skip, 0):]))


def source_found(
    source: Source, retrieved: List[ChunkModel], min_iou: float,
    root: str = "",
) -> bool:
    """Return True if one of the retrieved chunks matches *source*.

    Args:
        source: Expected source (file path + character range).
        retrieved: Chunks returned by the index.
        min_iou: Minimum IoU for a chunk to count as a match.
        root: Indexed folder, used to compare paths from its name on.

    Returns:
        True if a chunk of the same file overlaps *source* enough.
    """
    for chunk in retrieved:
        if not check_path(source["file_path"], from_root(chunk.file, root)):
            continue
        c_start, c_end = chunk_char_range(chunk)
        if check_iou(
            source["first_character_index"], source["last_character_index"],
            c_start, c_end,
        ) >= min_iou:
            return True
    return False


def load_questions(dataset_path: str) -> List[Question]:
    """Load the valid questions from the JSON dataset at *dataset_path*."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        questions: List[Question] = json.load(f)["rag_questions"]
    return [q for q in questions if q.get("is_valid", True)]


def find_recall_metric(
    index: HybridIndex, dataset_path: str, k: int, min_iou: float = 0.05
) -> float:
    """Calculate recall@k of *index* over the dataset.

    For each question, recall is the fraction of its expected sources
    found in the top-k retrieved chunks; the result is the mean over
    all questions.

    Args:
        index: Index used to retrieve chunks.
        dataset_path: Path to the questions dataset.
        k: Number of top chunks considered.
        min_iou: Minimum IoU for a chunk to match a source.

    Returns:
        Recall metric (0 to 1).

    Raises:
        ValueError: If k is not positive or the dataset is empty.
    """
    if k <= 0:
        raise ValueError("ERROR: k must be positive")
    questions = load_questions(dataset_path)
    if not questions:
        raise ValueError("ERROR: empty dataset")

    recall_lst: List[float] = []
    for question in questions:
        retrieved = index.search(question["question"], k=k)
        nb_source_find = sum(
            1 for source in question["sources"]
            if source_found(source, retrieved, min_iou, index.folder)
        )
        recall_lst.append(nb_source_find / len(question["sources"]))
    return sum(recall_lst) / len(recall_lst)
