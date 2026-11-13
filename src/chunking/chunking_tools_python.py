"""Syntax-aware chunking of Python files with tree-sitter.

Top-level nodes (functions, classes, statements) are merged into chunks of
at most ``CHUNK_LIMIT_CHARACTER`` bytes; a node too big for one chunk is
split recursively into its children, keeping the enclosing class/function
names in ``paths``.
"""

from typing import List, Optional, TypedDict
from tree_sitter import Language, Parser, Node
from pathlib import Path
import tree_sitter_python as tspython
import json
from ..config import (
    CHUNK_LIMIT_CHARACTER, PYTHON_CHUNK_PATH,
    )
from model import ChunkListModel, ChunkModel


PY = Language(tspython.language())
parser = Parser(PY)


class RawChunk(TypedDict):
    """Chunk boundaries before they are turned into a ``ChunkModel``."""

    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    paths: str


def chunk_node(node: Node, source: bytes, max_size: int,
               context: Optional[List[str]] = None) -> List[RawChunk]:
    """Chunk a single node, splitting it into its children if too big.

    Args:
        node: Syntax node to chunk.
        source: Full content of the file.
        max_size: Maximum chunk size in bytes.
        context: Names of the enclosing classes/functions.

    Returns:
        The chunks covering *node*.
    """
    context = context or []
    if node.end_byte - node.start_byte <= max_size or not node.named_children:
        s_line = node.start_point.row + 1
        e_line = node.end_point.row + 1
        return [format_chunk(
            node.start_byte, node.end_byte,
            s_line, e_line,
            context, source
            )
        ]

    name = node.child_by_field_name("name")
    n_context = context
    if name:
        n_context = context + [source[name.start_byte:name.end_byte].decode()]
    return merge_chunks(node.named_children, source, max_size, n_context)


def merge_chunks(
        nodes: List[Node], source: bytes, max_size: int,
        context: Optional[List[str]] = None) -> List[RawChunk]:
    """Group consecutive sibling nodes into chunks of at most *max_size*.

    Args:
        nodes: Sibling syntax nodes, in source order.
        source: Full content of the file.
        max_size: Maximum chunk size in bytes.
        context: Names of the enclosing classes/functions.

    Returns:
        The chunks covering all *nodes*.
    """
    context = context or []
    chunks: List[RawChunk] = []
    start: Optional[int] = None
    end: Optional[int] = None
    s_line = 0
    e_line = 0

    for node in nodes:
        size = node.end_byte - node.start_byte
        if size > max_size:
            if start is not None:
                chunks.append(
                    format_chunk(start, end, s_line, e_line, context, source)
                )
                start = None
                end = None
            chunks.extend(chunk_node(node, source, max_size, context))
        elif start is None:
            start, end = node.start_byte, node.end_byte
            s_line = node.start_point.row + 1
            e_line = node.end_point.row + 1
        elif node.end_byte - start <= max_size:
            end = node.end_byte
            e_line = node.end_point.row + 1
        else:
            chunks.append(
                format_chunk(start, end, s_line, e_line, context, source)
            )
            start, end = node.start_byte, node.end_byte
            s_line = node.start_point.row + 1
            e_line = node.end_point.row + 1

    if start is not None and end is not None:
        chunks.append(
            format_chunk(start, end, s_line, e_line, context, source)
        )
    return chunks


def format_chunk(start_byte: int, end_byte: Optional[int],
                 start_line: int, end_line: int,
                 context: List[str], source: bytes) -> RawChunk:
    """Build a ``RawChunk``; a missing *end_byte* means end of file.

    Args:
        start_byte: Offset of the first byte.
        end_byte: Offset just after the last byte, or None.
        start_line: First line (1-based).
        end_line: Last line (1-based, inclusive).
        context: Names of the enclosing classes/functions.
        source: Full content of the file.

    Returns:
        The chunk boundaries.
    """
    return {
        "start_byte": start_byte,
        "end_byte": end_byte if end_byte else len(source),
        "start_line": start_line,
        "end_line": end_line,
        "paths": " > ".join(context) if context else ""
    }


def get_chunk(path_file: str) -> List[RawChunk]:
    """Parse *path_file* and return its chunk boundaries.

    Args:
        path_file: Path of the Python file.

    Returns:
        The chunks of the file, in source order.
    """
    with open(path_file, "rb") as file:
        source = file.read()
        tree = parser.parse(source)
        nodes = tree.root_node.named_children
        return merge_chunks(nodes, source, CHUNK_LIMIT_CHARACTER)


def list_python_files(folder_path: str) -> List[str]:
    """Return every ``.py`` file under *folder_path*, sorted.

    Args:
        folder_path: Root folder to scan recursively.

    Returns:
        Sorted file paths.
    """
    root = Path(folder_path)
    return sorted(str(path) for path in root.rglob("*.py") if path.is_file())


def build_file_chunks(path: str, start_id: int = 0) -> List[ChunkModel]:
    """Chunk a single file; ids are numbered from *start_id*.

    Args:
        path: Path of the Python file.
        start_id: Id given to the first chunk.

    Returns:
        The chunks of the file.
    """
    return [
        ChunkModel(
            id=start_id + i,
            file=path,
            start_byte=chunk["start_byte"],
            end_byte=chunk["end_byte"],
            start_line=chunk["start_line"],
            end_line=chunk["end_line"],
            paths=chunk["paths"],
        )
        for i, chunk in enumerate(get_chunk(path))
    ]


def build_chunck_list(paths_file: List[str]) -> ChunkListModel:
    """Chunk every file of *paths_file* with consecutive ids.

    Args:
        paths_file: Paths of the Python files.

    Returns:
        All chunks, file after file.
    """
    all_chunks: List[ChunkModel] = []
    for path in paths_file:
        all_chunks.extend(build_file_chunks(path, len(all_chunks)))
    return ChunkListModel(chunks=all_chunks)


def save_chunk_list_to_json(
    chunk_list: ChunkListModel,
    output_path: str = PYTHON_CHUNK_PATH,
) -> Path:
    """Save *chunk_list* as JSON, creating parent folders if needed.

    Args:
        chunk_list: Chunks to save.
        output_path: Destination file.

    Returns:
        Path of the written file.
    """
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file:
        json.dump(chunk_list.to_dict(), file, indent=2, ensure_ascii=False)
    return target


def load_chunk_list_from_json(
    input_path: str = PYTHON_CHUNK_PATH,
) -> ChunkListModel:
    """Load a ChunkListModel previously saved by save_chunk_list_to_json.

    Args:
        input_path: Path to the JSON chunk file.

    Returns:
        The chunk list rebuilt from the file.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the JSON content is invalid or misses chunk fields.
    """
    try:
        with open(input_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return ChunkListModel(
            chunks=[ChunkModel(**chunk) for chunk in data["chunks"]]
        )
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {input_path}: {e}") from e
    except (KeyError, TypeError) as e:
        raise ValueError(f"Invalid chunk format in {input_path}: {e}") from e


def read_chunk(chunk: ChunkModel) -> str:
    """Return the source code of *chunk*, read from its file.

    Args:
        chunk: Chunk to read.

    Returns:
        The chunk's text.
    """
    with open(chunk.file, "rb") as file:
        file.seek(chunk.start_byte)
        return (
            file.read(
                chunk.end_byte - chunk.start_byte
            ).decode()
        )
