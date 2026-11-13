from typing import List, Dict
from tree_sitter import Language, Parser, Node
from pathlib import Path
import tree_sitter_python as tspython
import json

from ..config import CHUNK_LIMIT_CHARACTER, PYTHON_CHUNK_PATH
from model import ChunkListModel, ChunkModel


PY = Language(tspython.language())
parser = Parser(PY)


def chunk_node(node: Node, source: bytes, max_size: int,
               context: List[str] = []) -> List[Dict]:
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
        context: List[str] = []) -> List[Dict]:
    chunks = []
    start = None
    end = None
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


def format_chunk(start_byte: int, end_byte: int | None,
                 start_line: int, end_line: int,
                 context: list[str], source: bytes) -> Dict:
    return {
        "start_byte": start_byte,
        "end_byte": end_byte if end_byte else len(source),
        "start_line": start_line,
        "end_line": end_line,
        "paths": " > ".join(context) if context else ""
    }


def get_chunk(path_file: str) -> List[Dict]:
    with open(path_file, "rb") as file:
        source = file.read()
        tree = parser.parse(source)
        nodes = tree.root_node.named_children
        return merge_chunks(nodes, source, CHUNK_LIMIT_CHARACTER)


def list_python_files(folder_path: str) -> list[str]:
    root = Path(folder_path)
    return sorted(str(path) for path in root.rglob("*.py") if path.is_file())


def build_chunck_list(paths_file: List[str]) -> ChunkListModel:
    id = 0
    all_chunks: list[ChunkModel] = []
    for path in paths_file:
        chunks = get_chunk(path)
        for chunk in chunks:
            n_chunk = ChunkModel(
                id=id,
                file=path,
                start_byte=chunk["start_byte"],
                end_byte=chunk["end_byte"],
                start_line=chunk["start_line"],
                end_line=chunk["end_line"],
                paths=chunk["paths"],
            )
            all_chunks.append(n_chunk)
            id += 1
    return ChunkListModel(chunks=all_chunks)


def save_chunk_list_to_json(
    chunk_list: ChunkListModel,
    output_path: str = PYTHON_CHUNK_PATH,
) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file:
        json.dump(chunk_list.to_dict(), file, indent=2, ensure_ascii=False)
    return target


def read_chunk(chunk: ChunkModel) -> str:
    with open(chunk.file, "rb") as file:
        file.seek(chunk.start_byte)
        return (
            file.read(
                chunk.end_byte - chunk.start_byte
            ).decode()
        )
