from typing import List, Dict
from tree_sitter import Language, Parser, Node
import tree_sitter_python as tspython

from ..config import CHUNK_LIMIT_CHARACTER


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

    for node in nodes:
        size = node.end_byte - node.start_byte
        if size > max_size:
            if start is not None:
                s_line = node.start_point.row + 1
                e_line = node.end_point.row + 1
                chunks.append(
                    format_chunk(start, end, s_line, e_line, context, source)
                )
                start = None
                end = None
            chunks.extend(chunk_node(node, source, max_size, context))
        elif start is None:
            start, end = node.start_byte, node.end_byte
        elif node.end_byte - start <= max_size:
            end = node.end_byte
        else:
            s_line = node.start_point.row + 1
            e_line = node.end_point.row + 1
            chunks.append(
                format_chunk(start, end, s_line, e_line, context, source)
            )
            start, end = node.start_byte, node.end_byte

    if start is not None and end is not None:
        s_line = node.start_point.row + 1
        e_line = node.end_point.row + 1
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
