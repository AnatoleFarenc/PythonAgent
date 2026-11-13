import json
from pathlib import Path
from typing import List

from model import ChunkListModel, ChunkModel

from .chunking_tools_python import get_chunk
from ..config import PYTHON_CHUNK_PATH


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
