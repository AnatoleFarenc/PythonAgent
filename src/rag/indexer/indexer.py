"""Incremental indexing: only re-chunk and re-encode files that changed.

On disk (see config.py):
    chunk_python.json  -> the chunks
    chunk_python.npy   -> one vector per chunk, same order as the chunks
    manifest.json      -> model name, chunk size and one hash per file

At each run, every file's hash is compared to the one in the manifest:
unchanged files reuse their chunks and vectors, new or modified files are
re-chunked and re-encoded, deleted files disappear from the index.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypeAlias

import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer

from model import ChunkListModel, ChunkModel
from ..chunking.chunking_tools_python import (
    build_file_chunks, list_python_files, load_chunk_list_from_json,
    read_chunk, save_chunk_list_to_json,
)
from ..config import (
    CHUNK_LIMIT_CHARACTER, PYTHON_CHUNK_PATH, PYTHON_MANIFEST_PATH,
    PYTHON_VECTOR_CHUNK_PATH,
)

Vectors: TypeAlias = npt.NDArray[np.float32]


def chunk_text(c: ChunkModel) -> str:
    """Text that is embedded (and indexed by BM25) for a chunk."""
    return f"# File: {c.file}\n# Path: {c.paths}\n{read_chunk(c)}"


def file_hash(path: str) -> str:
    """SHA-256 fingerprint of the file content."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_cache(
    model_name: str,
) -> Tuple[Dict[str, str], Dict[str, List[Tuple[ChunkModel, int]]], Vectors]:
    """Load the previous index if it is usable with *model_name*.

    Args:
        model_name: Name of the model the cache must have been built with.

    Returns:
        (file hashes, {file: [(chunk, row in vectors)]}, vectors).
        Everything is empty when there is no valid cache, which forces
        a full rebuild.
    """
    empty: Tuple[Dict[str, str], Dict[str, List[Tuple[ChunkModel, int]]],
                 Vectors] = ({}, {}, np.zeros((0, 0), dtype=np.float32))
    try:
        manifest = json.loads(Path(PYTHON_MANIFEST_PATH).read_text())
        chunks = load_chunk_list_from_json(PYTHON_CHUNK_PATH).chunks
        vectors: Vectors = np.load(PYTHON_VECTOR_CHUNK_PATH)
    except (OSError, ValueError):
        return empty

    if (manifest.get("model") != model_name
            or manifest.get("chunk_limit") != CHUNK_LIMIT_CHARACTER
            or len(chunks) != len(vectors)):
        print("[index] cache incompatible -> full rebuild")
        return empty

    by_file: Dict[str, List[Tuple[ChunkModel, int]]] = {}
    for row, chunk in enumerate(chunks):
        by_file.setdefault(chunk.file, []).append((chunk, row))
    return manifest["files"], by_file, vectors


def _save_cache(
    model_name: str, hashes: Dict[str, str],
    chunks: ChunkListModel, vectors: Vectors,
) -> None:
    """Write chunks, vectors and manifest to disk.

    Args:
        model_name: Name of the model that produced *vectors*.
        hashes: SHA-256 of every indexed file.
        chunks: Chunks to save.
        vectors: Their embeddings (row ``i`` <-> chunk ``i``).
    """
    save_chunk_list_to_json(chunks, PYTHON_CHUNK_PATH)
    np.save(PYTHON_VECTOR_CHUNK_PATH, vectors)
    Path(PYTHON_MANIFEST_PATH).write_text(json.dumps({
        "model": model_name,
        "chunk_limit": CHUNK_LIMIT_CHARACTER,
        "files": hashes,
    }, indent=1))


def update_index(
    folder: str, model: Optional[SentenceTransformer], model_name: str,
) -> Tuple[ChunkListModel, Optional[Vectors]]:
    """Bring the on-disk index of *folder* up to date and return it.

    Args:
        folder: Root folder of the code to index.
        model: Embedding model used to encode new chunks, or None to skip
            embeddings (BM25 only). Without a model the chunks are still
            reused from the cache, but nothing is encoded and the cache is
            not rewritten, so its vectors stay valid for a later run with
            embeddings.
        model_name: Name of the model, stored to detect a model change.

    Returns:
        The chunks and their vectors (row i <-> chunk i); vectors are None
        when *model* is None.

    Raises:
        FileNotFoundError: If *folder* contains no Python file (wrong
            path): the cache is kept instead of being emptied.
    """
    files = list_python_files(folder)
    if not files:
        raise FileNotFoundError(f"no .py file found in '{folder}'")
    old_hashes, old_chunks, old_vectors = _load_cache(model_name)
    hashes = {f: file_hash(f) for f in files}

    chunks: List[ChunkModel] = []
    rows: List[int] = []
    reused = changed = 0
    for f in files:
        if old_hashes.get(f) == hashes[f]:
            # A file without chunks (e.g. empty __init__.py) has none here.
            for chunk, row in old_chunks.get(f, []):
                chunks.append(chunk)
                rows.append(row)
            reused += 1
        else:
            for chunk in build_file_chunks(f):
                chunks.append(chunk)
                rows.append(-1)
            changed += 1
    deleted = len(set(old_hashes) - set(hashes))

    for i, chunk in enumerate(chunks):
        chunk.id = i

    result = ChunkListModel(chunks=chunks)
    if model is None:
        print(f"[index] embeddings disabled: {reused} files reused, "
              f"{changed} chunked")
        return result, None

    to_encode = [i for i, r in enumerate(rows) if r == -1]
    dim = model.get_embedding_dimension() or old_vectors.shape[1]
    vectors: Vectors = np.zeros((len(chunks), dim), dtype=np.float32)
    keep = [i for i, r in enumerate(rows) if r != -1]
    if keep:
        vectors[keep] = old_vectors[[rows[i] for i in keep]]
    if to_encode:
        vectors[to_encode] = model.encode(
            [chunk_text(chunks[i]) for i in to_encode],
            normalize_embeddings=True, batch_size=32,
            show_progress_bar=len(to_encode) > 100,
        )

    print(f"[index] {reused} files reused, {changed} new/modified, "
          f"{deleted} deleted -> {len(to_encode)} chunks encoded")

    if changed or deleted or not old_hashes:
        _save_cache(model_name, hashes, result, vectors)
    return result, vectors
