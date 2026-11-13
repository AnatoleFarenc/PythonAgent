from .chunking.chunking_tools_python import (
    build_chunck_list, list_python_files, save_chunk_list_to_json,
    read_chunk
)
from .chunking.HybridIndex import HybridIndex


def main():
    paths_files = list_python_files("Pac-Man")
    chunks = build_chunck_list(paths_files)
    # save_chunk_list_to_json(chunks)
    # print(read_chunk(chunks.chunks[5]))

    texts = [
        f"# File: {c.file}\n# Path: {c.paths}\n{read_chunk(c)}"
        for c in chunks.chunks
    ]
    index = HybridIndex(chunks, texts, "BAAI/bge-small-en-v1.5")

    for c in index.search("How move the player"):
        print(c.file, c.start_line, c.end_line, c.paths)


if __name__ == "__main__":
    main()
