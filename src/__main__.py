from .chunking.indexer import (
    build_chunck_list, list_python_files, save_chunk_list_to_json
)


def main():
    paths_files = list_python_files("Pac-Man")
    chunks = build_chunck_list(paths_files)
    save_chunk_list_to_json(chunks)


if __name__ == "__main__":
    main()
