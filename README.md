# pythonRag — building blocks for a local coding assistant

The pieces of a coding assistant that works on a Python project, built to run **fully locally** on a single 8 GB GPU: code search with a RAG over the project's Python files, an MCP server that exposes the project as tools, and a gVisor sandbox that runs the code written by the model. The orchestration (the loop that drives the model) lives in [orchestrator/](orchestrator/) and is a work in progress.

## How it fits together

```
 orchestrator ──► LLM (local vLLM or remote API)
      │                  │ writes Python code
      │                  ▼
      │          Sandbox (gVisor container)
      │                  │ calls tools as Python functions
      │                  ▼
      └──────────  MCP server (src/mcp_server)
                         │
           search_code ──┤── RAG index (BM25 + bge-m3 on CPU)
    grep / read_file / ──┤
list_files / edit_file   └── your project folder
```

1. **The tools** ([src/mcp_server/](src/mcp_server/)): an MCP server exposes the project to the model. All paths are confined to the project folder.

   | Tool | What it does |
   |---|---|
   | `search_code(query, k)` | Finds code by meaning with the RAG (hybrid BM25 + embeddings search) |
   | `grep(pattern, path, glob)` | Exact regex search, returns `file:line: text` |
   | `read_file(path, start_line, end_line)` | Numbered lines of a file, 400 at most per call |
   | `list_files(path, pattern)` | Folder content or glob (`**/*.py`) |
   | `edit_file(path, old_str, new_str)` | Replaces one exact, unique piece of text and returns the diff |
   | `create_file(path, content)` | Creates a new file, never overwrites |

   `edit_file` and `create_file` are only available with `--writable`. A Python file is only written if it still parses, and `.git`, `.venv` and caches cannot be written to. After an edit, the RAG index is updated in the background.
2. **The RAG** ([src/rag/](src/rag/)): Python files are split into chunks with tree-sitter, encoded with `BAAI/bge-m3` and indexed with BM25. The index is cached in `data/chunk/` and only modified files are re-encoded. See [src/rag/README.md](src/rag/README.md) for details and the retrieval benchmark.
3. **The sandbox** ([src/sandbox/](src/sandbox/)): code runs in a Docker container with the gVisor (`runsc`) runtime, with no network, a memory limit and a time limit. It cannot read your files directly: it can only go through the tools above.
4. **The orchestrator** ([orchestrator/](orchestrator/)): work in progress.

## Requirements

- Linux, Python 3.14 and [uv](https://docs.astral.sh/uv/)
- **For the local model:** an NVIDIA GPU with 8 GB of memory and CUDA
- **For the sandbox:** Docker with the [gVisor](https://gvisor.dev/docs/user_guide/install/) `runsc` runtime registered

## Installation

```bash
make install                                                     # virtual environment + dependencies
docker build -f docker/sandbox.Dockerfile -t pythonrag-sandbox:dev .   # sandbox image
```

The sandbox image contains the code of `src/sandbox/`: rebuild it after changing that folder (`make test` does it automatically).

## Entry points

All commands are run from the repository root.

- **Local LLM server**: `make serve` starts `vllm serve` (see the variables below).
- **RAG alone**, question answering over the vLLM corpus with its own LLM: `make run`, and see [src/rag/README.md](src/rag/README.md).
- **MCP server**, to plug the tools into an MCP client:
  ```bash
  uv run python -m src.mcp_server --root ../my_project --writable                               # stdio
  uv run python -m src.mcp_server --root . --transport streamable-http --port 8765              # HTTP
  ```
- **Sandbox REPL**, to try the sandbox by hand: `uv run sandbox --help`.

### Makefile variables

Any variable can be set on the command line, e.g. `make serve LLM_MODEL=... LLM_GPU=0.9`.

| Variable | Default | Used by | Meaning |
|---|---|---|---|
| `LLM_MODEL` | `Qwen/Qwen2.5-Coder-3B-Instruct-AWQ` | `serve` | Model loaded by vLLM |
| `LLM_CONTEXT` | `16384` | `serve` | Context length in tokens (prompt + answer) |
| `LLM_GPU` | `0.85` | `serve` | Share of the GPU memory vLLM may use |
| `LLM_PORT` | `8000` | `serve` | Port of the local server |
| `ROOT` | `.` | `index` | Project to index |

### Make targets

| Target | What it does |
|---|---|
| `make serve` | Starts `vllm serve` with the settings above |
| `make run` | Runs the RAG demo (`python -m src`) |
| `make index` | Builds the code index of `ROOT` on the GPU (see [Code index](#code-index)) |
| `make test` | Runs the test suite |

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `AMBEDING_DEVICE` | `cpu` | Where bge-m3 runs: `cpu` or `cuda` |

## Choosing the local model

| Model | Command | GPU memory |
|---|---|---|
| Qwen2.5-Coder 3B (default) | `make serve` | ~5 GB |
| Qwen2.5-Coder 7B | `make serve LLM_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct-AWQ LLM_GPU=0.9` | ~7.5 GB |

bge-m3 runs on the CPU, so the whole GPU is left to the LLM.

## Code index

The first time a project is indexed, its Python files are chunked and encoded; this is cached in `data/chunk/` and later runs only re-encode new or modified files.

On the CPU, bge-m3 encodes about 2.6 chunks per second: a search or an edit is fast, but indexing a whole project from scratch is slow (a few minutes for a small project, about two hours for a codebase like vLLM). The cache does not depend on the device, so build it once on the GPU, with `make serve` stopped:

```bash
make index ROOT=../my_project
```

There is a single cache: indexing another folder replaces the previous one.

## Tests

```bash
make test
```

The sandbox isolation tests need Docker and `runsc` and are skipped without them (`--require-isolation` makes them fail instead).

## Troubleshooting

| Problem | Fix |
|---|---|
| `zsh: command not found: vllm` | vLLM is installed in `.venv`: use `make serve` or `uv run vllm ...` |
| `Ninja build failed ... --compress-mode=size` | FlashInfer cannot be compiled with this nvcc. `make serve` already sets `VLLM_USE_FLASHINFER_SAMPLER=0`; set it yourself when calling `vllm serve` directly |
| `... KV cache is needed, which is larger than the available KV cache memory` | The model does not fit with this context: raise `LLM_GPU` (e.g. `0.9`) or lower `LLM_CONTEXT` |
| `code search unavailable` | The index could not be built: see `data/logs/mcp_server.log`. `grep` and `read_file` still work |

## Project layout

```
orchestrator/       orchestration (work in progress)
src/
├── mcp_server/     MCP server: code tools on one project folder
├── rag/            chunking, incremental index, hybrid search, RAG generator
├── sandbox/        sandbox host side, worker, MCP client
└── models_agent/   shared pydantic models (sandbox config, results)
docker/             sandbox image
tests/              test suite
data/               index cache and logs (not in git)
```

## Limitations

- The code index covers Python files only. Other files are reachable with `grep` and `read_file`.
- `make` must be run from the repository root.
