# pythonRag — a local coding assistant

A coding assistant that runs in the terminal and works on your project. You ask it questions about the code ("where is the config loaded?", "what does `RejectionSampler` do?") or ask it to make a change, and it explores the code, reads the relevant lines, edits files and answers with `file:line` references.

It is built to run **fully locally** on a single 8 GB GPU: the LLM is served by vLLM, code search uses a RAG over the project's Python files, and the code written by the model runs in a gVisor sandbox. On a machine that cannot run a model, the same assistant can use a remote API (Groq, OpenRouter, Google AI Studio) instead.

## How it works

```
 you ──► CLI (src/agent) ──► LLM (local vLLM or remote API)
              │                      │ writes Python code
              │                      ▼
              │              Sandbox (gVisor container)
              │                      │ calls tools as Python functions
              │                      ▼
              └──────────────  MCP server (src/mcp_server)
                                     │
                       search_code ──┤── RAG index (BM25 + bge-m3 on CPU)
                grep / read_file / ──┤
            list_files / edit_file   └── your project folder
```

1. **The agent loop** ([src/agent/](src/agent/)): for each message, the model answers with a short thought and one Python code block. The code runs in the sandbox, its output is sent back to the model, and so on until the model calls `final_answer(...)`, whose text is the reply you see. The conversation then goes on: earlier answers stay in the context, the intermediate steps are dropped to save tokens.
2. **The tools** ([src/mcp_server/](src/mcp_server/)): an MCP server exposes the project to the model. All paths are confined to the project folder.

   | Tool | What it does |
   |---|---|
   | `search_code(query, k)` | Finds code by meaning with the RAG (hybrid BM25 + embeddings search) |
   | `grep(pattern, path, glob)` | Exact regex search, returns `file:line: text` |
   | `read_file(path, start_line, end_line)` | Numbered lines of a file, 400 at most per call |
   | `list_files(path, pattern)` | Folder content or glob (`**/*.py`) |
   | `edit_file(path, old_str, new_str)` | Replaces one exact, unique piece of text and returns the diff |
   | `create_file(path, content)` | Creates a new file, never overwrites |

   `edit_file` and `create_file` are not available in read-only mode. A Python file is only written if it still parses, and `.git`, `.venv` and caches cannot be written to. After an edit, the RAG index is updated in the background.
3. **The RAG** ([src/rag/](src/rag/)): Python files are split into chunks with tree-sitter, encoded with `BAAI/bge-m3` and indexed with BM25. The index is cached in `data/chunk/` and only modified files are re-encoded. See [src/rag/README.md](src/rag/README.md) for details and the retrieval benchmark.
4. **The sandbox** ([src/sandbox/](src/sandbox/)): the model's code runs in a Docker container with the gVisor (`runsc`) runtime, with no network, a memory limit and a time limit. It cannot read your files directly: it can only go through the tools above.

The agent loop is adapted from Agent-Smith, an earlier group project that targeted benchmarks (MBPP, SWE-bench) with remote APIs only.

## Requirements

- Linux, Python 3.14 and [uv](https://docs.astral.sh/uv/)
- **For the local model:** an NVIDIA GPU with 8 GB of memory and CUDA
- **For the sandbox:** Docker with the [gVisor](https://gvisor.dev/docs/user_guide/install/) `runsc` runtime registered. Without them, `--unsafe-local-sandbox` runs the code as a plain process, **not isolated** from your machine.
- **For a remote API:** an API key from Groq, OpenRouter or Google AI Studio

## Installation

```bash
make install                                                     # virtual environment + dependencies
docker build -f docker/sandbox.Dockerfile -t pythonrag-sandbox:dev .   # sandbox image
```

The sandbox image contains the code of `src/sandbox/`: rebuild it after changing that folder (`make test` does it automatically).

To use a remote API, create a `.env` file at the root. Each variable holds one key or a comma-separated list; when a key hits its quota, the next one is used.

```bash
GROQ_API_KEYS="key1, key2"
OPENROUTER_API_KEYS="key1"
GOOGLE_API_KEYS="key1"
```

`.env` is ignored by git.

## Quick start

All commands are run from the repository root.

**With the local model**, in two terminals:

```bash
make serve        # terminal 1: loads the model on the GPU (about a minute)
make assistant    # terminal 2: the chat
```

Both can be started at the same time: the assistant shows a spinner until the model and the code index are ready, then the `User >` prompt.

**With a remote API**, one terminal, no GPU needed:

```bash
make assistant-api                 # Groq
make assistant PROVIDER=openrouter # or another provider
```

By default the assistant works on the current folder (the repository itself). To work on another project:

```bash
make assistant ROOT=../my_project
```

## Options

### Makefile variables

Any variable can be set on the command line, e.g. `make serve LLM_MODEL=... LLM_GPU=0.9`.

| Variable | Default | Used by | Meaning |
|---|---|---|---|
| `LLM_MODEL` | `Qwen/Qwen2.5-Coder-3B-Instruct-AWQ` | `serve` | Model loaded by vLLM |
| `LLM_CONTEXT` | `16384` | `serve`, `assistant` | Context length in tokens (prompt + answer) |
| `LLM_GPU` | `0.85` | `serve` | Share of the GPU memory vLLM may use |
| `LLM_PORT` | `8000` | `serve`, `assistant` | Port of the local server |
| `PROVIDER` | `local` | `assistant` | `local`, `groq`, `openrouter` or `google` |
| `ROOT` | `.` | `assistant`, `index` | Project the assistant works on |
| `ARGS` | | `assistant` | Extra CLI options, e.g. `ARGS="--read-only"` |

### Make targets

| Target | What it does |
|---|---|
| `make serve` | Starts `vllm serve` with the settings above |
| `make assistant` | Starts the chat |
| `make assistant-api` | Same as `make assistant PROVIDER=groq` |
| `make index` | Builds the code index of `ROOT` on the GPU (see [Code index](#code-index)) |
| `make test` | Runs the test suite |

### CLI options

`make assistant` runs `python -m src.agent`, which can also be called directly (`uv run python -m src.agent ...`):

| Option | Default | Meaning |
|---|---|---|
| `--provider NAME` | `local` | Where the model runs |
| `--model ID` | provider default | Model to use with a remote API |
| `--root PATH` | `.` | Project folder |
| `--read-only` | off | No `edit_file` / `create_file`: the assistant can only read |
| `--no-embedding` | off | Code search with BM25 only, without bge-m3 |
| `--max-steps N` | `15` | Steps allowed per message before the assistant stops |
| `--unsafe-local-sandbox` | off | Run the model's code without Docker/gVisor (**not isolated**) |

### Providers

| Provider | Default model | API key variable |
|---|---|---|
| `local` | Whatever `vllm serve` is serving (asked to the server) | none |
| `groq` | `qwen/qwen3.8-27b` | `GROQ_API_KEYS` |
| `openrouter` | `openrouter/free` (OpenRouter picks a free model) | `OPENROUTER_API_KEYS` |
| `google` | `gemini-3.5-flash-lite` | `GOOGLE_API_KEYS` |

Use `--model` (or `ARGS="--model ..."` with make) to pick another model of a remote provider. Providers are defined in [src/agent/providers.py](src/agent/providers.py).

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `LOCAL_LLM_URL` | `http://localhost:8000/v1` | Address of the local vLLM server (set by `make assistant`) |
| `LOCAL_LLM_CONTEXT` | `16384` | Context length of the local model, must match `--max-model-len` (set by `make assistant`) |
| `AMBEDING_DEVICE` | `cpu` | Where bge-m3 runs: `cpu` or `cuda` |
| `*_API_KEYS` | | API keys, see [Installation](#installation) |

## Using the chat

```
local: Qwen/Qwen2.5-Coder-7B-Instruct-AWQ — project . 
User > where is the sandbox memory limit applied?
[1] Find where the memory limit is set.
print(grep("max_memory_mb"))
→ src/sandbox/worker.py:...

The memory limit is applied in ...
```

- Each step is shown in grey: the model's thought, the code it ran and the start of the result. The final answer is shown in normal text.
- `exit` or `quit` (with or without `/`), or Ctrl+D, quits. `/reset` clears the conversation.
- Ctrl+C stops the current request. If it happens while code is running, the sandbox is restarted.
- When the step limit is reached, type `continue` to let the assistant go on.
- The logs of the MCP server and of the RAG go to `data/logs/mcp_server.log`.

## Choosing the local model

| Model | Command | GPU memory | Quality |
|---|---|---|---|
| Qwen2.5-Coder 3B (default) | `make serve` | ~5 GB | Follows the format, but often answers without checking |
| Qwen2.5-Coder 7B | `make serve LLM_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct-AWQ LLM_GPU=0.9` | ~7.5 GB | Clearly better, still makes mistakes on exact values |

bge-m3 runs on the CPU, so the whole GPU is left to the LLM. A remote API model (`make assistant-api`) is the most reliable option.

## Code index

The first time a project is opened, its Python files are chunked and encoded; this is cached in `data/chunk/` and later runs only re-encode new or modified files.

On the CPU, bge-m3 encodes about 2.6 chunks per second: a search or an edit is fast, but indexing a whole project from scratch is slow (a few minutes for a small project, about two hours for a codebase like vLLM). The cache does not depend on the device, so build it once on the GPU, with `make serve` stopped:

```bash
make index ROOT=../my_project
make assistant ROOT=../my_project   # use the same ROOT
```

There is a single cache: indexing another folder replaces the previous one.

## Other entry points

- **RAG alone**, question answering over the vLLM corpus with its own LLM: see [src/rag/README.md](src/rag/README.md).
- **MCP server alone**, to plug the tools into another MCP client:
  ```bash
  uv run python -m src.mcp_server --root ../my_project --writable                               # stdio
  uv run python -m src.mcp_server --root . --transport streamable-http --port 8765              # HTTP
  ```
- **Sandbox REPL**, to try the sandbox by hand: `uv run sandbox --help`.

## Tests

```bash
make test
```

The tests of the agent loop and of the LLM client use a scripted fake model, so they need no GPU and no network. The sandbox isolation tests need Docker and `runsc` and are skipped without them (`--require-isolation` makes them fail instead).

## Troubleshooting

| Problem | Fix |
|---|---|
| `zsh: command not found: vllm` | vLLM is installed in `.venv`: use `make serve` or `uv run vllm ...` |
| `Ninja build failed ... --compress-mode=size` | FlashInfer cannot be compiled with this nvcc. `make serve` already sets `VLLM_USE_FLASHINFER_SAMPLER=0`; set it yourself when calling `vllm serve` directly |
| `... KV cache is needed, which is larger than the available KV cache memory` | The model does not fit with this context: raise `LLM_GPU` (e.g. `0.9`) or lower `LLM_CONTEXT` |
| `cannot reach http://localhost:8000/v1 — is vllm serve running?` | Start `make serve`, or check `LLM_PORT` |
| `No API key found: set GROQ_API_KEYS in .env` | Add the key to `.env` (note the `S` in `KEYS`) |
| `code search unavailable` | The index could not be built: see `data/logs/mcp_server.log`. `grep` and `read_file` still work |

## Project layout

```
src/
├── agent/          chat CLI, agent loop, LLM client, providers, prompt
├── mcp_server/     MCP server: code tools on one project folder
├── rag/            chunking, incremental index, hybrid search, RAG generator
├── sandbox/        sandbox host side, worker, MCP client
└── models_agent/   shared pydantic models (sandbox config, results)
docker/             sandbox image
tests/              test suite
data/               index cache and logs (not in git)
```

## Limitations

- The assistant cannot run tests or shell commands on the project yet.
- The code index covers Python files only. Other files are reachable with `grep` and `read_file`.
- The CLI and `make` must be run from the repository root.
