"""Project-wide constants: chunk size and on-disk index locations."""
import os

CHUNK_LIMIT_CHARACTER = 2000
PYTHON_CHUNK_PATH = "data/chunk/chunk_python.json"
PYTHON_VECTOR_CHUNK_PATH = "data/chunk/chunk_python.npy"
PYTHON_MANIFEST_PATH = "data/chunk/manifest.json"
# Folders never indexed: virtualenvs, caches, VCS metadata.
EXCLUDED_DIRS = frozenset({
    ".git", ".venv", "venv", "__pycache__", "node_modules",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
})

# Answer generation (vLLM)
LLM_MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct-AWQ"
LLM_GPU_MEMORY_UTILIZATION = 0.6
LLM_MAX_MODEL_LEN = 8192
LLM_MAX_TOKENS = 512
LLM_TEMPERATURE = 0.3
LLM_TOP_P = 0.8
LLM_ENFORCE_EAGER = True
INTERFACE_LOG_FILE = ""


# Ambeding
AMBEDING_MODEL = "BAAI/bge-m3"
# "cpu" leaves the whole GPU to the LLM served by vLLM. Searching only
# encodes the query (fast on CPU); the chunks are encoded once and cached,
# only new/modified files are re-encoded. "cuda" is faster for a first
# full indexing of a big project (make index): the cache does not depend
# on the device, so it can be built on GPU once and used on CPU after.
AMBEDING_DEVICE = os.environ.get("AMBEDING_DEVICE", "cpu")
# False: BM25 search only. Faster start, ~2.2 GB of GPU freed, no query
# encoding, but lower retrieval quality (no semantic matching).
USE_EMBEDDING = True

# recall
TEST_RECALL_FOLDER = "src/rag/vllm-0.10.1"
