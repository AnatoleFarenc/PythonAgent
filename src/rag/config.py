"""Project-wide constants: chunk size and on-disk index locations."""

CHUNK_LIMIT_CHARACTER = 2000
PYTHON_CHUNK_PATH = "data/chunk/chunk_python.json"
PYTHON_VECTOR_CHUNK_PATH = "data/chunk/chunk_python.npy"
PYTHON_MANIFEST_PATH = "data/chunk/manifest.json"

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
# False: BM25 search only. Faster start, ~2.2 GB of GPU freed, no query
# encoding, but lower retrieval quality (no semantic matching).
USE_EMBEDDING = True

# recall
TEST_RECALL_FOLDER = "src/rag/vllm-0.10.1"
