# rag

A RAG over Python code. It indexes a code base, retrieves the code chunks most relevant to a question, and an LLM served by vLLM writes the answer from those chunks.

The goal is to plug it into the project I am working on, so it can answer questions about that project's code. The vLLM 0.10.1 source code is only a test corpus: it is large enough to be realistic, and a dataset of questions with expected answers makes it possible to measure recall.

**Next step:** build the agent that will use this RAG.

## How it works

```
.py files ──► chunking ──► embeddings + BM25 ──► hybrid search ──► prompt ──► LLM (vLLM) ──► answer
```

1. **Chunking** ([chunking/chunking_tools_python.py](chunking/chunking_tools_python.py)): each file is parsed with tree-sitter. Top-level nodes (functions, classes, statements) are merged into chunks of at most `CHUNK_LIMIT_CHARACTER` bytes. A node that is too big is split into its children, and the chunk keeps the name of the enclosing class or function (`paths`).
2. **Incremental indexing** ([indexer/indexer.py](indexer/indexer.py)): each chunk is encoded with `BAAI/bge-m3`. Chunks, vectors and a manifest holding the SHA-256 of every file are saved in `data/chunk/`. On later runs, only new or modified files are re-encoded.
3. **Hybrid search** ([indexer/HybridIndex.py](indexer/HybridIndex.py)): two rankings are computed, one by keywords with BM25 (snake_case and camelCase are split), the other by cosine similarity between embeddings. They are merged with Reciprocal Rank Fusion to give the `k` best chunks.
4. **Generation** ([generator.py](generator.py)): chunks are added to the prompt in order of relevance as long as they fit in the model's context. `RagGenerator.answer_many` sends all questions to vLLM in a single batch.
5. **Evaluation** ([indexer/recall.py](indexer/recall.py)): `find_recall_metric` computes recall@k on a dataset. An expected source counts as found when a chunk from the same file overlaps it (IoU ≥ `min_iou`).

Settings (models, chunk size, GPU memory, log file) live in [config.py](config.py).

## Configuration

The indexed folder is set by `TEST_RECALL_FOLDER` in [config.py](config.py). To plug the RAG into another project, point it at that project.

`USE_EMBEDDING = False` disables bge-m3 and searches with BM25 only. Startup is faster, about 2.2 GB of GPU memory is freed and questions are not encoded, but retrieval is weaker (see [Benchmark](#benchmark)). The cached vectors are kept, so turning it back on needs no re-encoding.

The default model, `Qwen2.5-Coder-3B-Instruct-AWQ`, is chosen to fit on an 8 GB GPU alongside bge-m3. Weights and KV cache must fit in about 5 GB.

## Benchmark

Retrieval quality of the hybrid search (BM25 + bge-m3) compared with BM25 alone (`USE_EMBEDDING = False`), on two corpora:

- **vLLM 0.10.1**, a large code base: 1,641 files, 18,701 chunks. Evaluated on 99 questions from `dataset_code_public.json`. A source counts as found when a retrieved chunk overlaps its character range (IoU ≥ 0.05).
- **Pac-Man**, a small personal project closer to the target use case: 38 files, 187 chunks. Evaluated on 50 questions from `recall_at_k_questions.json`. A source counts as found when a retrieved chunk overlaps its line range.

| Corpus | Search | recall@3 | recall@5 | recall@10 |
|---|---|---|---|---|
| vLLM | BM25 only | 0.727 | 0.808 | 0.879 |
| vLLM | BM25 + embeddings | **0.788** | **0.848** | **0.909** |
| Pac-Man | BM25 only | 0.740 | 0.800 | 0.840 |
| Pac-Man | BM25 + embeddings | **0.810** | **0.930** | **0.970** |

| Corpus | Search | Index load | Time per query |
|---|---|---|---|
| vLLM | BM25 only | 1.0 s | 41 ms |
| vLLM | BM25 + embeddings | 3.3 s | 67 ms |
| Pac-Man | BM25 only | < 0.1 s | 0.2 ms |
| Pac-Man | BM25 + embeddings | 11.7 s | 9.6 ms |

Measured on an RTX 4060 (8 GB). Index load times for vLLM reuse the cached vectors. For Pac-Man, the time with embeddings includes encoding all 187 chunks from scratch. Time per query covers the search and the recall scoring.

**Takeaways**

- **Embeddings help on both corpora.** The gain is largest on the small project: +13 points of recall@5 on Pac-Man, +4 on vLLM. Questions about a personal project are phrased in natural language ("what happens when the player dies?"), so they often share few keywords with the code, and semantic matching makes up the difference.
- **BM25 alone is a reasonable fallback** when GPU memory or startup time matters: about 80 % recall@5 on both corpora, with no model to load.
- **Default to `USE_EMBEDDING = True`** for the target project. It is small, so its embeddings are cheap to compute, and that is where they bring the most.

## Testing on the vLLM corpus

The vLLM code used as the test corpus is shipped compressed in `vllm-0.10.1.zip`. All commands below are run from the project root.

**1. Unzip the corpus** into `src/rag/vllm-0.10.1/`, the folder `TEST_RECALL_FOLDER` points to:

```bash
unzip src/rag/vllm-0.10.1.zip -d src/rag/
```

**2. Ask questions** in the interactive prompt:

```bash
python -m src.rag.interface
```

The first run builds the index (about 18,700 chunks, several minutes on GPU). Later runs reuse it from `data/chunk/`. Quit with `exit`, Ctrl-D or Ctrl-C. vLLM logs go to `INTERFACE_LOG_FILE`, or are discarded when it is `""`.

Example question: `What activation formats does the fused batched MoE layer return in vLLM?`. Expected answer: a tuple of two `FusedMoEActivationFormat.BatchedExperts`.

**3. Measure retrieval quality** (recall@k) on the question dataset `data/datasets/dataset_code_public.json`:

```bash
python -c "
from src.rag.config import AMBEDING_MODEL, TEST_RECALL_FOLDER
from src.rag.indexer.HybridIndex import HybridIndex
from src.rag.indexer.recall import find_recall_metric
index = HybridIndex.from_folder(TEST_RECALL_FOLDER, AMBEDING_MODEL)
for k in (3, 5, 10):
    print(f'recall@{k}:', find_recall_metric(index, 'data/datasets/dataset_code_public.json', k))
"
```

This step only evaluates the search, so it does not load the LLM.
