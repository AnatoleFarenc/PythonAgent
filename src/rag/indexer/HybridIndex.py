"""Hybrid retrieval: BM25 (keywords) + dense vectors, fused with RRF.

When ``USE_EMBEDDING`` is False in config.py, only BM25 is used.
"""

from model import ChunkListModel, ChunkModel
from ..config import AMBEDING_DEVICE, USE_EMBEDDING
from .indexer import Vectors, chunk_text, update_index

import re
import numpy as np
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]
from sentence_transformers import SentenceTransformer
from typing import List, Optional


def tokenize(text: str) -> list[str]:
    """Lowercase tokens; splits snake_case and camelCase."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return [t.lower() for t in re.findall(r"[A-Za-z0-9]+", text)]


def get_texts_chunks(chunks: ChunkListModel) -> List[str]:
    """Return the indexed text of every chunk, in chunk order."""
    return [chunk_text(c) for c in chunks.chunks]


class HybridIndex:
    """Search index combining BM25 and embedding similarity.

    Attributes:
        chunks: Indexed chunks.
        vectors: Normalised embeddings, row ``i`` belongs to chunk ``i``
            (None when embeddings are disabled).
        model: Embedding model used to encode the queries (None when
            embeddings are disabled).
        bm25: BM25 index built on the chunk texts.
        folder: Root folder the chunks come from.
    """

    def __init__(
            self, chunks: ChunkListModel,
            vectors: Optional[Vectors],
            model: Optional[SentenceTransformer],
            folder: str = "") -> None:
        """Build the BM25 index on top of precomputed vectors.

        Args:
            chunks: Chunks to index.
            vectors: Their normalised embeddings (same order), or None.
            model: Model that produced *vectors*, used for queries, or
                None for a BM25-only index.
            folder: Root folder the chunks come from.
        """
        self.folder = folder
        self.chunks = chunks
        self.vectors = vectors
        self.model = model
        self.bm25 = BM25Okapi(
            [tokenize(t) for t in get_texts_chunks(chunks)])

    @classmethod
    def from_folder(
            cls, folder: str, model_name: str,
            use_embedding: bool = USE_EMBEDDING,
            device: str = AMBEDING_DEVICE) -> "HybridIndex":
        """Build the index of *folder*, re-encoding only modified files.

        Args:
            folder: Root folder of the code to index.
            model_name: Hugging Face name of the embedding model.
            use_embedding: False to skip the embedding model entirely
                (BM25 only: faster, less GPU memory, lower quality).
            device: Where the embedding model runs ("cpu" or "cuda").

        Returns:
            The ready-to-use index.
        """
        model = (SentenceTransformer(model_name, device=device)
                 if use_embedding else None)
        chunks, vectors = update_index(folder, model, model_name)
        return cls(chunks, vectors, model, folder)

    def search(
            self, query: str,
            k: int = 5, pool: int = 100,
            rrf_k: int = 10) -> List[ChunkModel]:
        """Return the *k* chunks most relevant to *query*.

        BM25 and dense rankings are each cut to *pool* results, then
        merged with Reciprocal Rank Fusion: ``score = sum 1 / (rrf_k + rank)``.
        Without embeddings, the BM25 ranking is returned directly.

        Args:
            query: Natural-language question.
            k: Number of chunks to return.
            pool: Number of candidates kept from each ranking.
            rrf_k: RRF smoothing constant (lower = favours top ranks).

        Returns:
            The best chunks, most relevant first.
        """
        bm25_scores = self.bm25.get_scores(tokenize(query))
        bm25_rank = np.argsort(bm25_scores)[::-1][:pool]
        if self.model is None or self.vectors is None:
            return [self.chunks.chunks[i] for i in bm25_rank[:k]]

        q = self.model.encode(query, normalize_embeddings=True,
                              show_progress_bar=False)
        dense_rank = np.argsort(self.vectors @ q)[::-1][:pool]

        fused: dict[int, float] = {}
        for ranking in (bm25_rank, dense_rank):
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)

        best = sorted(fused, key=lambda i: fused[i], reverse=True)[:k]
        return [self.chunks.chunks[i] for i in best]
