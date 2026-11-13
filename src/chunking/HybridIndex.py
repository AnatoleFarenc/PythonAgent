from model import ChunkListModel, ChunkModel

import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from typing import List


def tokenize(text: str) -> list[str]:
    """Lowercase tokens; splits snake_case and camelCase."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return [t.lower() for t in re.findall(r"[A-Za-z0-9]+", text)]


class HybridIndex:
    def __init__(
            self, chunks: ChunkListModel,
            texts: List[str], model_name: str) -> None:
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])
        self.model = SentenceTransformer(model_name)
        self.vectors = self.model.encode(
            texts, normalize_embeddings=True,
            batch_size=32, show_progress_bar=False
        )

    def search(
            self, query: str,
            k: int = 5, pool: int = 50,
            rrf_k: int = 60) -> List[ChunkModel]:
        bm25_scores = self.bm25.get_scores(tokenize(query))
        bm25_rank = np.argsort(bm25_scores)[::-1][:pool]

        q = self.model.encode(query, normalize_embeddings=True)
        dense_rank = np.argsort(self.vectors @ q)[::-1][:pool]

        fused: dict[int, float] = {}
        for ranking in (bm25_rank, dense_rank):
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)

        best = sorted(fused, key=lambda i: fused[i], reverse=True)[:k]
        return [self.chunks.chunks[i] for i in best]
