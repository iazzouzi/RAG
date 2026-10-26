"""Retrieve relevant source chunks using BM25 and FAISS."""

from typing import Any, cast
import bm25s
import faiss
from .models import MinimalSource
import json
import Stemmer
from sentence_transformers import SentenceTransformer


class HybridRetriever:
    """Combine lexical and vector retrieval with rank fusion."""

    def __init__(self) -> None:
        """Load persisted indexes and the sentence-embedding model."""

        try:
            self.chunks = self._chunks_loader("data/processed/chunks.json")
            self.bm25_retriever = self._bm25_loader("data/processed/bm25")
            self.faiss_retriever = self._faiss_loader("data/processed/faiss")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except (OSError, RuntimeError, ValueError) as e:
            raise SystemExit(
                f"Could not load retrieval indexes: {e}"
            )

    def _chunks_loader(self, path: str) -> list[dict[str, Any]]:
        """Load chunk metadata from a JSON file.

        Args:
            path: Path to the chunk metadata JSON file.

        Returns:
            The serialized chunk records.
        """
        try:
            with open(path) as file:
                chunks = json.load(file)
        except (OSError, json.JSONDecodeError) as e:
            raise SystemExit(
                f"Could not load chunk metadata: {e}"
            )
        return cast(list[dict[str, Any]], chunks)

    def _bm25_loader(self, path: str) -> Any:
        """Load a persisted BM25 index.

        Args:
            path: Directory containing the saved BM25 index.

        Returns:
            The loaded BM25 retriever.
        """
        try:
            return bm25s.BM25.load(path)
        except (OSError, RuntimeError, ValueError) as e:
            raise SystemExit(f"Could not load BM25 index: {e}")

    def _faiss_loader(self, path: str) -> Any:
        """Load a persisted FAISS index.

        Args:
            path: Path to the saved FAISS index.

        Returns:
            The loaded FAISS index.
        """
        try:
            return faiss.read_index(path)
        except (OSError, RuntimeError, ValueError) as e:
            raise SystemExit(f"Could not load FAISS index: {e}")

    def _bm25_retrieval(self, query: str, top_k: int) -> list[str]:
        """Retrieve chunk identifiers using BM25.

        Args:
            query: Search query.
            top_k: Maximum number of identifiers to return.

        Returns:
            BM25-ranked chunk identifiers.
        """
        try:
            stemmer = Stemmer.Stemmer("english")
            q_tokens = bm25s.tokenize(query, stemmer=stemmer)
            ids, _ = self.bm25_retriever.retrieve(q_tokens, k=top_k)
            return [str(i) for i in ids[0]]
        except (IndexError, RuntimeError, ValueError) as e:
            raise SystemExit(f"BM25 retrieval failed: {e}")

    def _faiss_retrieval(self, query: str, top_k: int) -> list[str]:
        """Retrieve chunk identifiers using vector similarity.

        Args:
            query: Search query.
            top_k: Maximum number of identifiers to return.

        Returns:
            FAISS-ranked chunk identifiers.
        """
        try:
            q_emb = self.model.encode([query])
            _, ids = self.faiss_retriever.search(q_emb, top_k)
            return [str(i) for i in ids[0] if i != -1]
        except (IndexError, RuntimeError, ValueError) as e:
            raise SystemExit(f"FAISS retrieval failed: {e}")

    def _reciprocal_rank_fusion(
            self, bm25_rank: list[str], faiss_rank: list[str]) -> list[str]:
        """Fuse BM25 and FAISS rankings into one ranking.

        Args:
            bm25_rank: Chunk identifiers ordered by BM25 relevance.
            faiss_rank: Chunk identifiers ordered by vector relevance.

        Returns:
            Chunk identifiers ordered by combined relevance.
        """
        scores: dict[str, float] = {}
        for rank, w in ((bm25_rank, 2.0), (faiss_rank, 1.0)):
            for r, cid in enumerate(rank, start=1):
                scores[cid] = scores.get(cid, 0.0) + w / (10 + r)
        return sorted(scores, key=lambda c: scores[c], reverse=True)

    def hybrid_retrieval(self, query: str, top_k: int) -> list[MinimalSource]:
        """Retrieve source metadata using both search indexes.

        Args:
            query: Search query.
            top_k: Number of source records to return.

        Returns:
            The most relevant source metadata records.
        """
        bm25_rank = self._bm25_retrieval(query, top_k*10)
        faiss_rank = self._faiss_retrieval(query, top_k*10)
        sources: list[MinimalSource] = []
        for id in self._reciprocal_rank_fusion(bm25_rank, faiss_rank)[:top_k]:
            try:
                chunk = self.chunks[int(id)]
                sources.append(MinimalSource(
                    file_path=chunk['file_path'],
                    first_character_index=chunk['first_character_index'],
                    last_character_index=chunk['last_character_index']
                ))
            except (IndexError, KeyError, TypeError, ValueError) as e:
                raise SystemExit(
                    f"Invalid retrieved chunk {id}: {e}"
                )
        return sources
