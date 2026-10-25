import bm25s
import faiss
from .models import MinimalSource
import json
import Stemmer
from sentence_transformers import SentenceTransformer

class HybridRetriever:

    def __init__(self):

        self.chunks = self._chunks_loader("data/processed/chunks.json")

        self.bm25_retriever = self._bm25_loader("data/processed/bm25")

        self.faiss_retriever = self._faiss_loader("data/processed/faiss")

        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def _chunks_loader(self, path: str):
        try:
            with open(path) as file:
                chunks = json.load(file)
        except IOError as e:
            raise SystemExit(e)
        return chunks

    def _bm25_loader(self, path: str):
        return bm25s.BM25.load(path)

    def _faiss_loader(self, path: str):
        return faiss.read_index(path)

    def _bm25_retrieval(self, query: str, top_k: int) -> list[str]:
        stemmer = Stemmer.Stemmer("english")
        q_tokens = bm25s.tokenize(query, stemmer=stemmer)
        ids, _ = self.bm25_retriever.retrieve(q_tokens, k=top_k)
        return [str(i) for i in ids[0]]

    def _faiss_retrieval(self, query: str, top_k: int) -> list[str]:
        q_emb = self.model.encode([query])
        _, ids = self.faiss_retriever.search(q_emb, top_k)
        return [str(i) for i in ids[0]]

    def reciprocal_rank_fusion(self, bm25_ranking: list[str], faiss_ranking: list[str], k: int = 60) -> list[str]:
        scores: dict[str, float] = {}
        for ranking in (bm25_ranking, faiss_ranking):
            for rank, chunk_id in enumerate(ranking, start=1):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        return sorted(scores, key=lambda c: scores[c], reverse=True)

    def hybrid_retrieval(self, query: str, top_k: int) -> list[MinimalSource]:
        bm25_ranking = self._bm25_retrieval(query, top_k*3)
        faiss_ranking = self._faiss_retrieval(query, top_k*3)
        sources: list[MinimalSource] = []
        for id in self.reciprocal_rank_fusion(bm25_ranking, faiss_ranking)[:top_k]:
            chunk = self.chunks[int(id)]
            sources.append(MinimalSource(
                file_path=chunk['file_path'],
                first_character_index=chunk['first_character_index'],
                last_character_index=chunk['last_character_index']
            ))
        return sources