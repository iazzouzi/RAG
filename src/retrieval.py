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
        return [str(i) for i in ids[0] if i != -1]

    def reciprocal_rank_fusion(self, bm25_rank: list[str], faiss_rank: list[str]) -> list[str]:
        scores: dict[str, float] = {}
        for rank, w in ((bm25_rank, 2.0), (faiss_rank, 1.0)):
            for r, cid in enumerate(rank, start=1):
                scores[cid] = scores.get(cid, 0.0) + w / (10 + r)
        return sorted(scores, key=lambda c: scores[c], reverse=True)

    def hybrid_retrieval(self, query: str, top_k: int) -> list[MinimalSource]:
        bm25_rank = self._bm25_retrieval(query, top_k*10)
        faiss_rank = self._faiss_retrieval(query, top_k*10)
        sources: list[MinimalSource] = []
        for id in self.reciprocal_rank_fusion(bm25_rank, faiss_rank)[:top_k]:
            chunk = self.chunks[int(id)]
            sources.append(MinimalSource(
                file_path=chunk['file_path'],
                first_character_index=chunk['first_character_index'],
                last_character_index=chunk['last_character_index']
            ))
        return sources