import bm25s
import chromadb
from .models import MinimalSource
import json
import Stemmer

class HybridRetriever:

    def __init__(self):

        self.chunks = self._chunks_loader("data/processed/chunks.json")

        self.bm25_retriever = self._bm25_loader("data/processed/bm25")

        self.chroma_collection = self._chroma_loader("data/processed/chroma", "chroma")

    def _chunks_loader(self, path: str):
        try:
            with open(path) as file:
                chunks = json.load(file)
        except IOError as e:
            raise SystemExit(e)
        return chunks

    def _bm25_loader(self, path: str):
        return bm25s.BM25.load(path)

    def _chroma_loader(self, path: str, coll_name: str):
        client = chromadb.PersistentClient(path)
        return client.get_collection(coll_name)

    def _bm25_retrieval(self, query: str, top_k: int) -> list[str]:
        stemmer = Stemmer.Stemmer("english")
        q_tokens = bm25s.tokenize(query, stemmer=stemmer)
        lex_ids, lex_scores = self.bm25_retriever.retrieve(q_tokens, k=top_k)
        return [str(idx) for idx in lex_ids[0]]

    def _chroma_retrieval(self, query: str, top_k: int) -> list[str]:
        q_result = self.chroma_collection.query(query_texts=[query], n_results=top_k)
        return q_result['ids'][0]

    def reciprocal_rank_fusion(self, bm25_ranking: list[str], chroma_ranking: list[str], k=60) -> list[str]:
        rrf_scores = {}
        for rank, chunk_id in enumerate(bm25_ranking, start=1):
            rrf_scores[chunk_id] = 1.0 / k + rank
        for rank, chunk_id in enumerate(chroma_ranking, start=1):
            if chunk_id in rrf_scores:
                rrf_scores[chunk_id] += 1.0 / k + rank
            else:
                rrf_scores[chunk_id] = 1.0 / k + rank
        return [item[0] for item in sorted(rrf_scores.items(), key= lambda item: item[1], reverse=True)]

    def _hybrid_retrieval(self, query: str, top_k: int) -> list[MinimalSource]:
        bm25_ranking = self._bm25_retrieval(query, top_k)
        chroma_ranking = self._chroma_retrieval(query, top_k)
        sources: list[MinimalSource] = []
        for id in self.reciprocal_rank_fusion(bm25_ranking, chroma_ranking)[:top_k]:
            chunk = self.chunks[int(id)]
            sources.append(MinimalSource(
                file_path=chunk['file_path'],
                first_character_index=chunk['first_character_index'],
                last_character_index=chunk['last_character_index']
            ))
        return sources