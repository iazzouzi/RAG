from .index import Index
from .retrieval import HybridRetriever
from fire import Fire

class RAG:
    @staticmethod
    def index(max_chunk_size: int):
        if max_chunk_size > 2000 or max_chunk_size < 100:
            raise SystemExit('Error') 
        Index(chunk_size=max_chunk_size)

    @staticmethod
    def search(query: str, k: int):
        if not query:
            raise SystemExit('Error')
        if k <= 0:
            raise SystemExit('Error')
        retriever = HybridRetriever()
        sources = retriever._hybrid_retrieval(query, k)
        for source in sources:
            print(f"{source.file_path} [{source.first_character_index}:{source.last_character_index}]")

Fire(RAG)

