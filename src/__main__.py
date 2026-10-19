from .index import Index
from fire import Fire

class RAG:
    @staticmethod
    def index(max_chunk_size: int):
        if max_chunk_size > 2000 or max_chunk_size < 100:
            raise SystemExit('Error') 
        Index(chunk_size=max_chunk_size)

Fire(RAG)

