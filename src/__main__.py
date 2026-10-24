from .index import Index
from .retrieval import HybridRetriever
from .models import RagDataset, MinimalSearchResults, StudentSearchResults
from fire import Fire
from pathlib import Path
from pydantic import ValidationError
import json
from tqdm import tqdm

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
        sources = retriever.hybrid_retrieval(query, k)
        for source in sources:
            print(f"{source.file_path} [{source.first_character_index}:{source.last_character_index}]")

    @staticmethod
    def search_dataset(dataset_path: str, k: int, save_directory: str):
        try:
            dataset = RagDataset.model_validate_json(Path(dataset_path).read_text())
        except (OSError, ValidationError) as e:
            raise SystemExit(e)
        retriever = HybridRetriever()
        results = []
        for q in tqdm(dataset.rag_questions, desc="Searching"):
            sources = retriever.hybrid_retrieval(q.question, k)
            results.append(MinimalSearchResults(
                question_id=q.question_id,
                question=q.question,
                retrieved_sources=sources
            ))
        out = Path(save_directory); out.mkdir(parents=True, exist_ok=True)
        file = out / Path(dataset_path).name
        file.write_text(
            StudentSearchResults(search_results=results, k=k).model_dump_json(indent=2))
        print(f"Saved student_search_results to {file}")

Fire(RAG)

