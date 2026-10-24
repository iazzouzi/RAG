from .index import Index
from .retrieval import HybridRetriever
from .models import RagDataset, MinimalSearchResults, StudentSearchResults, MinimalAnswer, StudentSearchResultsAndAnswer
from fire import Fire
from pathlib import Path
from pydantic import ValidationError
import json
from tqdm import tqdm
from .llm import LLM

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

    @staticmethod
    def answer(query: str, k: int):
        retriever = HybridRetriever()
        sources = retriever.hybrid_retrieval(query, k)
        contexts = []
        for source in sources:
            first_idx = source.first_character_index
            last_idx = source.last_character_index
            with open(source.file_path) as file:
                contexts.append(file.read()[first_idx:last_idx+1])
        llm = LLM()
        answer = llm.generate(query, contexts)
        print(answer)

    @staticmethod
    def answer_dataset(student_search_results_path: str, save_directory: str):
        try:
            search_results = StudentSearchResults.model_validate_json(Path(student_search_results_path).read_text())
        except (ValidationError, OSError) as e:
            raise SystemExit(e)
        retriever = HybridRetriever()
        llm = LLM()
        answers = []
        k = search_results.k
        for res in tqdm(search_results.search_results, desc="Answering"):
            sources = retriever.hybrid_retrieval(res.question, k)
            contexts = []
            for source in sources:
                first_idx = source.first_character_index
                last_idx = source.last_character_index
                with open(source.file_path) as file:
                    contexts.append(file.read()[first_idx:last_idx+1])
            answer = llm.generate(res.question, contexts)
            answers.append(MinimalAnswer(
                question_id=res.question_id,
                question=res.question,
                retrieved_sources=res.retrieved_sources,
                answer=answer
            ))
        out = Path(save_directory); out.mkdir(parents=True, exist_ok=True)
        file = out / Path(student_search_results_path).name
        file.write_text(
            StudentSearchResultsAndAnswer(search_results=answers, k=k).model_dump_json(indent=2))
        print(f"Saved student_search_results_and_answer to {file}")

try:
    Fire(RAG)
except BaseException as e:
    print(f'Error: {e}')

