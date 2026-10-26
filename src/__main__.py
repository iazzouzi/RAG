from .index import Index
from .retrieval import HybridRetriever
from .models import (
    MinimalAnswer,
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from fire import Fire
from pathlib import Path
from pydantic import ValidationError
from tqdm import tqdm
from .llm import LLM


class RAG:
    """Expose indexing, retrieval, answering, and evaluation commands."""

    @staticmethod
    def index(max_chunk_size: int) -> None:
        """Index the source corpus using the requested chunk size.

        Args:
            max_chunk_size: Maximum number of characters in each chunk.

        Returns:
            None.
        """
        if max_chunk_size > 2000 or max_chunk_size < 200:
            raise SystemExit(
                'Error: max_chunk_size must be between 200 and 2000')
        Index(chunk_size=max_chunk_size)

    @staticmethod
    def search(query: str, k: int) -> None:
        """Print the top matching source chunks for a query.

        Args:
            query: Text to search for.
            k: Number of source chunks to print.

        Returns:
            None.
        """
        if not query:
            raise SystemExit('Error: Query cannot be empty')
        if k <= 0:
            raise SystemExit('Error: k must be a positive integer')
        retriever = HybridRetriever()
        sources = retriever.hybrid_retrieval(query, k)
        for source in sources:
            first_idx = source.first_character_index
            last_idx = source.last_character_index
            print(f"{source.file_path} [{first_idx}:{last_idx}]")

    @staticmethod
    def search_dataset(
        dataset_path: str, k: int, save_directory: str
    ) -> None:
        """Retrieve sources for every question in a dataset.

        Args:
            dataset_path: Path to the JSON dataset of questions.
            k: Number of sources to retrieve per question.
            save_directory: Directory where the results JSON is written.

        Returns:
            None.
        """
        try:
            dataset = RagDataset.model_validate_json(
                Path(dataset_path).read_text())
        except (OSError, ValidationError) as e:
            raise SystemExit(f'Error: {e}')
        retriever = HybridRetriever()
        results = []
        for q in tqdm(dataset.rag_questions, desc="Searching",
                      unit="question"):
            sources = retriever.hybrid_retrieval(q.question, k)
            results.append(MinimalSearchResults(
                question_id=q.question_id,
                question=q.question,
                retrieved_sources=sources
            ))
        out = Path(save_directory)
        out.mkdir(parents=True, exist_ok=True)
        file = out / Path(dataset_path).name
        try:
            file.write_text(
                StudentSearchResults(
                    search_results=results, k=k
                ).model_dump_json(indent=2))
        except OSError as e:
            raise SystemExit(
                f"Could not save search results: {e}")
        print(f"Saved student_search_results to {file}")

    @staticmethod
    def answer(query: str, k: int) -> None:
        """Generate and print an answer supported by retrieved context.

        Args:
            query: Question to answer.
            k: Number of source chunks to retrieve.

        Returns:
            None.
        """
        retriever = HybridRetriever()
        sources = retriever.hybrid_retrieval(query, k)
        contexts = []
        for source in sources:
            first_idx = source.first_character_index
            last_idx = source.last_character_index
            try:
                with open(
                    source.file_path, encoding="utf-8", newline=""
                ) as input_file:
                    contexts.append(input_file.read()[first_idx:last_idx])
            except OSError as e:
                raise SystemExit(
                    f"Could not read source {source.file_path}: {e}")
        llm = LLM()
        answer = llm.generate(query, contexts)
        print(answer)

    @staticmethod
    def answer_dataset(
        student_search_results_path: str, save_directory: str
    ) -> None:
        """Generate answers for previously retrieved dataset results.

        Args:
            student_search_results_path: Path to retrieved search results.
            save_directory: Directory where the answer JSON is written.

        Returns:
            None.
        """
        try:
            search_results = StudentSearchResults.model_validate_json(
                Path(student_search_results_path).read_text()
            )
        except (ValidationError, OSError) as e:
            raise SystemExit(f'Error: {e}')
        llm = LLM()
        answers = []
        k = search_results.k
        for res in tqdm(search_results.search_results, desc="Answering",
                        unit="question"):
            contexts = []
            for source in res.retrieved_sources:
                first_idx = source.first_character_index
                last_idx = source.last_character_index
                try:
                    with open(
                        source.file_path, encoding="utf-8", newline=""
                    ) as input_file:
                        contexts.append(input_file.read()[first_idx:last_idx])
                except OSError as e:
                    raise SystemExit(
                        f"Could not read source {source.file_path}: {e}")
            answer = llm.generate(res.question, contexts)
            answers.append(MinimalAnswer(
                question_id=res.question_id,
                question=res.question,
                retrieved_sources=res.retrieved_sources,
                answer=answer
            ))
        out = Path(save_directory)
        out.mkdir(parents=True, exist_ok=True)
        output_file = out / Path(student_search_results_path).name
        try:
            output_file.write_text(
                StudentSearchResultsAndAnswer(
                    search_results=answers, k=k
                ).model_dump_json(indent=2))
        except OSError as e:
            raise SystemExit(f"Could not save answers: {e}")
        print(f"Saved student_search_results_and_answer to {output_file}")

    @staticmethod
    def evaluate(
        student_search_results_path: str, dataset_path: str
    ) -> None:
        """Calculate recall at several cutoffs for search results.

        Args:
            student_search_results_path: Path to retrieved search results.
            dataset_path: Path to the ground-truth question dataset.

        Returns:
            None.
        """
        try:
            dataset = RagDataset.model_validate_json(
                Path(dataset_path).read_text())
            search_results = StudentSearchResults.model_validate_json(
                Path(student_search_results_path).read_text())
        except (ValidationError, OSError) as e:
            raise SystemExit(f'Error: {e}')

        def iou(s1: int, e1: int, s2: int, e2: int) -> float:
            """Calculate intersection-over-union for two character spans.

            Args:
                s1: Start offset of the first span.
                e1: End offset of the first span.
                s2: Start offset of the second span.
                e2: End offset of the second span.

            Returns:
                The intersection-over-union score.
            """
            inter = max(0, min(e1, e2) - max(s1, s2))
            union = (e1 - s1) + (e2 - s2) - inter
            return inter / union if union > 0 else 0.0

        retrieved_map = {
            q.question_id: q.retrieved_sources
            for q in search_results.search_results}
        for k in (1, 3, 5, 10):
            q_counter = 0
            total_scores = 0.0
            for q in dataset.rag_questions:
                found = 0
                for q_src in q.sources:
                    for r_src in retrieved_map.get(q.question_id, [])[:k]:
                        s1 = r_src.first_character_index
                        e1 = r_src.last_character_index
                        s2 = q_src.first_character_index
                        e2 = q_src.last_character_index
                        if (
                            r_src.file_path == q_src.file_path
                            and iou(s1, e1, s2, e2) >= 0.05
                        ):
                            found += 1
                            break
                total_scores += found / len(q.sources)
                q_counter += 1
            recall = total_scores / q_counter if q_counter else 0.0
            print(f"Recall@{k}: {recall} ({(recall * 100):.1f}%)")


try:
    Fire(RAG)
except Exception as e:
    raise SystemExit(f"Command failed: {e}")
