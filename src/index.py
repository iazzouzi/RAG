"""Build searchable indexes from the source corpus."""

import json
import bm25s
import Stemmer
import faiss
from pathlib import Path
from .models import Chunk
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from sentence_transformers import SentenceTransformer


class Index:
    """Split source files and build BM25 and FAISS indexes."""

    def __init__(self, chunk_size: int) -> None:
        """Initialize index configuration and build the indexes.

        Args:
            chunk_size: Maximum number of characters in each chunk.
        """

        try:
            self.corpus_path = "data/raw"
            self.save_dir = "data/processed"
            self.chunk_size = chunk_size
            self.py_splitter = self._load_splitter(Language.PYTHON)
            self.md_splitter = self._load_splitter(Language.MARKDOWN)
            self.txt_splitter = self._load_splitter(None)
            self._load()
        except (OSError, RuntimeError, ValueError) as e:
            raise SystemExit(f"Indexing failed: {e}")

    def _load_splitter(
        self, language: Language | None
    ) -> RecursiveCharacterTextSplitter:
        """Create a text splitter for a language or plain text.

        Args:
            language: Language-specific splitter to use, or ``None`` for text.

        Returns:
            A configured recursive character text splitter.
        """
        try:
            if language is None:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    add_start_index=True)
            else:
                splitter = RecursiveCharacterTextSplitter.from_language(
                    chunk_size=self.chunk_size,
                    language=language,
                    add_start_index=True
                )
            return splitter
        except (RuntimeError, ValueError) as e:
            raise SystemExit(
                f"Could not configure text splitter: {e}"
            )

    def _file_chunker(self, file_path: Path) -> list[Chunk]:
        """Read a source file and convert it into indexed chunks.

        Args:
            file_path: Path to the source file.

        Returns:
            Chunks extracted from the file, or an empty list on read failure.
        """
        try:
            with open(file_path, encoding="utf-8", newline="") as file:
                text = file.read()
            ext = file_path.suffix.lower()
            splitter = {
                '.py': self.py_splitter,
                '.md': self.md_splitter,
                '.txt': self.txt_splitter
            }[ext]
            docs = splitter.create_documents([text])
        except (KeyError, OSError, RuntimeError, ValueError) as e:
            print(f"Warning: could not process {file_path}: {e}")
            return []
        chunks: list[Chunk] = []
        for doc in docs:
            chunks.append(Chunk(
                file_path=str(file_path),
                first_character_index=doc.metadata["start_index"],
                last_character_index=(
                    doc.metadata["start_index"] + len(doc.page_content)
                ),
                text=doc.page_content
            ))
        return chunks

    def _bm25_index(self, corpus: list[str]) -> None:
        """Build and save a BM25 index for the corpus.

        Args:
            corpus: Documents to tokenize and index.

        Returns:
            None.
        """
        try:
            stemmer = Stemmer.Stemmer("english")
            corpus_tokens = bm25s.tokenize(texts=corpus, stemmer=stemmer)
            retriever = bm25s.BM25()
            retriever.index(corpus_tokens)
            retriever.save(f"{self.save_dir}/bm25")
        except (OSError, RuntimeError, ValueError) as e:
            raise SystemExit(f"Could not build BM25 index: {e}")

    def _faiss_index(self, corpus: list[str]) -> None:
        """Build and save a FAISS vector index for the corpus.

        Args:
            corpus: Documents to embed and index.

        Returns:
            None.
        """
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            emb = model.encode(
                corpus, batch_size=64, show_progress_bar=True)
            index = faiss.IndexFlatIP(emb.shape[1])
            index.add(emb)
            faiss.write_index(index, f'{self.save_dir}/faiss')
        except (OSError, RuntimeError, ValueError) as e:
            raise SystemExit(
                f"Could not build FAISS index: {e}"
            )

    def _load(self) -> None:
        """Chunk eligible files, persist metadata, and build search indexes.

        Returns:
            None.
        """
        VALID_EXTENTIONS = {".py", ".md", ".txt"}
        chunks: list[Chunk] = []
        try:
            paths = Path(self.corpus_path).rglob("*")
            for file_path in paths:
                if (
                    file_path.is_file()
                    and file_path.suffix.lower() in VALID_EXTENTIONS
                ):
                    chunks.extend(self._file_chunker(file_path))
        except OSError as e:
            raise SystemExit(f"Could not read corpus: {e}")
        try:
            with open(f"{self.save_dir}/chunks.json", 'w') as file:
                json.dump(
                    [chunk.model_dump() for chunk in chunks], file, indent=2
                )
        except (OSError, TypeError, ValueError) as e:
            raise SystemExit(
                f"Could not save chunk metadata: {e}"
            )
        corpus = [f"{c.file_path}\n{c.text}" for c in chunks]
        self._bm25_index(corpus)
        self._faiss_index(corpus)
        print(
            f"Ingestion complete! Indexed {len(chunks)} under {self.save_dir}"
        )
