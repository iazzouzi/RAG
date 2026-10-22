import tqdm
import json
import bm25s
import Stemmer
import chromadb
from pathlib import Path
from .models import Chunk
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

class Index:

    def __init__(self, chunk_size: str):

        self.corpus_path = "data/raw"

        self.save_dir = "data/processed"

        self.chunk_size = chunk_size

        self.chunk_overlap = chunk_size // 5

        self.py_splitter = self._load_splitter(Language.PYTHON)

        self.md_splitter = self._load_splitter(Language.MARKDOWN)

        self.txt_splitter = self._load_splitter(None)

        self._load()

    def _load_splitter(self, language: Language) -> RecursiveCharacterTextSplitter:
        if not language:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                add_start_index=True)
        else:
            splitter = RecursiveCharacterTextSplitter.from_language(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                language=language,
                add_start_index=True)
        return splitter

    def _file_chunker(self, file_path: Path) -> list[Chunk]:
        try:
            with open(file_path) as file:
                text = file.read()
        except IOError:
            return []
        ext = file_path.suffix.lower()
        ext_map = {
            '.py': self.py_splitter,
            '.md': self.md_splitter,
            '.txt': self.txt_splitter
        }
        docs = ext_map[ext].create_documents([text])
        chunks: list[Chunk] = []
        for doc in docs:
            chunks.append(Chunk(
                file_path=str(file_path),
                first_character_index=doc.metadata["start_index"],
                last_character_index=doc.metadata["start_index"] + len(doc.page_content),
                text=doc.page_content
            ))
        return chunks

    def _bm25_index(self, corpus: list[str]):
        stemmer = Stemmer.Stemmer("english")
        corpus_tokens = bm25s.tokenize(texts=corpus, stemmer=stemmer)
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        retriever.save(f"{self.save_dir}/bm25")

    def _chroma_index(self, corpus: list[str]):
        client = chromadb.PersistentClient(path=f"{self.save_dir}/chroma")
        collection = client.get_or_create_collection("chroma")
        BATCH_SIZE = 100
        for i in tqdm.tqdm(range(0, len(corpus), BATCH_SIZE), desc="Chromadb indexing"):
            batch = corpus[i: i + BATCH_SIZE]
            collection.upsert(
                ids=[str(i + j) for j in range(len(batch))],
                documents=batch)

    def _load(self) -> None:
        VALID_EXTENTIONS = {".py", ".md", ".txt"}
        chunks: list[Chunk] = []
        for file_path in Path(self.corpus_path).rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENTIONS:
                chunks.extend(self._file_chunker(file_path))
        try:
            with open(f"{self.save_dir}/chunks.json", 'w') as file:
                json.dump([chunk.model_dump() for chunk in chunks], file, indent=2)
        except IOError:
            raise SystemExit('Error')
        corpus = [chunk.text for chunk in chunks]
        self._bm25_index(corpus)
        self._chroma_index(corpus)
        print(f"Ingestion complete! Indexed {len(chunks)} under {self.save_dir}")
