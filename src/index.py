import json
from pathlib import Path
from .models import Chunk
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

class Index:

    def __init__(self, chunk_size: str):
        self.corpus_path = "data/raw"

        self.py_splitter = RecursiveCharacterTextSplitter.from_language(
            chunk_size=chunk_size,
            chunk_overlap=chunk_size // 5,
            language=Language.PYTHON,
            add_start_index=True)

        self.md_splitter = RecursiveCharacterTextSplitter.from_language(
            chunk_size=chunk_size,
            chunk_overlap=chunk_size // 5,
            language=Language.MARKDOWN,
            add_start_index=True)

        self.txt_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_size // 5,
            add_start_index=True)

        self._load()

    def file_chunker(self, file_path: Path) -> list[Chunk]:
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

    def _load(self) -> None:
        VALID_EXTENTIONS = {".py", ".md", ".txt"}
        chunks: list[Chunk] = []
        for file_path in Path(self.corpus_path).rglob("*"):
            if file_path.is_file and file_path.suffix.lower() in VALID_EXTENTIONS:
                chunks.extend(self.file_chunker(file_path))
        try:
            with open("data/processed/chunks.json", 'w') as file:
                json.dump([chunk.model_dump() for chunk in chunks], file, indent=4)
        except IOError:
            raise SystemExit('Error')
        print(f"Ingestion complete! Indexed {len(chunks)} under data/processed/")
            


        
                



