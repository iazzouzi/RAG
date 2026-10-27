*This project has been created as part of the 42 curriculum by iazzouzi.*

# RAG against the machine

## Description

A Retrieval-Augmented Generation (RAG) system that answers questions about a codebase (the vLLM 0.10.1 repository). It indexes the repository, retrieves the most relevant snippets for a question, and hands them to a small local model (`Qwen/Qwen3-0.6B`) that writes a grounded answer. Retrieval quality is measured with recall@k.

Features:

- Two chunking strategies (Python code, Markdown/text) with a configurable chunk size (default 2000 characters, never above).
- Lexical retrieval with **BM25** (bonus: semantic retrieval with **all-MiniLM-L6-v2 + FAISS**, and **hybrid** fusion of both).
- Single-query and batch (dataset) search, answer generation, and a built-in `evaluate` command computing recall@k.
- Pydantic data models for every structure exchanged between stages.
- Fire CLI, tqdm progress bars, graceful handling of bad input.

## Instructions

### Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- Enough free disk space for dependencies and model weights (several GB).

### Installation

```bash
make install        # uv sync, unzips data/vllm-0.10.1.zip into data/raw if present
```

### Expected layout

```
src/                      # Python module (python -m src)
pyproject.toml, uv.lock
data/raw/                 # vLLM repository (indexed corpus)
data/processed/           # index produced by `index`
data/datasets/UnansweredQuestions/, data/datasets/AnsweredQuestions/
data/output/search_results/<DatasetScope>/
data/output/search_results_and_answer/<DatasetScope>/
```

### Makefile rules

| Rule | Purpose |
|------|---------|
| `make install` | Install dependencies and prepare `data/` |
| `make run` | Full pipeline: index, search dataset, moulinette, evaluate, answer |
| `make debug` | Run `src` under `pdb` |
| `make clean` | Remove `__pycache__` and `.mypy_cache` |
| `make lint` | `flake8` + `mypy` (with the required flags) |

### CLI

All commands run as `uv run python -m src <command> [options]`.

| Command | Description |
|---------|-------------|
| `index --max_chunk_size N` | Chunk `data/raw/` and build the indexes in `data/processed/` (N between 200 and 2000) |
| `search <query> --k K` | Print the top-K sources for one query |
| `search_dataset --dataset_path P --k K --save_directory D` | Search a whole dataset, write a `StudentSearchResults` JSON |
| `answer <query> --k K` | Retrieve, then answer one query |
| `answer_dataset --student_search_results_path P --save_directory D` | Generate answers, write a `StudentSearchResultsAndAnswer` JSON |
| `evaluate --student_search_results_path P --dataset_path G` | Print recall@1/3/5/10 against a ground-truth dataset |

## Example usage

```bash
# 1. Index the corpus
uv run python -m src index --max_chunk_size 2000

# 2. Single query
uv run python -m src search "How to configure the OpenAI compatible server?" --k 5
# data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md [9867:11867]
# ...

# 3. Search a dataset
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions

# 4. Evaluate retrieval
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json

# 5. Generate answers
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

## System architecture

```
                 ┌────────────┐   chunks.json
 data/raw/ ───▶  │  Chunker   │ ──────────────┐
 (.py .md .txt)  └────────────┘               │
                       │ chunk texts          ▼
                       ├──────────▶ BM25 index (bm25s)      ┐
                       └──────────▶ FAISS index (MiniLM)    ┤  data/processed/
                                                            │
 question ──▶ HybridRetriever ── BM25 top-N ─┐              │
                                             ├─▶ RRF ─▶ top-k sources
                              ── FAISS top-N ┘     (file, start, end)
                                                       │
                                      read chunk text from data/raw
                                                       ▼
                                           LLM (Qwen3-0.6B) ─▶ answer
```

Modules (`src/`):

- `__main__.py`: Fire CLI (`RAG` class) exposing all commands and the `evaluate` logic (IoU ≥ 0.05 overlap on same file).
- `index.py`: `Index` class: reads files, chunks them, saves `chunks.json`, builds the BM25 and FAISS indexes.
- `retrieval.py`: `HybridRetriever`: loads the indexes, runs both searches and fuses the rankings.
- `llm.py`: `LLM` wrapper around Qwen/Qwen3-0.6B for answer generation.
- `models.py`: pydantic models (`MinimalSource`, `UnansweredQuestion`, `AnsweredQuestion`, `RagDataset`, `MinimalSearchResults`, `MinimalAnswer`, `StudentSearchResults`, `StudentSearchResultsAndAnswer`, plus the internal `Chunk`).

## Chunking strategy

Files with extensions `.py`, `.md` and `.txt` under `data/raw/` are indexed. Chunking uses LangChain's `RecursiveCharacterTextSplitter` with a strategy per file type:

- **Python**: `Language.PYTHON` separators: the splitter tries to cut on class and function definitions first, then on blank lines, lines and finally words, so chunks tend to align with logical code units.
- **Markdown**: `Language.MARKDOWN` separators: cuts on headers, code fences and horizontal rules first, keeping sections together.
- **Text**: generic recursive splitter (paragraphs, lines, words).

Each chunk stores `file_path`, `first_character_index`, `last_character_index` (computed from `start_index` + chunk length) and its text. File reading uses `newline=""` so character offsets match exactly what the grader and the answer stage read back. `--max_chunk_size` is validated between 200 and 2000 so no retrieved source can exceed the 2000-character `max_context_length`.

## Retrieval method

1. **BM25 (lexical)**: `bm25s` with the Snowball English stemmer (PyStemmer). Each chunk is indexed as `"<file_path>\n<chunk text>"`, so file and directory names (e.g. `lora`, `openai_compatible_server`) are searchable, which helps docs questions that mention a feature name.
2. **Dense retrieval (semantic)**: chunks are embedded with `all-MiniLM-L6-v2` (CPU) and stored in a FAISS `IndexFlatIP`; queries are embedded the same way. This catches paraphrased questions that share few words with the source.
3. **Hybrid fusion**: each retriever returns its top `10·k` candidates; they are merged with weighted **Reciprocal Rank Fusion**: `score(c) = Σ w / (10 + rank)` with `w = 2.0` for BM25 and `w = 1.0` for FAISS. BM25 is weighted higher because identifiers and exact terms matter a lot for code questions. The top-k fused chunks are returned as `(file_path, first_character_index, last_character_index)`.

## Answer generation

The retrieved character ranges are read back from the original files and concatenated into the prompt, which instructs `Qwen/Qwen3-0.6B` to answer using only the provided context. Thinking mode is disabled, decoding is greedy (`do_sample=False`) for reproducibility, and generation is capped at 256 new tokens.

## Performance analysis

Measured with the moulinette on the public datasets (`--k 10`, `--max_chunk_size 2000`, 14,638 chunks indexed):

| Metric | Docs (100 questions) | Code (99 questions) | Requirement |
|--------|------|------|-------------|
| Recall@1 | 57.0 % | 32.3 % | n/a |
| Recall@3 | 85.0 % | 58.6 % | n/a |
| Recall@5 | **88.0 %** | **66.7 %** | ≥ 80 % docs / ≥ 50 % code |
| Recall@10 | 92.0 % | 73.7 % | n/a |

Both Recall@5 thresholds are met with margin (+8 points on docs, +16.7 on code).

| Constraint | Requirement | Measured |
|------------|-------------|----------|
| Indexing time (whole corpus) | ≤ 5 min | ~3 min 11 s (full `index` run: chunking, BM25, embeddings, FAISS) |
| Retrieval of 200 questions | ≤ 90 s | ~1 s per 100 questions (≈90-150 questions/s), excluding model load |

Observations:

- Recall climbs sharply between @1 and @3 (57 → 85 % on docs), meaning the right chunk is usually retrieved but not always ranked first. Fusion and re-weighting matter most at the top ranks.
- Code questions are harder: relevant answers are spread over many small functions and identifiers, so recall@1 is much lower than on docs, though it still reaches 66.7 % at @5.
- Retrieval is fast because both indexes are precomputed and loaded once per run; the embedding model is only used to encode the query.

Chunk size effect (`--max_chunk_size`): smaller chunks give more precise spans but split context; larger chunks keep context but dilute BM25 and embedding signals. _Add your own recall@5 comparison for e.g. 500 / 1000 / 2000 here if you ran one._

## Design decisions

- **Hybrid over single method**: BM25 handles exact identifiers; embeddings handle paraphrases. RRF fuses ranks without needing to normalise incompatible scores.
- **Path in indexed text**: prepending the file path gives lexical hooks for docs questions.
- **Index persisted to disk**: `chunks.json` (metadata) + `bm25/` + `faiss`, so querying never re-chunks or re-embeds the corpus.
- **Chunk text not duplicated in outputs**: results only carry source locations; text is re-read from `data/raw` when answering.
- **Small deterministic LLM setup**: greedy decoding and no thinking mode keep outputs stable and fast on CPU with a 0.6B model.
- **Defensive CLI**: empty queries, `k <= 0`, missing files and malformed JSON exit with a clear error message instead of a traceback.

## Challenges faced

- **Keeping spans valid**: recall depends on exact character offsets, so files are read with `newline=""` and offsets come from the splitter's `start_index`.
- **Code vs docs recall**: code questions often quote identifiers verbatim while docs paraphrase; combining BM25 and embeddings with a BM25-favouring weight balanced both.
- **Index build time**: embedding the whole corpus on CPU is the slowest step; batching (64) and a lightweight model keep it within the 5-minute budget.
- **Small model limits**: Qwen3-0.6B reasons weakly, so the prompt is kept short and strictly grounded in the retrieved context.

## Bonus

- Semantic embeddings (all-MiniLM-L6-v2 + FAISS)
- Hybrid retrieval (weighted RRF)

## Resources

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- [The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)
- [Reciprocal Rank Fusion (Cormack et al., 2009)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [bm25s](https://github.com/xhluca/bm25s), [FAISS](https://github.com/facebookresearch/faiss), [Sentence-Transformers](https://www.sbert.net/), [LangChain text splitters](https://python.langchain.com/docs/concepts/text_splitters/)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Pydantic](https://docs.pydantic.dev/), [Python Fire](https://github.com/google/python-fire), [uv](https://docs.astral.sh/uv/)

### Use of AI

AI assistants were used for: brainstorming the pipeline structure and retrieval strategy, explaining library APIs (bm25s, FAISS, LangChain splitters), reviewing code for edge cases, and drafting this README. All generated suggestions were read, tested and understood before being kept, and the final code was reviewed with peers.