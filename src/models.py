"""Pydantic models for indexed chunks, datasets, and search results."""

from pydantic import BaseModel, Field
from typing import List
import uuid


class MinimalSource(BaseModel):
    """Identify a source-file span."""

    file_path: str
    first_character_index: int
    last_character_index: int


class Chunk(MinimalSource):
    """Store a source span together with its text."""

    text: str


class UnansweredQuestion(BaseModel):
    """Represent a question without a generated answer."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represent a question with reference sources and an answer."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Represent a collection of answered and unanswered questions."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Store retrieved sources for one question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Store a generated answer with its retrieved sources."""

    answer: str


class StudentSearchResults(BaseModel):
    """Store retrieval results for a dataset and its cutoff."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Store answers generated from dataset retrieval results."""

    search_results: List[MinimalAnswer]
    k: int
