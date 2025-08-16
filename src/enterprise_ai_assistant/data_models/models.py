from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Literal, Optional, TypedDict
from langchain_core.messages import BaseMessage
from langchain.schema import Document

class VectorStore(BaseModel):
    """
    A vectorstore containing unstructured company documents such as
    policies, reports, manuals, meeting transcripts, and reference
    materials. Supports semantic search and retrieval for answering
    questions based on internal knowledge.
    """
    query: str


class SearchEngine(BaseModel):
    """A search engine for retrieving company-related or external information from the web."""
    query: str


class DataframeCSVAgent(BaseModel):
    """A DataFrame agent for querying structured data from CSV and Excel files."""
    query: str


class SQLDatabaseAgent(BaseModel):
    """An agent for querying structured company data stored in relational databases (MySQL, PostgreSQL, SQLite, etc.)."""
    query: str


class RelevanceGrader(BaseModel):
    """Binary relevance score for a retrieved document vs the query."""
    grade: Literal["relevant", "irrelevant"] = Field(
        ..., description="Use 'relevant' if the context helps answer the query; otherwise 'irrelevant'."
    )    

class HallucinationGrader(BaseModel):
    """Binary score for hallucination in the LLM's response vs context."""
    grade: Literal["yes", "no"] = Field(
        ..., description="'yes' if the response is hallucinated (NOT grounded in context), otherwise 'no'."
    )


class AnswerGrader(BaseModel):
    """Binary score: does the response answer the user's query?"""
    grade: Literal["yes", "no"] = Field(
        ..., description="'yes' if the response answers the query; otherwise 'no'."
    )


class AgentState(TypedDict):
    """Graph state shared by nodes."""
    query: str
    chat_history: List[BaseMessage]
    generation: str
    documents: List[Document]
    # Added key to store the quality gate decision
    quality_gate_decision: str