from langchain_core.pydantic_v1 import BaseModel, Field


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