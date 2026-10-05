from typing import List, Literal
from pydantic import BaseModel, Field

SearchStrategy = Literal["semantic", "lexical", "hybrid", "ensemble"]


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)

class AskRequest(BaseModel):
    documents: List[str] = Field(min_length=1, max_length=20)
    question: str = Field(min_length=1, max_length=4000)
    search_strategy: SearchStrategy = "ensemble"
    history: List[ChatTurn] = Field(default_factory=list)

class ParseRequest(BaseModel):
    document_id: str


class ImportRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)


class ParseBatchRequest(BaseModel):
    document_ids: List[str] = Field(min_length=1, max_length=20)

class SearchRequest(BaseModel):
    documents: List[str] = Field(min_length=1, max_length=20)
    query: str = Field(min_length=1, max_length=4000)
    strategy: SearchStrategy = "ensemble"
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results to return")

class MultiSearchRequest(BaseModel):
    documents: List[str] = Field(min_length=1, max_length=20)
    queries: List[str] = Field(min_length=1, max_length=20)
    strategy: SearchStrategy = "ensemble"
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results to return")
