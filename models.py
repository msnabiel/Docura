from typing import Union, List
from pydantic import BaseModel, Field
# Pydantic models
class HackRxRunRequest(BaseModel):
    documents: Union[str, List[str]]
    questions: List[str]
    search_strategy: str = Field(default="ensemble", description="Search strategy: semantic, lexical, hybrid, ensemble")

class ParseRequest(BaseModel):
    url: str

class ParseBatchRequest(BaseModel):
    urls: List[str]

class SearchRequest(BaseModel):
    query: str
    strategy: str = Field(default="ensemble", description="Search strategy: semantic, lexical, hybrid, ensemble")
    top_k: int = Field(default=10, description="Number of results to return")

class MultiSearchRequest(BaseModel):
    queries: List[str]
    strategy: str = Field(default="ensemble", description="Search strategy: semantic, lexical, hybrid, ensemble")
    top_k: int = Field(default=10, description="Number of results to return")

class IngestionResult(BaseModel):
    source: str
    success: bool