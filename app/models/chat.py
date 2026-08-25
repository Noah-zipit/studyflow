"""
Pydantic models for chat operations.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class MessageRole(str, Enum):
    """Message roles in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """Chat message model."""
    role: MessageRole = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = Field(None, description="Message timestamp")


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., min_length=1, max_length=2000, description="User question")
    collection_id: Optional[str] = Field(None, description="Document collection to search")
    max_tokens: Optional[int] = Field(default=1000, description="Maximum tokens in response")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Response randomness")
    include_sources: bool = Field(default=True, description="Include source documents in response")


class SourceDocument(BaseModel):
    """Source document reference."""
    filename: str = Field(..., description="Source document filename")
    page: Optional[int] = Field(None, description="Page number")
    chunk_id: str = Field(..., description="Chunk identifier")
    similarity_score: float = Field(..., description="Similarity score")
    content_preview: str = Field(..., description="Preview of relevant content")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    answer: str = Field(..., description="AI-generated answer")
    sources: List[SourceDocument] = Field(default=[], description="Source documents used")
    tokens_used: int = Field(..., description="Tokens consumed")
    response_time: float = Field(..., description="Response time in seconds")
    collection_id: Optional[str] = Field(None, description="Collection used for search")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")


class ChatError(BaseModel):
    """Error model for chat failures."""
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")
    error_code: Optional[str] = Field(None, description="Error code for debugging")