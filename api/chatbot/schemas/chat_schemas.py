from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class ChatRequest(BaseModel):
    message: str = Field(..., description="User's question/message")
    chat_history: Optional[List[dict]] = Field(default_factory=list, description="Previous chat history")


class SelectedTextChatRequest(BaseModel):
    message: str = Field(..., description="User's question")
    selected_text: str = Field(..., description="Text selected by user from the book")
    chat_history: Optional[List[dict]] = Field(default_factory=list, description="Previous chat history")


class IngestRequest(BaseModel):
    force_recreate: bool = Field(default=False, description="Whether to recreate the collection")


class ChatResponse(BaseModel):
    response: str = Field(..., description="Chatbot's response")
    sources: Optional[List[dict]] = Field(default_factory=list, description="List of sources used")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class IngestResponse(BaseModel):
    status: str = Field(..., description="Ingestion status")
    documents_processed: int = Field(..., description="Number of documents processed")
    chunks_created: int = Field(..., description="Number of chunks created")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)