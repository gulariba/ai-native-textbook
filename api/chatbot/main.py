from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import asyncio
from typing import List, Dict, Any
import os
import sys
from datetime import datetime
from sqlalchemy.orm import Session

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import Config
from schemas.chat_schemas import (
    ChatRequest, SelectedTextChatRequest, IngestRequest,
    ChatResponse, IngestResponse, HealthCheckResponse
)
from rag_service import RAGService
from models.database import SessionLocal, create_tables, get_db
from services.chat_history_service import ChatHistoryService
from utils.qdrant_manager import QdrantManager


# Global variable to hold the RAG service instance
rag_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler to initialize resources on startup.
    """
    global rag_service
    
    # Initialize the RAG service
    rag_service = RAGService()
    
    # Create database tables
    create_tables()
    
    print("RAG service initialized and database tables created")
    
    yield
    
    # Cleanup when the application shuts down
    print("Shutting down RAG service...")


# Initialize FastAPI app
app = FastAPI(
    title="Book RAG Chatbot API",
    description="API for interacting with the book content using RAG technology",
    version="1.0.0",
    lifespan=lifespan
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.utcnow()
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """
    Ingest all documents from the docs directory into the vector database.
    """
    global rag_service
    
    try:
        # Get the path to the docs directory
        docs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "docs")
        
        if not os.path.exists(docs_path):
            raise HTTPException(status_code=404, detail=f"Docs directory not found at {docs_path}")
        
        # Ingest documents
        result = rag_service.ingest_documents(docs_path, force_recreate=request.force_recreate)
        
        return IngestResponse(
            status="success",
            documents_processed=result["documents_processed"],
            chunks_created=result["chunks_created"],
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting documents: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Chat endpoint that uses the RAG pipeline to answer questions from the book.
    """
    global rag_service

    try:
        # Extract session_id from headers or create a new one
        session_id = request.headers.get("session-id")
        if not session_id:
            session_id = "default-session"  # In production, you'd want to generate a real session ID

        # Get chat history service
        chat_history_service = ChatHistoryService(db)

        # Retrieve previous chat history from the database
        db_messages = chat_history_service.get_session_history(session_id)
        chat_history = [
            {"role": msg.role, "content": msg.content}
            for msg in db_messages
        ]

        # Generate response using RAG
        result = rag_service.generate_response(request.message, chat_history)

        # Store the user message in the database
        chat_history_service.add_message(
            session_id=session_id,
            role="user",
            content=request.message
        )

        # Store the assistant response in the database
        chat_history_service.add_message(
            session_id=session_id,
            role="assistant",
            content=result["response"],
            sources=result.get("sources", [])
        )

        return ChatResponse(
            response=result["response"],
            sources=result.get("sources", []),
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")


@app.post("/chat/selected", response_model=ChatResponse)
async def chat_selected_text(request: SelectedTextChatRequest, db: Session = Depends(get_db)):
    """
    Chat endpoint that uses only the selected text to answer questions.
    """
    global rag_service

    try:
        # Extract session_id from headers or create a new one
        session_id = request.headers.get("session-id")
        if not session_id:
            session_id = "default-session"  # In production, you'd want to generate a real session ID

        # Get chat history service
        chat_history_service = ChatHistoryService(db)

        # Retrieve previous chat history from the database
        db_messages = chat_history_service.get_session_history(session_id)
        chat_history = [
            {"role": msg.role, "content": msg.content}
            for msg in db_messages
        ]

        # Generate response using only selected text
        result = rag_service.generate_response_from_selected_text(
            request.message,
            request.selected_text,
            chat_history
        )

        # Store the user message in the database (with selected text context)
        user_message_with_context = f"Question: {request.message}\nSelected Text: {request.selected_text}"
        chat_history_service.add_message(
            session_id=session_id,
            role="user",
            content=user_message_with_context
        )

        # Store the assistant response in the database
        chat_history_service.add_message(
            session_id=session_id,
            role="assistant",
            content=result["response"],
            sources=result.get("sources", [])
        )

        return ChatResponse(
            response=result["response"],
            sources=result.get("sources", []),
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing selected text chat request: {str(e)}")


@app.get("/")
async def root():
    """
    Root endpoint to verify the API is running.
    """
    return {"message": "Book RAG Chatbot API is running", "version": "1.0.0"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True  # Set to False in production
    )