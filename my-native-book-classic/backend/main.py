from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from rag import RAGSystem
from ingest import ingest_documents
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Book RAG API",
    description="API for retrieving and answering questions from book documents using RAG",
    version="1.0.0"
)

# Set up CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG System
rag_system = RAGSystem()

class QuestionRequest(BaseModel):
    question: str

class SelectedTextQuestionRequest(BaseModel):
    question: str
    selected_text: str

@app.get("/")
async def root():
    return {"message": "Book RAG API is running!"}

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    Answer questions using the entire book
    """
    try:
        logger.info(f"Received question: {request.question}")
        answer = rag_system.ask_question(request.question)
        return {"question": request.question, "answer": answer}
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask_selected")
async def ask_with_selected_text(request: SelectedTextQuestionRequest):
    """
    Answer questions using only the user's selected text
    """
    try:
        logger.info(f"Received question: {request.question}")
        logger.info(f"With selected text: {request.selected_text[:100]}...")
        
        answer = rag_system.ask_with_selected_text(
            question=request.question,
            selected_text=request.selected_text
        )
        return {
            "question": request.question,
            "selected_text": request.selected_text,
            "answer": answer
        }
    except Exception as e:
        logger.error(f"Error processing question with selected text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_docs():
    """
    Ingest documents from the docs/ directory into Qdrant
    """
    try:
        logger.info("Starting document ingestion process")
        result = ingest_documents()
        logger.info(f"Ingestion completed. Processed {result['processed']} documents")
        return result
    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)