from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from typing import Optional
from dotenv import load_dotenv
import httpx

# Load environment variables
load_dotenv()

# Import retrieval functions
from retrieval import get_relevant_chunks, format_context_for_prompt

app = FastAPI(title="Simple RAG Chatbot", description="A RAG chatbot using FastAPI, OpenRouter, and Qdrant", version="1.0.0")


class ChatRequest(BaseModel):
    message: str
    selected_text: Optional[str] = None  # For phase 4


class ChatResponse(BaseModel):
    response: str


@app.get("/")
async def root():
    return {"message": "Welcome to the Simple RAG Chatbot API"}


# Global HTTP client instance to reuse connections
http_client = httpx.AsyncClient(timeout=30.0)


async def call_openrouter_api(message: str):
    """
    Call OpenRouter API with the provided message
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenRouter API key not found in environment variables")

    # Using a free model from OpenRouter - Qwen 2.5 7B
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "qwen/qwen-2.5-7b-instruct",  # Free model
        "messages": [
            {"role": "user", "content": message}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }

    try:
        response = await http_client.post(url, headers=headers, json=data)

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"OpenRouter API error: {response.text}")

        result = response.json()
        return result["choices"][0]["message"]["content"]
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request to OpenRouter API timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


async def call_openrouter_with_rag(query: str, context: str):
    """
    Call OpenRouter API with RAG context
    """
    # Construct the message with context
    full_message = f"Context information:\n{context}\n\nQuestion: {query}\n\nPlease answer the question based only on the provided context information."

    return await call_openrouter_api(full_message)


@app.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    """
    Chat endpoint that connects to OpenRouter API with RAG capabilities
    """
    try:
        # If selected_text is provided, use that instead of RAG retrieval
        if chat_request.selected_text:
            # Phase 4: Use selected_text directly
            context = f"Additional context provided by user:\n{chat_request.selected_text}"
            response = await call_openrouter_with_rag(chat_request.message, context)
        else:
            # Phase 3: Use RAG - retrieve relevant chunks from Qdrant
            relevant_chunks = get_relevant_chunks(chat_request.message)
            context = format_context_for_prompt(relevant_chunks)
            response = await call_openrouter_with_rag(chat_request.message, context)

        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# For Vercel deployment
app_instance = app

# Properly close the HTTP client when the application shuts down
@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)