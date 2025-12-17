# Book RAG Chatbot API

This is a FastAPI-based backend service for a Retrieval-Augmented Generation (RAG) chatbot designed to interact with book content. It uses vector embeddings to find relevant information in your book and generates responses using OpenAI's language models.

## Features

- **Document Ingestion**: Automatically processes Markdown and PDF files from the `docs/` directory
- **Vector Storage**: Uses Qdrant for efficient vector similarity search
- **Chat History**: Maintains conversation history with PostgreSQL database
- **Multiple Chat Modes**: Standard chat and selected text chat
- **OpenAI Integration**: Uses OpenAI for embeddings and language generation

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Frontend      │ ──> │   FastAPI       │ ──> │   Qdrant        │
│   (Docusaurus)  │     │   Backend       │     │   (Vector DB)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                   │
                                   ▼
                         ┌─────────────────┐
                         │ PostgreSQL DB   │
                         │ (Chat History)  │
                         └─────────────────┘
```

## Endpoints

### `GET /health`
Check if the service is running.

### `POST /ingest`
Ingest documents from the `docs/` directory into the vector database.
- Request: `{"force_recreate": true/false}`
- Response: Ingestion results

### `POST /chat`
Chat with the RAG system.
- Request: `{"message": "your question", "chat_history": []}`
- Response: AI-generated answer with sources

### `POST /chat/selected`
Chat using only selected text as context.
- Request: `{"message": "your question", "selected_text": "selected text", "chat_history": []}`
- Response: AI-generated answer with sources

## Environment Variables

Create a `.env` file with the following variables:

```bash
# Qdrant configuration
QDRANT_URL=your-qdrant-url
QDRANT_API_KEY=your-qdrant-api-key

# OpenAI configuration
OPENAI_API_KEY=your-openai-api-key

# Database configuration
DATABASE_URL=postgresql://username:password@host:port/database
```

## Setup and Running

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables** by creating a `.env` file

3. **Start the backend service**:
   ```bash
   python start_server.py
   ```

4. The API will be available at `http://localhost:8000`

## Testing

Run the test script to verify all functionality:

```bash
python test_backend.py
```

## Integration with Frontend

The frontend (Docusaurus) communicates with this backend through the following API route:
- `pages/api/chat/route.js` - Connects to the FastAPI backend

Make sure to set the `BACKEND_API_URL` environment variable in your frontend to point to your running backend service.

## Configuration

The system behavior can be configured via `config/settings.py`:

- `CHUNK_SIZE`: Size of text chunks (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `TOP_K`: Number of similar chunks to retrieve (default: 5)
- `SIMILARITY_THRESHOLD`: Minimum similarity threshold (default: 0.5)
- `LLM_MODEL`: OpenAI model to use (default: gpt-4o-mini)

## Troubleshooting

1. **Connection Issues**: Ensure Qdrant and PostgreSQL are running
2. **API Key Issues**: Verify your API keys are correct and have proper permissions
3. **Document Processing**: Check that your documents are in the `docs/` directory and are in Markdown or PDF format

## Development

- The application uses Pydantic for request/response validation
- SQLAlchemy for database operations
- LangChain components for document processing and RAG functionality