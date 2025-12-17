# AI Native Book - RAG Chatbot

This website is built using [Docusaurus](https://docusaurus.io/), a modern static website generator. It includes a RAG (Retrieval-Augmented Generation) chatbot that can answer questions about the content in your book.

## Features

- RAG-based chatbot for interactive book Q&A
- Support for both local open-source models and cloud services
- Document ingestion from Markdown and PDF files
- Chat history persistence
- Multiple chat modes (full context and selected text only)

## Installation

### Prerequisites

- Node.js (v18 or higher)
- Python (v3.9 or higher)
- Docker and Docker Compose (for containerized setup)

### Quick Setup with Docker (Recommended)

1. Clone the repository
2. Make sure you have documents in the `docs/` directory
3. Run the application using Docker Compose:

```bash
docker-compose up -d
```

This will start:
- Qdrant vector database
- FastAPI backend for the chatbot
- Docusaurus web application

### Manual Setup

1. Install Node.js dependencies:
```bash
yarn
```

2. Install Python backend dependencies:
```bash
cd api/chatbot
pip install -r requirements.txt
```

3. Start Qdrant (if not using cloud service):
```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant
```

4. Start the backend:
```bash
cd api/chatbot
python start_server.py
```

5. In a new terminal, start the frontend:
```bash
yarn start
```

## Configuration

### Using Free/Open-Source Models (Default)

The application is configured by default to use free, open-source alternatives:

```env
USE_OPENAI=false
QDRANT_URL=http://localhost:6333
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=microsoft/DialoGPT-medium
DATABASE_URL=sqlite:///./chatbot.db
```

### Using Cloud Services (Optional)

To use cloud services like OpenAI and Qdrant Cloud, update your `.env` file:

```env
USE_OPENAI=true
OPENAI_API_KEY=your_openai_api_key
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
```

## Environment Variables

The application supports the following environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_OPENAI` | `false` | Whether to use OpenAI services (true) or open-source models (false) |
| `QDRANT_URL` | `http://localhost:6333` | URL for Qdrant vector database |
| `QDRANT_API_KEY` | (empty) | API key for Qdrant (not needed for local instance) |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Model for generating text embeddings |
| `LLM_MODEL` | `microsoft/DialoGPT-medium` | Model for text generation |
| `DATABASE_URL` | `sqlite:///./chatbot.db` | Database connection string |

## Usage

### 1. Ingest Documents

Before the chatbot can answer questions, you need to upload your book content:

1. Place your Markdown (.md/.mdx) or PDF files in the `/docs` directory
2. Call the ingestion endpoint:

```bash
curl -X POST http://localhost:8000/ingest
```

### 2. Chat with the Bot

Once documents are ingested, you can interact with the chatbot through:

1. The web interface (Docusaurus app) - chatbot button appears on pages
2. Direct API calls to `POST http://localhost:8000/chat`

### 3. API Endpoints

The backend exposes the following endpoints:

- `GET /health` - Check service status
- `POST /ingest` - Ingest documents from the docs directory
- `POST /chat` - Chat using full document context
- `POST /chat/selected` - Chat using only selected text context

## Local Development

```bash
yarn start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

## Build for Production

```bash
yarn build
```

This command generates static content into the `build` directory and can be served using any static content hosting service.

## Deployment

### Docker Deployment

For easy deployment to any cloud platform that supports Docker:

```bash
docker-compose up -d --build
```

### Vercel Deployment

1. Push your code to a GitHub repository
2. Connect your repository to Vercel
3. Make sure the environment variables are set in Vercel Dashboard (Settings > Environment Variables)
4. Vercel will automatically build and deploy your Docusaurus site

When deploying the backend service, make sure to:
- Set the same environment variables as in your `.env` file
- Ensure your `/docs` directory is available for document ingestion

## Technical Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Frontend      │ ──> │   FastAPI       │ ──> │   Qdrant        │
│   (Docusaurus)  │     │   Backend       │     │   (Vector DB)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                   │
                                   ▼
                         ┌─────────────────┐
                         │ SQLite DB       │
                         │ (Chat History)  │
                         └─────────────────┘
```

The application uses an open-source stack with components that can run locally without external dependencies.
