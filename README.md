# Simple RAG Chatbot

This is a Simple RAG (Retrieval-Augmented Generation) Chatbot built with FastAPI, OpenRouter, and Qdrant.

## Features

- FastAPI backend
- Integration with OpenRouter for free LLM access (Qwen model)
- Qdrant vector database for document storage and retrieval
- Support for both RAG mode and selected_text mode
- Vercel deployment ready

## Phases Implemented

### Phase 1: Setup
- Setting up FastAPI
- Connecting to OpenRouter API
- Calling a free model (Qwen 2.5 7B)
- Creating a simple chat endpoint

### Phase 2: Document Processing
- Loading QWEN.md files
- Creating embeddings
- Storing embeddings in Qdrant

### Phase 3: RAG Implementation
- Retrieval mechanism to find relevant chunks
- Integration with OpenRouter API
- Complete RAG pipeline

### Phase 4: Selected Text Mode
- Input: question + selected_text
- Ignore Qdrant when selected_text is provided
- Answer ONLY using selected_text

## Installation

### Prerequisites

- Python 3.8+
- Pip package manager

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your environment:
   - Copy `.env.example` to `.env`
   - Add your OpenRouter API key and Qdrant credentials to the `.env` file

3. Run the application:
```bash
python main.py
```

4. The API will be available at `http://localhost:8000`

## API Endpoints

- `GET /` - Health check endpoint
- `POST /chat` - Chat endpoint with RAG capabilities

## Environment Variables

- `OPENROUTER_API_KEY` - Your OpenRouter API key (get one at https://openrouter.ai/)
- `QDRANT_API_KEY` - Your Qdrant API key
- `QDRANT_URL` - Your Qdrant cluster URL

## Deployment

This project is configured for Vercel deployment with the vercel.json file.

1. Push your code to the GitHub repository
2. Connect your repository to Vercel
3. Set the environment variables in Vercel dashboard
4. Vercel will automatically build and deploy your application
