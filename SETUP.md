# Setup Guide

This guide will help you set up the AI Native Book application for development and deployment.

## Prerequisites

- Node.js (v18 or higher)
- Python (v3.9 or higher)
- Docker and Docker Compose (for containerized setup)
- Git

## Quick Start with Docker (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

2. Make sure you have documents in the `docs/` directory

3. Run the entire stack with Docker Compose:
```bash
docker-compose up -d
```

4. Wait for all services to start (this may take a few minutes)

5. Ingest your documents:
```bash
curl -X POST http://localhost:8000/ingest
```

6. Access the application:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - Qdrant UI: http://localhost:6333

## Manual Setup

### 1. Frontend Setup

1. Navigate to the project directory:
```bash
cd D:\AiNativeBook\my-ainative-book\my-native-book-classic
```

2. Install Node.js dependencies:
```bash
yarn install
# or
npm install
```

3. Create a `.env.local` file for development:
```bash
cp .env .env.local
# Edit .env.local with any local-specific settings
```

### 2. Backend Setup

1. Navigate to the backend directory:
```bash
cd api/chatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Your `.env` file is already configured for local development with free services

### 3. Set up Qdrant

Option A: Local Docker container (recommended for development):
```bash
docker run -d --name qdrant-local -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant
```

Option B: Qdrant Cloud (for production):
1. Sign up at [Qdrant Cloud](https://cloud.qdrant.io/)
2. Create a cluster
3. Get your cluster URL and API key
4. Update your `.env` file with the Qdrant Cloud settings

## Running the Applications

### 1. Start the Backend

In the `api/chatbot` directory:
```bash
python start_server.py
```

### 2. Start the Frontend

In the project root directory:
```bash
yarn start
```

## Document Ingestion

Before the chatbot can answer questions, you need to upload your book content:

1. Place your Markdown (.md/.mdx) or PDF files in the `/docs` directory
2. Call the ingestion endpoint:

```bash
curl -X POST http://localhost:8000/ingest
```

Or visit the API documentation at `http://localhost:8000/docs` and use the interactive interface.

## Development Commands

### Frontend
- `yarn start` - Start development server
- `yarn build` - Build for production
- `yarn serve` - Serve production build locally
- `yarn deploy` - Deploy to GitHub Pages

### Backend
- `python start_server.py` - Start the FastAPI backend
- `python test_backend.py` - Run backend tests
- `uvicorn main:app --reload` - Alternative way to start backend with auto-reload

## Configuration Options

### Using Free/Open-Source Models (Default)

The application is configured by default to use free, open-source alternatives:

```
USE_OPENAI=false
QDRANT_URL=http://localhost:6333
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=microsoft/DialoGPT-medium
DATABASE_URL=sqlite:///./chatbot.db
```

### Using Cloud Services (Optional)

To use cloud services like OpenAI and Qdrant Cloud, update your `.env` file:

```
USE_OPENAI=true
OPENAI_API_KEY=your_openai_api_key
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: If you encounter import errors, make sure you've installed all dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **Connection errors**: If the backend can't connect to Qdrant, make sure Qdrant is running:
   ```bash
   docker ps | grep qdrant
   ```

3. **Port already in use**: If you get port binding errors, check if the services are already running or use different ports.

4. **Model loading issues**: For the first run, the open-source models will be downloaded, which may take time and bandwidth.

### Backend API Endpoints

- `GET /health` - Check service status
- `GET /` - Root endpoint confirmation
- `POST /ingest` - Ingest documents from the docs directory
- `POST /chat` - Chat using full document context
- `POST /chat/selected` - Chat using only selected text context
- `GET /docs` - Interactive API documentation

### Frontend Integration

The frontend communicates with the backend through:
- `pages/api/chat/route.js` - API route for chat functionality
- Environment variable `BACKEND_API_URL` determines the backend location

## Updating Dependencies

### Frontend Dependencies
```bash
yarn upgrade-interactive
# or to update all
yarn upgrade
```

### Backend Dependencies
```bash
pip list --outdated
pip install --upgrade -r requirements.txt
```

## Making Changes

1. For frontend changes, edit files in the `src/`, `docs/`, or `blog/` directories
2. For backend changes, edit files in the `api/chatbot/` directory
3. For documentation, edit Markdown files in the `docs/` directory
4. For styling, modify the CSS in `src/css/` or create new components in `src/components/`