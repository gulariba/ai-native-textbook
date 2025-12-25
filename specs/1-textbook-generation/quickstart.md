# Quickstart Guide: AI-Native Physical & Humanoid Robotics Textbook

**Date**: 2025-12-08  
**Version**: 1.0  
**Project**: AI-Native Physical & Humanoid Robotics Textbook Generation

## Overview

This guide provides step-by-step instructions to set up, configure, and run the AI-Native Physical & Humanoid Robotics Textbook application with RAG chatbot, personalization, and Urdu translation capabilities.

## Prerequisites

- **Node.js**: v18.x or higher
- **Python**: v3.11 or higher
- **Package Managers**: npm v9+ and pip v22+
- **System Requirements**: 
  - 8GB+ RAM recommended
  - 4+ CPU cores recommended
  - 2GB+ available disk space
- **Access Keys** (for full functionality):
  - OpenRouter API key for LLM access
  - Vector database account (ChromaDB/Pinecone/Weaviate)
  - Cloud storage account (for deployment)

## System Architecture

The application follows a microservices architecture with:
- **Frontend**: React application for user interface
- **Backend**: FastAPI services for AI operations, user management, and content delivery
- **Vector Database**: For RAG functionality
- **Translation Service**: For Urdu localization

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/ai-native-textbook.git
cd ai-native-textbook
git checkout 1-textbook-generation
```

### 2. Set Up Backend

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Set Up Frontend

```bash
# Navigate to frontend directory (from project root)
cd frontend

# Install Node.js dependencies
npm install
```

### 4. Configure Environment Variables

Create `.env` files in both backend and frontend directories:

**Backend `.env`**:
```env
# LLM Configuration
LLM_PROVIDER=openrouter  # or openai, anthropic, huggingface
OPENROUTER_API_KEY=your_openrouter_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Vector Database Configuration
VECTOR_DB_PROVIDER=chromadb  # or pinecone, weaviate
PINECONE_API_KEY=your_pinecone_key_here  # if using Pinecone
CHROMA_DB_PATH=./chroma_data  # if using ChromaDB

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost/textbook_db

# Translation Configuration
URDU_TRANSLATION_MODEL=opus-mt-en-ur

# Application Configuration
APP_ENV=development
DEBUG=true
```

**Frontend `.env`**:
```env
# API Configuration
REACT_APP_API_BASE_URL=http://localhost:8000
REACT_APP_DEBUG=true

# UI Configuration
REACT_APP_DEFAULT_LANGUAGE=en
```

## Running the Application

### 1. Start the Backend Server

```bash
# Ensure you're in the backend directory with venv activated
cd backend

# Run the backend server
uvicorn src.main:app --reload --port 8000
```

The backend will start on `http://localhost:8000` with auto-reload enabled.

### 2. Start the Frontend

```bash
# In a new terminal, navigate to frontend directory
cd frontend

# Start the development server
npm start
```

The frontend will start on `http://localhost:3000` and automatically open in your browser.

## Initial Setup Tasks

### 1. Initialize the Database

After starting the backend, run the database initialization:

```bash
cd backend
python -m src.utils.init_db
```

### 2. Index Textbook Content for RAG

To populate the RAG system with textbook content:

```bash
cd backend
python -m src.services.rag_service --init
```

### 3. Load Sample Textbook Chapters

To load sample textbook content:

```bash
cd backend
python -m src.utils.load_sample_content
```

## Using the Application

### 1. Textbook Navigation

1. Open the frontend at `http://localhost:3000`
2. Browse available chapters in the textbook
3. Access personalized content based on your profile

### 2. RAG Chatbot

1. Navigate to any chapter page
2. Use the chatbot interface to ask questions about the content
3. The RAG system will provide answers based on the current chapter's content

### 3. Personalization

1. Go to your profile settings
2. Set your technical background (beginner, intermediate, advanced)
3. Adjust content preferences
4. The system will adapt content delivery based on your profile

### 4. Urdu Translation

1. Select Urdu as your preferred language in settings
2. Textbook content will be displayed in both English and Urdu
3. RAG chatbot responses will also be available in Urdu

## API Testing

### Test the Backend API

```bash
# Health check
curl http://localhost:8000/health

# Get available chapters
curl http://localhost:8000/api/v1/textbook/chapters

# Test RAG functionality
curl -X POST http://localhost:8000/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key components of a humanoid robot?", "chapter_id": "chapter-1"}'
```

## Deployment

### 1. Building for Production

**Backend**:
```bash
cd backend
pip install -r requirements-prod.txt
python -m src.main:app --host 0.0.0.0 --port 8000
```

**Frontend**:
```bash
cd frontend
npm run build
```

### 2. Deployment Options

- **Frontend**: Deploy static build files to Vercel, Netlify, or GitHub Pages
- **Backend**: Deploy as container to AWS, GCP, Azure, or use serverless options
- **Database**: Use managed PostgreSQL service
- **Vector DB**: Use managed vector database service (Pinecone, Weaviate Cloud)

## Troubleshooting

### Common Issues

1. **Port Already in Use**: Change the port in startup commands
2. **Dependency Issues**: Ensure correct Python/Node.js versions
3. **API Keys Missing**: Check `.env` files are properly configured
4. **Database Connection**: Verify PostgreSQL is running and credentials are correct

### Performance Tips

- Use production-grade vector database (Pinecone/Weaviate) for better RAG performance
- Enable caching layers for frequently accessed content
- Implement CDN for static assets
- Use environment-specific configurations

## Next Steps

1. Customize textbook content for your specific curriculum needs
2. Add new chapters and exercises
3. Fine-tune personalization algorithms based on user feedback
4. Expand language support beyond Urdu
5. Implement advanced analytics for learning insights