# Deployment Guide

This guide will help you deploy the AI Native Book application to production.

## GitHub Repository Setup

### 1. Initialize the Repository

```bash
# Navigate to your project directory
cd D:\AiNativeBook\my-ainative-book\my-native-book-classic

# Initialize git if not already done
git init

# Add all files to the repository
git add .

# Make your first commit
git commit -m "Initial commit: AI Native Book with RAG Chatbot"
```

### 2. Link to Remote Repository

```bash
# Add your remote repository
git remote add origin https://github.com/your-username/your-repository-name.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Backend Deployment Options

### Option A: Self-Hosting (Recommended for full control)

The backend needs to be deployed separately from the frontend. Here are the options:

#### Deploy on a VPS or Cloud Instance

1. Clone your repository on the server:
```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

2. Set up Python environment:
```bash
cd api/chatbot
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp ../../.env .
# Edit the .env file to match your production settings
```

4. Set up Qdrant (if not using cloud service):
```bash
docker run -d --name qdrant -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant
```

5. Run the backend:
```bash
python start_server.py
```

6. Set up a reverse proxy (like Nginx) to handle requests and SSL.

#### Deploy on Railway (Free tier available)

1. Sign up at [Railway](https://railway.app)
2. Connect your GitHub repository
3. Set environment variables in Railway dashboard
4. Deploy!

### Option B: Containerized Deployment

Use the provided `docker-compose.yml` to deploy the entire stack:

```bash
# Build and start all services
docker-compose up -d --build

# Monitor the logs
docker-compose logs -f
```

## Frontend Deployment to Vercel

### 1. Install Vercel CLI

```bash
npm install -g vercel
```

### 2. Login to Vercel

```bash
vercel login
```

### 3. Deploy to Vercel

```bash
# Navigate to your project root
cd D:\AiNativeBook\my-ainative-book\my-native-book-classic

# Deploy to Vercel
vercel --prod
```

### 4. Set Environment Variables in Vercel Dashboard

After deployment, set these environment variables in your Vercel project settings:

- `BACKEND_API_URL` - URL of your deployed backend (e.g., `https://your-backend.onrender.com`)

## Environment Configuration

### Frontend Environment Variables

For the Vercel deployment, you need to set:

| Variable | Description | Example |
|----------|-------------|---------|
| `BACKEND_API_URL` | URL of your deployed backend service | `https://your-backend.onrender.com` |

### Backend Environment Variables

These should be configured in your backend deployment:

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_OPENAI` | `false` | Whether to use OpenAI services (true) or open-source models (false) |
| `QDRANT_URL` | `http://localhost:6333` | URL for Qdrant vector database |
| `QDRANT_API_KEY` | (empty) | API key for Qdrant (not needed for local instance) |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Model for generating text embeddings |
| `LLM_MODEL` | `microsoft/DialoGPT-medium` | Model for text generation |
| `DATABASE_URL` | `sqlite:///./chatbot.db` | Database connection string |

## Post-Deployment Steps

### 1. Ingest Your Documents

After deploying your backend, you need to ingest your documents:

```bash
curl -X POST https://your-backend-url/ingest
```

### 2. Test the System

1. Visit your Vercel-deployed frontend
2. Try using the chatbot feature
3. Verify that it connects to your backend

### 3. Troubleshooting

If the chatbot doesn't work:

1. Check that your `BACKEND_API_URL` is set correctly in Vercel dashboard
2. Verify that your backend is running and accessible
3. Check the browser console for any errors
4. Verify that CORS settings allow your frontend domain

## Architecture Notes

This application follows a microservices architecture:
- Frontend (Docusaurus) deployed on Vercel
- Backend (FastAPI) deployed separately
- Vector database (Qdrant) for document embeddings
- SQLite for chat history storage

This allows for:
- Independent scaling of components
- Easier maintenance
- Better security (separate concerns)

## Scaling Recommendations

- For production, consider using Qdrant Cloud instead of self-hosting
- Monitor your backend resource usage and scale as needed
- Consider using a managed database service instead of SQLite for production
- Set up monitoring and alerting for both frontend and backend