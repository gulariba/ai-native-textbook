import os
from typing import Optional


class Config:
    # Qdrant configuration - Can use local instance for free
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")  # Local Qdrant for free usage
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")  # Empty for local instance

    # Open-source alternatives for OpenAI
    USE_OPENAI: bool = os.getenv("USE_OPENAI", "false").lower() == "true"

    # If not using OpenAI, use local Hugging Face models
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "microsoft/DialoGPT-medium")  # Or other open-source model

    # OpenAI configuration (only used if USE_OPENAI is True)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Neon Postgres configuration - Using SQLite for free local option
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./chatbot.db")

    # Collection name for storing document embeddings
    COLLECTION_NAME: str = "book_embeddings"

    # Chunk settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Similarity settings
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.5

    # Free/local model settings
    MAX_TOKENS: int = 512
    TEMPERATURE: float = 0.7