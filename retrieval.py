"""
retrieval.py
Utility functions for retrieving relevant chunks from Qdrant
"""
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import numpy as np

# Load environment variables
load_dotenv()

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_relevant_chunks(query: str, top_k: int = 3):
    """
    Retrieve the most relevant chunks for a given query using Qdrant
    """
    try:
        # Create embedding for the query
        query_embedding = embedding_model.encode([query])[0].tolist()
        
        # Search in Qdrant for similar vectors
        search_result = qdrant_client.search(
            collection_name="documents",
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True
        )
        
        # Extract the content from the results
        relevant_chunks = []
        for result in search_result:
            chunk = {
                "content": result.payload["content"],
                "file_path": result.payload["file_path"],
                "score": result.score
            }
            relevant_chunks.append(chunk)
        
        return relevant_chunks
    except Exception as e:
        print(f"Error retrieving chunks: {e}")
        raise

def format_context_for_prompt(chunks):
    """
    Format the retrieved chunks into a context string for the LLM prompt
    """
    context_parts = []
    
    for i, chunk in enumerate(chunks):
        context_parts.append(f"Document {i+1}:")
        context_parts.append(f"Source: {chunk['file_path']}")
        context_parts.append(f"Content: {chunk['content'][:500]}...")  # Limit content length
        context_parts.append("---")
    
    return "\n".join(context_parts)