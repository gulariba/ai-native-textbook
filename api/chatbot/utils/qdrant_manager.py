from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from typing import List, Dict, Any
import os
from .config.settings import Config


class QdrantManager:
    def __init__(self):
        self.client = QdrantClient(
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY,
            # For local setup, you would use:
            # host="localhost",
            # port=6333
        )
        self.collection_name = Config.COLLECTION_NAME

        # Initialize embedding model based on configuration
        if Config.USE_OPENAI and Config.OPENAI_API_KEY:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
            self.use_openai = True
            self.embedding_model = Config.EMBEDDING_MODEL
        else:
            from sentence_transformers import SentenceTransformer
            self.sentence_transformer = SentenceTransformer(Config.EMBEDDING_MODEL)
            self.use_openai = False

    def create_collection(self, recreate: bool = False):
        """
        Create a Qdrant collection for storing document embeddings.
        """
        # Check if collection exists
        collection_exists = False
        try:
            self.client.get_collection(self.collection_name)
            collection_exists = True
        except:
            collection_exists = False

        if collection_exists and recreate:
            print(f"Recreating collection: {self.collection_name}")
            self.client.delete_collection(self.collection_name)

        if not collection_exists or recreate:
            print(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),  # OpenAI embedding dimensions
            )

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embeddings for a text using OpenAI or sentence transformers.
        """
        if self.use_openai:
            response = self.openai_client.embeddings.create(
                input=text,
                model=Config.EMBEDDING_MODEL
            )
            return response.data[0].embedding
        else:
            # Use sentence transformers for embedding
            embedding = self.sentence_transformer.encode([text])
            return embedding[0].tolist()  # Convert numpy array to list

    def store_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Store document chunks with their embeddings in Qdrant.
        """
        points = []
        for i, chunk in enumerate(chunks):
            # Generate embedding for the chunk content
            embedding = self.embed_text(chunk['content'])

            points.append(models.PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "content": chunk['content'],
                    "source": chunk.get('source', 'unknown'),
                    "title": chunk.get('title', 'unknown'),
                    "chunk_index": i
                }
            ))

        # Upload points to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks in Qdrant based on the query.
        """
        query_embedding = self.embed_text(query)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )

        return [
            {
                "content": hit.payload.get("content", ""),
                "source": hit.payload.get("source", "unknown"),
                "title": hit.payload.get("title", "unknown"),
                "score": hit.score
            }
            for hit in results
        ]

    def clear_collection(self):
        """
        Clear all points from the collection.
        """
        try:
            self.client.delete_collection(self.collection_name)
            print(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            print(f"Error clearing collection: {e}")