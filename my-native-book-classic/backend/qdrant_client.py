import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
from dotenv import load_dotenv
import os
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaDBClient:
    def __init__(self, collection_name: str = "book_chunks"):
        """
        Initialize ChromaDB client and create/get collection
        """
        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection_name = collection_name

        # Get or create collection
        try:
            # Try to get existing collection
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"Connected to existing collection '{self.collection_name}'")
        except:
            # Create collection if it doesn't exist
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine distance
            )
            logger.info(f"Created new collection '{self.collection_name}'")

    def upsert_points(self, points: List[Dict[str, Any]]):
        """
        Add multiple points to the collection
        """
        try:
            # Extract data from points
            ids = [point["id"] for point in points]
            embeddings = [point["vector"] for point in points]
            metadatas = [point["payload"] for point in points]

            # Add documents to ChromaDB
            self.collection.add(
                documents=[item["text"] for item in metadatas],
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )

            logger.info(f"Added {len(points)} points to collection '{self.collection_name}'")

        except Exception as e:
            logger.error(f"Error adding points: {str(e)}")
            raise

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Any]:
        """
        Search for similar vectors in the collection
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k
            )

            # Format results to match the expected format
            formatted_results = []
            for i in range(len(results["ids"][0])):
                result = type('Result', (), {})()  # Create a simple object
                result.payload = results["metadatas"][0][i]
                result.distance = results["distances"][0][i] if results["distances"] else 0
                formatted_results.append(result)

            logger.debug(f"Search returned {len(formatted_results)} results")

            return formatted_results

        except Exception as e:
            logger.error(f"Error performing search: {str(e)}")
            raise

if __name__ == "__main__":
    # Test the ChromaDB client
    client = ChromaDBClient()
    print("ChromaDB client initialized successfully")
    print("Collection ensured to exist")