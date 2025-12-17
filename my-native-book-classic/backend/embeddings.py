import os
from dotenv import load_dotenv
import logging
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI()

def get_embedding(text: str, model: str = "text-embedding-ada-002") -> list[float]:
    """
    Get embedding for a text using OpenAI's embedding API
    """
    try:
        response = client.embeddings.create(
            input=text,
            model=model
        )

        embedding = response.data[0].embedding
        logger.debug(f"Generated embedding of length {len(embedding)} for text: {text[:50]}...")

        return embedding

    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        raise

def get_embeddings(texts: list[str], model: str = "text-embedding-ada-002") -> list[list[float]]:
    """
    Get embeddings for multiple texts using OpenAI's embedding API
    """
    try:
        response = client.embeddings.create(
            input=texts,
            model=model
        )

        embeddings = [item.embedding for item in response.data]
        logger.debug(f"Generated {len(embeddings)} embeddings")

        return embeddings

    except Exception as e:
        logger.error(f"Error getting embeddings: {str(e)}")
        raise

if __name__ == "__main__":
    # Test the embedding function
    test_text = "This is a test sentence for embedding."
    embedding = get_embedding(test_text)
    print(f"Embedding length: {len(embedding)}")
    print(f"First 10 values: {embedding[:10]}")