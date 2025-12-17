import os
import glob
import hashlib
from pathlib import Path
from typing import List, Dict, Any
import markdown
from bs4 import BeautifulSoup
from embeddings import get_embedding
from qdrant_client import ChromaDBClient
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_markdown_file(file_path: str) -> str:
    """
    Read and extract text content from a markdown file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Convert markdown to HTML then extract text
    html = markdown.markdown(content)
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text()
    
    return text

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move start position by chunk_size minus overlap
        start += chunk_size - overlap
        
        # Handle the last chunk to ensure we don't miss content
        if start + chunk_size > len(text):
            # If the remaining text is too small for another chunk, include it
            if len(text) - start > overlap:
                chunks.append(text[start:])
            break
    
    return chunks

def generate_document_id(file_path: str, chunk_idx: int = 0) -> str:
    """
    Generate a unique document ID based on file path and chunk index
    """
    hash_input = f"{file_path}_{chunk_idx}".encode('utf-8')
    doc_id = hashlib.md5(hash_input).hexdigest()
    return doc_id

def ingest_documents(docs_dir: str = "../docs") -> Dict[str, Any]:
    """
    Ingest all markdown documents from a directory into ChromaDB
    """
    logger.info(f"Starting ingestion from directory: {docs_dir}")

    # Convert relative path to absolute path
    docs_abs_path = os.path.abspath(docs_dir)
    if not os.path.exists(docs_abs_path):
        raise FileNotFoundError(f"Documents directory does not exist: {docs_abs_path}")

    # Find all markdown files
    md_files = glob.glob(os.path.join(docs_abs_path, "**/*.md"), recursive=True)
    mdx_files = glob.glob(os.path.join(docs_abs_path, "**/*.mdx"), recursive=True)
    all_files = md_files + mdx_files

    if not all_files:
        logger.warning(f"No markdown files found in {docs_abs_path}")
        return {"processed": 0, "indexed": 0, "errors": []}

    logger.info(f"Found {len(all_files)} markdown files to process")

    # Initialize ChromaDB client
    chroma_client = ChromaDBClient()
    indexed_count = 0
    errors = []

    for file_path in all_files:
        try:
            logger.info(f"Processing file: {file_path}")

            # Read the markdown content
            content = read_markdown_file(file_path)

            # Split content into chunks
            chunks = chunk_text(content)
            logger.info(f"Split {file_path} into {len(chunks)} chunks")

            points_to_add = []

            for idx, chunk in enumerate(chunks):
                if chunk.strip():  # Only process non-empty chunks
                    # Generate embedding for the chunk
                    embedding = get_embedding(chunk)

                    # Create point for ChromaDB
                    point = {
                        "id": generate_document_id(file_path, idx),
                        "vector": embedding,
                        "payload": {
                            "text": chunk,
                            "source_file": file_path,
                            "chunk_index": idx
                        }
                    }

                    points_to_add.append(point)

            if points_to_add:
                # Add points to ChromaDB
                chroma_client.upsert_points(points_to_add)
                indexed_count += len(points_to_add)
                logger.info(f"Indexed {len(points_to_add)} chunks from {file_path}")

        except Exception as e:
            error_msg = f"Error processing file {file_path}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)

    logger.info(f"Ingestion completed. Indexed {indexed_count} total chunks from {len(all_files)} files")

    return {
        "processed": len(all_files),
        "indexed": indexed_count,
        "errors": errors
    }

if __name__ == "__main__":
    result = ingest_documents()
    print(f"Processed: {result['processed']} files")
    print(f"Indexed: {result['indexed']} chunks")
    if result['errors']:
        print(f"Errors: {len(result['errors'])}")
        for error in result['errors'][:5]:  # Show first 5 errors
            print(f" - {error}")