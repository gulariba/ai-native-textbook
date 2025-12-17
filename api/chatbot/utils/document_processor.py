import os
import re
from typing import List, Dict, Any
import fitz  # PyMuPDF for PDF handling
from pathlib import Path
import markdown


def clean_markdown_text(text: str) -> str:
    """
    Clean markdown text by removing markdown formatting but preserving the content.
    """
    # Remove markdown headers but keep the text
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    
    # Remove bold/italic formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    
    # Remove code blocks but keep the content
    text = re.sub(r'```.*?\n(.*?)```', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]*)`', r'\1', text)
    
    # Remove links but keep the link text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    # Remove images
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'', text)
    
    # Remove quotes
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
    
    # Remove horizontal rules
    text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    
    # Remove extra whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()


def extract_text_from_markdown(file_path: str) -> str:
    """
    Extract clean text from a markdown file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Convert markdown to HTML and then extract text
    html = markdown.markdown(content)
    
    # Remove HTML tags to get plain text
    clean_content = re.sub(r'<[^>]+>', '', html)
    
    return clean_content


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file.
    """
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks.
    """
    if not text.strip():
        return []
        
    # Split text into sentences
    sentences = re.split(r'[.!?]+\s+', text)
    
    chunks = []
    current_chunk = ""
    current_length = 0
    
    for sentence in sentences:
        # Add space if needed
        sentence_with_space = f" {sentence}" if current_chunk else sentence
        
        # Check if adding this sentence would exceed chunk size
        if current_length + len(sentence_with_space) <= chunk_size:
            current_chunk += sentence_with_space
            current_length += len(sentence_with_space)
        else:
            # If the chunk is not empty, save it
            if current_chunk:
                chunks.append({
                    'content': current_chunk.strip(),
                    'length': len(current_chunk)
                })
            
            # Start a new chunk
            # If the sentence is longer than chunk_size, split it
            if len(sentence) > chunk_size:
                # Split the long sentence into smaller parts
                parts = [sentence[i:i+chunk_size] for i in range(0, len(sentence), chunk_size)]
                for i, part in enumerate(parts):
                    if i == len(parts) - 1:  # Last part
                        current_chunk = part
                        current_length = len(part)
                    else:  # All other parts become their own chunks
                        chunks.append({
                            'content': part,
                            'length': len(part)
                        })
            else:
                current_chunk = sentence
                current_length = len(sentence)
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append({
            'content': current_chunk.strip(),
            'length': len(current_chunk)
        })
    
    # Apply overlap
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapping_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapping_chunks.append(chunk)
            else:
                # Take the overlap from the previous chunk
                prev_chunk_end = chunks[i-1]['content'][-chunk_overlap:]
                
                # Combine overlap with current chunk
                new_content = prev_chunk_end + " " + chunk['content']
                overlapping_chunks.append({
                    'content': new_content,
                    'length': len(new_content)
                })
        
        chunks = overlapping_chunks
    
    return chunks


def get_all_documents(docs_path: str) -> List[Dict[str, Any]]:
    """
    Recursively get all markdown and PDF documents from the docs directory.
    """
    documents = []
    docs_dir = Path(docs_path)
    
    for file_path in docs_dir.rglob('*'):
        if file_path.suffix.lower() in ['.md', '.mdx', '.pdf']:
            try:
                if file_path.suffix.lower() in ['.md', '.mdx']:
                    content = extract_text_from_markdown(str(file_path))
                elif file_path.suffix.lower() == '.pdf':
                    content = extract_text_from_pdf(str(file_path))
                
                # Create document object
                doc = {
                    'source': str(file_path),
                    'content': content,
                    'title': file_path.stem
                }
                
                documents.append(doc)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    return documents