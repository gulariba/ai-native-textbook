from typing import List, Dict, Any
from .utils.qdrant_manager import QdrantManager
from .utils.document_processor import get_all_documents, chunk_text
from .config.settings import Config
import logging
import os


class RAGService:
    def __init__(self):
        self.qdrant_manager = QdrantManager()
        self.config = Config()

        # Initialize appropriate client based on configuration
        if self.config.USE_OPENAI and self.config.OPENAI_API_KEY:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
            self.use_openai = True
        else:
            # Use open-source models
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.LLM_MODEL)
            self.model = AutoModelForCausalLM.from_pretrained(self.config.LLM_MODEL)

            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Set up text generation pipeline
            self.generator = pipeline(
                'text-generation',
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )

            self.use_openai = False

    def ingest_documents(self, docs_path: str, force_recreate: bool = False):
        """
        Ingest all documents from the specified path into the vector database.
        """
        # Get all documents from the docs directory
        documents = get_all_documents(docs_path)
        
        print(f"Found {len(documents)} documents to process")
        
        # Prepare chunks
        all_chunks = []
        for doc in documents:
            chunks = chunk_text(
                doc['content'], 
                chunk_size=self.config.CHUNK_SIZE, 
                chunk_overlap=self.config.CHUNK_OVERLAP
            )
            
            # Add document metadata to each chunk
            for chunk in chunks:
                chunk['source'] = doc['source']
                chunk['title'] = doc['title']
            
            all_chunks.extend(chunks)
        
        print(f"Created {len(all_chunks)} chunks from documents")
        
        # Create Qdrant collection (recreate if needed)
        self.qdrant_manager.create_collection(recreate=force_recreate)
        
        # Store chunks in Qdrant
        self.qdrant_manager.store_chunks(all_chunks)
        
        return {
            "documents_processed": len(documents),
            "chunks_created": len(all_chunks)
        }

    def generate_response(self, query: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Generate a response to a query using the RAG pipeline.
        """
        if chat_history is None:
            chat_history = []

        # Search for relevant chunks in the vector database
        relevant_chunks = self.qdrant_manager.search_similar(
            query,
            top_k=self.config.TOP_K
        )

        # Filter chunks based on similarity threshold
        filtered_chunks = [
            chunk for chunk in relevant_chunks
            if chunk['score'] >= self.config.SIMILARITY_THRESHOLD
        ]

        if not filtered_chunks:
            return {
                "response": "I couldn't find relevant information in the book to answer your question.",
                "sources": []
            }

        # Prepare context by joining relevant chunks
        context = "\n\n".join([chunk['content'] for chunk in filtered_chunks])

        # Prepare the prompt for the LLM
        system_prompt = f"""
        You are an AI assistant helping users with questions about the AI Native Development book.
        Answer the user's question based ONLY on the provided context from the book.
        If the answer is not in the context, clearly state that you don't have enough information from the book to answer the question.
        Be concise and helpful in your responses.
        """

        # Add chat history if available
        if self.use_openai:
            messages = [{"role": "system", "content": system_prompt}]

            for message in chat_history:
                messages.append({"role": message.get("role", "user"), "content": message["content"]})

            # Add the current query with context
            messages.append({
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}\n\nPlease answer the question based only on the provided context."
            })

            # Call the OpenAI API
            response = self.openai_client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=messages,
                max_tokens=1000,
                temperature=0.3
            )

            return {
                "response": response.choices[0].message.content,
                "sources": filtered_chunks
            }
        else:
            # Use open-source model
            # Combine system prompt with context and query
            full_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"

            # Generate response using local model
            inputs = self.tokenizer.encode(full_prompt, return_tensors="pt", truncation=True, max_length=512)

            with self.model.device as device:
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + self.config.MAX_TOKENS,
                    temperature=self.config.TEMPERATURE,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True
                )

            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract the generated answer part (after the prompt)
            answer_start = response_text.find("Answer:") + len("Answer:")
            if answer_start > len("Answer:") - 1:  # If "Answer:" was found
                response_text = response_text[answer_start:].strip()
            else:
                # If "Answer:" wasn't properly found, return the continuation part
                response_text = response_text[len(full_prompt):].strip()

            return {
                "response": response_text,
                "sources": filtered_chunks
            }

    def generate_response_from_selected_text(self, query: str, selected_text: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Generate a response using only the selected text as context.
        """
        if chat_history is None:
            chat_history = []

        # Prepare the prompt for the LLM
        system_prompt = f"""
        You are an AI assistant helping users with questions about the AI Native Development book.
        Answer the user's question based ONLY on the provided selected text from the book.
        If the answer is not in the selected text, clearly state that you don't have enough information from the selected text to answer the question.
        Be concise and helpful in your responses.
        """

        if self.use_openai:
            # Add chat history if available
            messages = [{"role": "system", "content": system_prompt}]

            for message in chat_history:
                messages.append({"role": message.get("role", "user"), "content": message["content"]})

            # Add the current query with selected text as context
            messages.append({
                "role": "user",
                "content": f"Selected Text:\n{selected_text}\n\nQuestion: {query}\n\nPlease answer the question based only on the provided selected text."
            })

            # Call the OpenAI API
            response = self.openai_client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=messages,
                max_tokens=1000,
                temperature=0.3
            )

            return {
                "response": response.choices[0].message.content,
                "sources": [{"content": selected_text, "source": "selected_text", "title": "Selected Text", "score": 1.0}]
            }
        else:
            # Use open-source model
            # Combine system prompt with selected text and query
            full_prompt = f"{system_prompt}\n\nSelected Text:\n{selected_text}\n\nQuestion: {query}\n\nAnswer:"

            # Generate response using local model
            inputs = self.tokenizer.encode(full_prompt, return_tensors="pt", truncation=True, max_length=512)

            with self.model.device as device:
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + self.config.MAX_TOKENS,
                    temperature=self.config.TEMPERATURE,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True
                )

            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract the generated answer part (after the prompt)
            answer_start = response_text.find("Answer:") + len("Answer:")
            if answer_start > len("Answer:") - 1:  # If "Answer:" was found
                response_text = response_text[answer_start:].strip()
            else:
                # If "Answer:" wasn't properly found, return the continuation part
                response_text = response_text[len(full_prompt):].strip()

            return {
                "response": response_text,
                "sources": [{"content": selected_text, "source": "selected_text", "title": "Selected Text", "score": 1.0}]
            }