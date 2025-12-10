from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
import os
from qdrant_client import ChromaDBClient
from embeddings import get_embedding

# Load environment variables
load_dotenv()

class RAGSystem:
    def __init__(self):
        # Initialize OpenAI client
        self.openai_client = OpenAI()
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set")

        # Initialize ChromaDB client
        self.vector_client = ChromaDBClient()
    
    def retrieve_context(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve relevant context from ChromaDB based on the query
        """
        query_embedding = get_embedding(query)
        results = self.vector_client.search(query_embedding, top_k=top_k)

        context_list = []
        for result in results:
            context_list.append(result.payload.get("text", ""))

        return context_list

    def retrieve_context_for_selected_text(self, selected_text: str, top_k: int = 5) -> List[str]:
        """
        Retrieve context related to selected text from ChromaDB
        """
        query_embedding = get_embedding(selected_text)
        results = self.vector_client.search(query_embedding, top_k=top_k)

        context_list = []
        for result in results:
            context_list.append(result.payload.get("text", ""))

        # Include the selected text as well
        context_list.insert(0, selected_text)

        return context_list
    
    def generate_answer(self, question: str, context: List[str]) -> str:
        """
        Generate an answer using OpenAI based on question and context
        """
        context_str = "\n".join(context)

        # Create a prompt that combines the question and context
        prompt = f"""
        You are an AI assistant helping users understand a book about AI-native development.
        Use the following context to answer the user's question.
        If the context doesn't contain the information needed to answer the question, say so.
        Be concise and direct in your response.

        Context:
        {context_str}

        Question:
        {question}

        Answer:
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI assistant helping users understand a book about AI-native development. Respond concisely and accurately based on the provided context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=500,
                temperature=0.3,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def ask_question(self, question: str) -> str:
        """
        Answer a question using the entire book
        """
        # Retrieve relevant context from Qdrant
        context = self.retrieve_context(question)
        
        # Generate answer based on context
        answer = self.generate_answer(question, context)
        
        return answer
    
    def ask_with_selected_text(self, question: str, selected_text: str) -> str:
        """
        Answer a question using only the selected text
        """
        # Retrieve context related to the selected text
        context = self.retrieve_context_for_selected_text(selected_text)
        
        # Generate answer based on the selected text and related context
        answer = self.generate_answer(question, context)
        
        return answer