#!/usr/bin/env python
"""
Start script for the Book RAG Chatbot API
"""
import uvicorn
import sys
import os


def main():
    """
    Main entry point to start the FastAPI server
    """
    # Add the project root to the Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    sys.path.insert(0, project_root)

    # Run the FastAPI application with uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,  # Set to False in production
        log_level="info"
    )


if __name__ == "__main__":
    main()