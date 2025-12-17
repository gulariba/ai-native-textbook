"""
Test script for the Book RAG Chatbot API
"""
import requests
import os
import time
import json
from pathlib import Path

# Configuration
BACKEND_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")


def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check endpoint...")
    try:
        response = requests.get(f"{BACKEND_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"[PASS] Health check: {data['status']} (timestamp: {data['timestamp']})")
            return True
        else:
            print(f"[FAIL] Health check failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"[FAIL] Health check error: {e}")
        return False


def test_ingest_documents():
    """Test the ingest documents endpoint"""
    print("Testing document ingestion...")
    try:
        # Get the docs directory path (relative to project root)
        project_root = Path(__file__).parent.parent.parent
        docs_path = project_root / "docs"
        
        if not docs_path.exists():
            print(f"✗ Docs directory does not exist: {docs_path}")
            return False
            
        # Count markdown files in docs directory
        md_files = list(docs_path.rglob("*.md")) + list(docs_path.rglob("*.mdx"))
        if not md_files:
            print(f"✗ No markdown files found in {docs_path}")
            return False
        
        print(f"Found {len(md_files)} markdown files to process")

        # Ingest documents
        response = requests.post(f"{BACKEND_URL}/ingest", json={"force_recreate": True})
        if response.status_code == 200:
            data = response.json()
            print(f"[PASS] Ingestion completed: {data['status']}")
            print(f"  - Documents processed: {data['documents_processed']}")
            print(f"  - Chunks created: {data['chunks_created']}")
            return True
        else:
            print(f"[FAIL] Ingestion failed with status: {response.status_code}, response: {response.text}")
            return False
    except Exception as e:
        print(f"[FAIL] Ingestion error: {e}")
        return False


def test_chat():
    """Test the chat endpoint"""
    print("Testing chat endpoint...")
    try:
        # First, ensure documents are ingested (wait a bit in case ingestion is still processing)
        time.sleep(2)
        
        # Send a test chat message
        test_message = {
            "message": "What is this book about?",
            "chat_history": []
        }
        
        response = requests.post(f"{BACKEND_URL}/chat", json=test_message)
        if response.status_code == 200:
            data = response.json()
            print(f"[PASS] Chat response received")
            print(f"  - Response preview: {data['response'][:100]}...")
            print(f"  - Sources found: {len(data['sources'])}")
            return True
        else:
            print(f"[FAIL] Chat failed with status: {response.status_code}, response: {response.text}")
            return False
    except Exception as e:
        print(f"[FAIL] Chat error: {e}")
        return False


def test_chat_with_selected_text():
    """Test the chat with selected text endpoint"""
    print("Testing chat with selected text endpoint...")
    try:
        # Send a test message with selected text
        test_message = {
            "message": "Can you explain the concept mentioned in this text?",
            "selected_text": "Artificial Intelligence is the simulation of human intelligence processes by computer systems.",
            "chat_history": []
        }
        
        response = requests.post(f"{BACKEND_URL}/chat/selected", json=test_message)
        if response.status_code == 200:
            data = response.json()
            print(f"[PASS] Selected text chat response received")
            print(f"  - Response preview: {data['response'][:100]}...")
            return True
        else:
            print(f"[FAIL] Selected text chat failed with status: {response.status_code}, response: {response.text}")
            return False
    except Exception as e:
        print(f"[FAIL] Selected text chat error: {e}")
        return False


def main():
    """Main test function"""
    print("Starting backend tests...\n")
    
    # Test order: health check first, then ingestion, then chat functionality
    tests = [
        ("Health Check", test_health_check),
        ("Document Ingestion", test_ingest_documents),
        ("Chat Functionality", test_chat),
        ("Chat with Selected Text", test_chat_with_selected_text)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        result = test_func()
        results.append((test_name, result))
    
    print(f"\n--- Test Results ---")
    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    return all_passed


if __name__ == "__main__":
    main()