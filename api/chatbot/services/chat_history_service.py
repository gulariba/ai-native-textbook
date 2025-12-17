from sqlalchemy.orm import Session
from .models.database import ChatSession, ChatMessage
from typing import List, Optional
import json


class ChatHistoryService:
    def __init__(self, db: Session):
        self.db = db

    def create_session(self) -> ChatSession:
        """Create a new chat session."""
        session = ChatSession()
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a chat session by ID."""
        return self.db.query(ChatSession).filter(ChatSession.session_id == session_id).first()

    def add_message(self, session_id: str, role: str, content: str, sources: Optional[List[dict]] = None) -> ChatMessage:
        """Add a message to a chat session."""
        # Get the session
        session = self.db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
        if not session:
            # Create a new session if it doesn't exist
            session = self.create_session()
            session_id = session.session_id

        # Convert sources to JSON string if provided
        sources_json = json.dumps(sources) if sources else None

        # Create the message
        message = ChatMessage(
            session_id=session.id,
            role=role,
            content=content,
            sources=sources_json
        )
        
        self.db.add(message)
        self.db.commit()
        self.db.refresh(message)
        
        return message

    def get_session_history(self, session_id: str, limit: int = 20) -> List[ChatMessage]:
        """Get chat history for a session."""
        session = self.db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
        if not session:
            return []
        
        messages = self.db.query(ChatMessage)\
            .filter(ChatMessage.session_id == session.id)\
            .order_by(ChatMessage.timestamp)\
            .limit(limit)\
            .all()
        
        return messages

    def get_recent_sessions(self, limit: int = 10) -> List[ChatSession]:
        """Get recent active chat sessions."""
        sessions = self.db.query(ChatSession)\
            .filter(ChatSession.is_active == True)\
            .order_by(ChatSession.updated_at.desc())\
            .limit(limit)\
            .all()
        
        return sessions

    def clear_session_history(self, session_id: str):
        """Clear all messages for a session (but keep the session)."""
        session = self.db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
        if session:
            # Delete all messages for this session
            self.db.query(ChatMessage).filter(ChatMessage.session_id == session.id).delete()
            self.db.commit()