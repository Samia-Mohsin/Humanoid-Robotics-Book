from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from core.database import Base
from pydantic import BaseModel, Field
from typing import Optional, List
from uuid import uuid4

class ChatSession(Base):
    """
    Chat session model representing a user's ongoing conversation
    """
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True, default=lambda: str(uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=True)  # Auto-generated from first query
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)  # Optional session expiry

    # Relationship
    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<ChatSession(id={self.id}, user_id={self.user_id}, title='{self.title}')>"

class ChatMessage(Base):
    """
    Chat message model representing individual messages in a conversation
    """
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(String, unique=True, index=True, default=lambda: str(uuid4()))
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    role = Column(String(10), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    context_used = Column(Text, nullable=True)  # Context that was used for this response
    sources = Column(String, nullable=True)  # JSON string of source references
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    feedback_score = Column(Integer, nullable=True)  # User feedback: -1 (negative), 0 (neutral), 1 (positive)

    # Relationships
    session = relationship("ChatSession", back_populates="messages")
    user = relationship("User")

    def __repr__(self):
        return f"<ChatMessage(id={self.id}, role='{self.role}', session_id={self.session_id})>"

# The relationship will be defined in the User model after all models are loaded
# This is a forward reference that will be resolved after all models are defined


# Pydantic schemas for API requests/responses
class ChatSessionBase(BaseModel):
    title: Optional[str] = None

class ChatSessionCreate(ChatSessionBase):
    pass

class ChatSessionInDB(ChatSessionBase):
    id: int
    session_id: str
    user_id: int
    created_at: str
    updated_at: str
    expires_at: Optional[str]

    class Config:
        from_attributes = True

class ChatMessageBase(BaseModel):
    role: str
    content: str
    context_used: Optional[str] = None
    sources: Optional[str] = None
    feedback_score: Optional[int] = None

class ChatMessageCreate(ChatMessageBase):
    session_id: str
    user_id: int

class ChatMessageInDB(ChatMessageBase):
    id: int
    message_id: str
    session_id: str
    user_id: int
    timestamp: str

    class Config:
        from_attributes = True

class ChatRequest(BaseModel):
    message: str
    user_id: str
    conversation_id: Optional[str] = None
    context: Optional[str] = ""

class ChatResponse(BaseModel):
    id: str
    response: str
    timestamp: str
    sources: Optional[List[str]] = []
    context_used: bool = False