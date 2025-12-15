from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from core.database import Base

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)  # Could be foreign key if User table is in same DB
    title = Column(String, nullable=True)  # Auto-generated title from first message
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Boolean, default=True)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, nullable=False)  # Could be foreign key if ChatSession table is in same DB
    user_id = Column(Integer, nullable=False)  # Could be foreign key if User table is in same DB
    role = Column(String, nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    context_used = Column(Text, nullable=True)  # Context retrieved from vector store
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    is_selected_text_query = Column(Boolean, default=False)  # Whether this was based on selected text
    selected_text = Column(Text, nullable=True)  # The selected text that was queried