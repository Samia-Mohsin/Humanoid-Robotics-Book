from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship to user preferences
    preferences = relationship("UserPreferences", uselist=False, back_populates="user")

    # Relationship to chat sessions
    chat_sessions = relationship("ChatSession", back_populates="user")

class UserPreferences(Base):
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), unique=True, nullable=False)

    # Background information
    experience_level = Column(String, default="intermediate")  # beginner, intermediate, advanced
    content_difficulty = Column(String, default="moderate")   # easy, moderate, difficult
    preferred_language = Column(String, default="en")
    learning_style = Column(String, default="visual")        # visual, auditory, reading, kinesthetic
    programming_languages = Column(JSON, default=[])          # e.g., ["Python", "C++"]
    ai_ml_experience = Column(String, default="intermediate") # none, beginner, intermediate, advanced
    hardware_experience = Column(String, default="beginner")  # none, beginner, intermediate, advanced
    learning_goals = Column(Text)
    gpu_access = Column(Boolean, default=False)
    robotics_kit_experience = Column(String, default="none")  # none, basic, intermediate, advanced
    preferred_topics = Column(JSON, default=[])              # e.g., ["AI", "Simulation", "Control Systems"]

    # Settings
    notification_preferences = Column(JSON, default=lambda: {"email": True, "push": False, "digest": True})
    personalization_enabled = Column(Boolean, default=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship back to user
    user = relationship("User", back_populates="preferences")

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    # Relationship to user and messages
    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session")

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String, nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    selected_text = Column(Text, nullable=True)  # Text that was selected when message was sent
    context_used = Column(JSON, nullable=True)   # Context retrieved from vector store
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationship to session
    session = relationship("ChatSession", back_populates="messages")

# Create engine and session
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:password@localhost/dbname")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency to get DB session
async def get_async_session():
    async with SessionLocal() as session:
        yield session