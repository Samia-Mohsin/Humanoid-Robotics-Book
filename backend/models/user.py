from sqlalchemy import Column, Integer, String, DateTime, Boolean, JSON, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from core.database import Base
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from uuid import uuid4
import json

class User(Base):
    """
    User model representing platform users and their preferences
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True, default=lambda: str(uuid4()))
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    preferences = Column(JSON, default=dict)  # JSON field for user preferences
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}', name='{self.name}')>"

    # Relationships
    chat_sessions = relationship("ChatSession", back_populates="user")
    profile = relationship("UserProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")
    chapters = relationship("ChapterContent", back_populates="user")
    translation_jobs = relationship("TranslationJob", back_populates="user")
    ingestion_logs = relationship("IngestionLog", back_populates="user")
    progress_records = relationship("UserProgress", back_populates="user")

# Pydantic schemas for API requests/responses
class UserBase(BaseModel):
    email: str
    name: Optional[str] = None

class UserCreate(UserBase):
    password: str
    preferences: Optional[Dict[str, Any]] = Field(default_factory=dict)

class UserUpdate(BaseModel):
    name: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None

class UserInDB(UserBase):
    id: int
    user_id: str
    is_active: bool
    is_admin: bool
    preferences: Dict[str, Any]
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True

class UserPublic(BaseModel):
    id: int
    user_id: str
    email: str
    name: Optional[str]
    is_active: bool
    preferences: Dict[str, Any]
    created_at: str

    class Config:
        from_attributes = True