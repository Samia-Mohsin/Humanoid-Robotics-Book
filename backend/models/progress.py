from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from core.database import Base
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from uuid import uuid4

class UserProgress(Base):
    """
    User progress model tracking user progress through educational content
    """
    __tablename__ = "user_progress"

    id = Column(Integer, primary_key=True, index=True)
    progress_id = Column(String, unique=True, index=True, default=lambda: str(uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    chapter_id = Column(Integer, ForeignKey("chapter_content.id"), nullable=False)
    completion_percentage = Column(Float, default=0.0)  # 0.0 to 100.0
    time_spent_seconds = Column(Integer, default=0)
    last_accessed = Column(DateTime(timezone=True), server_default=func.now())
    bookmarks = Column(JSON, default=list)  # List of positions in content where user bookmarked
    notes = Column(JSON, default=list)  # List of user notes with position and content
    quiz_scores = Column(JSON, default=list)  # List of quiz attempts with scores
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="progress_records")
    chapter = relationship("ChapterContent", back_populates="progress_records")

    def __repr__(self):
        return f"<UserProgress(id={self.id}, user_id={self.user_id}, chapter_id={self.chapter_id}, completion={self.completion_percentage}%)>"



# Pydantic schemas for API requests/responses
class UserProgressBase(BaseModel):
    user_id: str
    chapter_id: str
    completion_percentage: float
    time_spent_seconds: int

class UserProgressCreate(UserProgressBase):
    bookmarks: Optional[List[int]] = []
    notes: Optional[List[Dict[str, Any]]] = []
    quiz_scores: Optional[List[Dict[str, Any]]] = []

class UserProgressUpdate(BaseModel):
    completion_percentage: Optional[float] = None
    time_spent_seconds: Optional[int] = None
    bookmarks: Optional[List[int]] = None
    notes: Optional[List[Dict[str, Any]]] = None
    quiz_scores: Optional[List[Dict[str, Any]]] = None

class UserProgressInDB(UserProgressBase):
    id: int
    progress_id: str
    last_accessed: str
    bookmarks: List[int]
    notes: List[Dict[str, Any]]
    quiz_scores: List[Dict[str, Any]]
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True

class ChapterProgressResponse(BaseModel):
    user_id: str
    chapter_id: str
    completion_percentage: float
    time_spent: int
    last_accessed: str
    bookmarks: List[int]
    notes_count: int
    quiz_attempts: int

    class Config:
        from_attributes = True