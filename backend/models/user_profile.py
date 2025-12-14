from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from core.database import Base
from pydantic import BaseModel
from typing import Optional, Dict, Any
from uuid import uuid4


class UserProfile(Base):
    """
    User profile model storing extended user information and preferences
    """
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True, nullable=False)  # References the user's UUID
    preferences = Column(JSON, default=dict)  # JSON field for user preferences
    settings = Column(JSON, default=dict)  # Additional user settings
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    timezone = Column(String(50), default="UTC")
    language = Column(String(10), default="en")

    # Relationship
    user = relationship("User", back_populates="profile", foreign_keys=[user_id])

    def __repr__(self):
        return f"<UserProfile(id={self.id}, user_id='{self.user_id}')>"


# Pydantic schemas for API requests/responses
class UserProfileBase(BaseModel):
    preferences: Optional[Dict[str, Any]] = {}
    settings: Optional[Dict[str, Any]] = {}
    timezone: Optional[str] = "UTC"
    language: Optional[str] = "en"


class UserProfileCreate(UserProfileBase):
    user_id: str


class UserProfileUpdate(BaseModel):
    preferences: Optional[Dict[str, Any]] = None
    settings: Optional[Dict[str, Any]] = None
    timezone: Optional[str] = None
    language: Optional[str] = None


class UserProfileInDB(UserProfileBase):
    id: int
    user_id: str
    created_at: str
    updated_at: str
    last_login_at: Optional[str] = None

    class Config:
        from_attributes = True