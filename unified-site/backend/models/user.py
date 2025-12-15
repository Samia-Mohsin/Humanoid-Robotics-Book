from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from core.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=True)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)

    # User preferences and background information
    programming_languages = Column(JSON, nullable=True)  # e.g., ["Python", "C++", "ROS"]
    ai_ml_experience = Column(String, nullable=True)  # "beginner", "intermediate", "advanced", "expert"
    hardware_experience = Column(Text, nullable=True)  # description of hardware experience
    learning_goals = Column(Text, nullable=True)  # user's learning objectives
    gpu_access = Column(Boolean, default=False)  # whether user has GPU access
    robotics_kit_experience = Column(String, nullable=True)  # experience with robotics kits
    preferred_topics = Column(JSON, nullable=True)  # topics user is most interested in


class UserPreferences(Base):
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    experience_level = Column(String, default="intermediate")  # "beginner", "intermediate", "advanced"
    content_difficulty = Column(String, default="balanced")  # "simplified", "balanced", "advanced"
    preferred_language = Column(String, default="en")  # language preference
    learning_style = Column(String, default="balanced")  # "theoretical", "practical", "balanced"
    notification_preferences = Column(JSON, nullable=True)  # notification settings
    personalization_enabled = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())