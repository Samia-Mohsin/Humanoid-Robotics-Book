from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from core.database import Base

class ContentChunk(Base):
    __tablename__ = "content_chunks"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    source_file = Column(String, nullable=False)  # Path to the original markdown file
    section = Column(String, nullable=True)  # Section/chapter identifier
    page_number = Column(Integer, nullable=True)  # Page number if applicable
    embedding_vector = Column(String, nullable=True)  # Store embedding as JSON string
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    language = Column(String, default="en")  # Language of the content
    difficulty_level = Column(String, default="intermediate")  # "beginner", "intermediate", "advanced"