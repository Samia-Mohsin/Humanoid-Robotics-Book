from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from core.database import Base
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from uuid import uuid4

class ChapterContent(Base):
    """
    Chapter content model representing book chapters with their content and metadata
    """
    __tablename__ = "chapter_content"

    id = Column(Integer, primary_key=True, index=True)
    chapter_id = Column(String, unique=True, index=True, default=lambda: str(uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Nullable for system content
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)  # Original content in source language
    translated_content = Column(JSON, default=dict)  # JSON field for translations
    chapter_metadata = Column(JSON, default=dict)  # JSON field for chapter metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="chapters")
    translation_jobs = relationship("TranslationJob", back_populates="chapter")
    ingestion_logs = relationship("IngestionLog", back_populates="chapter")
    progress_records = relationship("UserProgress", back_populates="chapter")

    def __repr__(self):
        return f"<ChapterContent(id={self.id}, title='{self.title}', chapter_id='{self.chapter_id}')>"

class TranslationJob(Base):
    """
    Translation job model tracking ongoing and completed translation tasks
    """
    __tablename__ = "translation_jobs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, unique=True, index=True, default=lambda: str(uuid4()))
    chapter_id = Column(Integer, ForeignKey("chapter_content.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Nullable for system-initiated jobs
    source_language = Column(String, default="en")
    target_language = Column(String, nullable=False)
    status = Column(String(20), default="pending")  # pending, in_progress, completed, failed
    progress_percentage = Column(Integer, default=0)
    result_url = Column(String, nullable=True)  # URL to access translated content when complete
    error_message = Column(Text, nullable=True)  # Error details if job failed
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    chapter = relationship("ChapterContent", back_populates="translation_jobs")
    user = relationship("User", back_populates="translation_jobs")

    def __repr__(self):
        return f"<TranslationJob(id={self.id}, job_id='{self.job_id}', status='{self.status}')>"

class IngestionLog(Base):
    """
    Ingestion log model tracking content ingestion and indexing operations
    """
    __tablename__ = "ingestion_logs"

    id = Column(Integer, primary_key=True, index=True)
    log_id = Column(String, unique=True, index=True, default=lambda: str(uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Nullable for system operations
    operation_type = Column(String(20), nullable=False)  # add, update, delete, reindex
    content_id = Column(String, nullable=False)  # Identifier of the content being processed
    status = Column(String(20), default="started")  # started, processing, completed, failed
    error_details = Column(Text, nullable=True)  # Details if operation failed
    processed_chunks = Column(Integer, default=0)  # Number of content chunks processed
    total_chunks = Column(Integer, default=0)  # Total number of chunks to process
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    user = relationship("User", back_populates="ingestion_logs")
    chapter = relationship("ChapterContent", back_populates="ingestion_logs")

    def __repr__(self):
        return f"<IngestionLog(id={self.id}, log_id='{self.log_id}', status='{self.status}')>"


# Pydantic schemas for API requests/responses
class ChapterContentBase(BaseModel):
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = {}

class ChapterContentCreate(ChapterContentBase):
    translated_content: Optional[Dict[str, str]] = {}

class ChapterContentUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    translated_content: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, Any]] = None

class ChapterContentInDB(ChapterContentBase):
    id: int
    chapter_id: str
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True

class TranslationJobBase(BaseModel):
    chapter_id: str
    target_language: str
    source_language: Optional[str] = "en"

class TranslationJobCreate(TranslationJobBase):
    pass

class TranslationJobInDB(TranslationJobBase):
    id: int
    job_id: str
    user_id: Optional[int] = None
    status: str
    progress_percentage: int
    result_url: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None

    class Config:
        from_attributes = True

class IngestionLogBase(BaseModel):
    operation_type: str
    content_id: str
    status: str
    processed_chunks: int = 0
    total_chunks: int = 0

class IngestionLogCreate(IngestionLogBase):
    pass

class IngestionLogInDB(IngestionLogBase):
    id: int
    log_id: str
    error_details: Optional[str] = None
    started_at: str
    completed_at: Optional[str] = None

    class Config:
        from_attributes = True