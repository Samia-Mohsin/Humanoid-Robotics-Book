from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class TranslateRequest(BaseModel):
    chapter_text: str = Field(..., description="The text content of the chapter to translate", example="ROS2 (Robot Operating System 2) is a flexible framework...")
    target_language: str = Field(..., description="Target language code (e.g., 'ur' for Urdu)", example="ur")
    source_language: Optional[str] = Field("en", description="Source language code (default: English)", example="en")
    user_id: Optional[str] = Field(None, description="Optional user identifier for tracking", example="user-123")
    preserve_format: Optional[bool] = Field(True, description="Whether to preserve original formatting", example=True)

class TranslateResponse(BaseModel):
    chapter_id: str = Field(..., description="ID of the chapter being translated", example="chapter-ros2-intro")
    translated_content: str = Field(..., description="The translated content", example="ROS2 (روبوٹ آپریٹنگ سسٹم 2) ایک لچکدار فریم ورک ہے...")
    target_language: str = Field(..., description="Target language code", example="ur")
    source_language: str = Field(..., description="Source language code", example="en")
    processing_time: float = Field(..., description="Time taken to process the translation in seconds", example=2.5)
    metadata: Optional[dict] = Field(default={}, description="Additional metadata about the translation")

class TranslationJobBase(BaseModel):
    chapter_id: str = Field(..., description="ID of the chapter to translate", example="chapter-1")
    target_language: str = Field(..., description="Target language code", example="ur")
    source_language: Optional[str] = Field("en", description="Source language code", example="en")

class TranslationJobCreate(TranslationJobBase):
    user_id: Optional[str] = Field(None, description="ID of user requesting translation", example="user-123")

class TranslationJobInDB(TranslationJobBase):
    id: int
    job_id: str
    user_id: Optional[int] = None
    status: str = Field(..., description="Current status of the translation job", example="completed")
    progress_percentage: int = Field(..., description="Progress percentage (0-100)", example=100)
    result_url: Optional[str] = Field(None, description="URL to access translated content", example="https://api.example.com/translations/job-123")
    error_message: Optional[str] = Field(None, description="Error details if job failed")
    created_at: str
    completed_at: Optional[str] = None

    class Config:
        from_attributes = True