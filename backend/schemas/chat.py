from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID, uuid4

class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's message to the chatbot", example="What is ROS2?")
    user_id: str = Field(..., description="User identifier", example="user-123")
    conversation_id: Optional[str] = Field(None, description="Optional conversation identifier for continuity", example="conv-456")
    context: Optional[str] = Field("", description="Additional context from selected text", example="I found this concept confusing...")

class ChatResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for the response message", example="msg-789")
    response: str = Field(..., description="The chatbot's response to the user", example="ROS2 is a flexible framework for writing robot applications...")
    timestamp: str = Field(..., description="When the response was generated", example="2025-12-14T10:30:00Z")
    sources: Optional[List[str]] = Field(default=[], description="Sources referenced in the response", example=["/docs/chapter1", "/docs/ros2-intro"])
    context_used: bool = Field(default=False, description="Whether context was used in generating the response", example=True)

class ChatSessionBase(BaseModel):
    title: Optional[str] = Field(None, description="Auto-generated title based on first query")

class ChatSessionCreate(ChatSessionBase):
    user_id: str = Field(..., description="ID of the user creating the session")

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
    role: str = Field(..., description="Role of the message sender", example="user")
    content: str = Field(..., description="Content of the message", example="What is ROS2?")
    context_used: Optional[str] = Field(None, description="Context that was used for this response")
    sources: Optional[str] = Field(None, description="JSON string of source references")
    feedback_score: Optional[int] = Field(None, description="User feedback rating for the response (-1 to 1)")

class ChatMessageCreate(ChatMessageBase):
    session_id: str = Field(..., description="ID of the chat session")
    user_id: int = Field(..., description="ID of the user sending the message")

class ChatMessageInDB(ChatMessageBase):
    id: int
    message_id: str
    session_id: str
    user_id: int
    timestamp: str

    class Config:
        from_attributes = True