from pydantic import BaseModel, Field, EmailStr
from typing import Optional, Dict, Any
from datetime import datetime

class UserProfileBase(BaseModel):
    email: EmailStr = Field(..., description="User's email address", example="user@example.com")
    name: Optional[str] = Field(None, description="User's display name", example="John Doe")

class UserProfileCreate(UserProfileBase):
    password: str = Field(..., description="User's password", min_length=8)
    preferences: Optional[Dict[str, Any]] = Field(default={}, description="User preferences and settings")

class UserProfileUpdate(BaseModel):
    name: Optional[str] = Field(None, description="Updated display name", example="John Smith")
    preferences: Optional[Dict[str, Any]] = Field(None, description="Updated user preferences and settings")

class UserProfileInDB(UserProfileBase):
    id: int
    user_id: str
    is_active: bool
    is_admin: bool
    preferences: Dict[str, Any]
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True

class UserProfileResponse(BaseModel):
    user_id: str = Field(..., description="Unique user identifier", example="user-123")
    email: EmailStr = Field(..., description="User's email address", example="user@example.com")
    name: Optional[str] = Field(None, description="User's display name", example="John Doe")
    preferences: Dict[str, Any] = Field(default={}, description="User preferences and settings")
    is_active: bool = Field(..., description="Whether the account is active", example=True)
    is_admin: bool = Field(..., description="Whether the user has admin privileges", example=False)
    created_at: str = Field(..., description="Account creation timestamp", example="2025-12-14T08:00:00Z")
    updated_at: str = Field(..., description="Last profile update timestamp", example="2025-12-14T09:30:00Z")

    class Config:
        from_attributes = True

class UpdateUserProfileRequest(BaseModel):
    preferences: Optional[Dict[str, Any]] = Field(None, description="Updated user preferences and settings to save")

    class Config:
        json_schema_extra = {
            "example": {
                "preferences": {
                    "learning_style": "visual",
                    "language_preferences": ["en", "ur"],
                    "accessibility_settings": {
                        "high_contrast": False,
                        "font_size": "normal",
                        "screen_reader": False,
                        "reduced_motion": True
                    }
                }
            }
        }