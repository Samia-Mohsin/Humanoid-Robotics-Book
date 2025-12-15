from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_async_session
from services.personalization_service import PersonalizationService
from api.deps import get_current_user
from models.user import User

router = APIRouter()

class UpdatePreferencesRequest(BaseModel):
    experience_level: Optional[str] = None
    content_difficulty: Optional[str] = None
    preferred_language: Optional[str] = None
    learning_style: Optional[str] = None
    notification_preferences: Optional[Dict[str, Any]] = None
    personalization_enabled: Optional[bool] = None
    programming_languages: Optional[List[str]] = None
    ai_ml_experience: Optional[str] = None
    hardware_experience: Optional[str] = None
    learning_goals: Optional[str] = None
    gpu_access: Optional[bool] = None
    robotics_kit_experience: Optional[str] = None
    preferred_topics: Optional[List[str]] = None

class PreferencesResponse(BaseModel):
    experience_level: str
    content_difficulty: str
    preferred_language: str
    learning_style: str
    notification_preferences: Dict[str, Any]
    personalization_enabled: bool
    programming_languages: List[str]
    ai_ml_experience: str
    hardware_experience: str
    learning_goals: str
    gpu_access: bool
    robotics_kit_experience: str
    preferred_topics: List[str]

@router.get("/profile/preferences", response_model=PreferencesResponse)
async def get_user_preferences(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Get current user's preferences and background information
    """
    personalization_service = PersonalizationService()

    try:
        preferences = await personalization_service.get_user_preferences(current_user.id)
        return preferences
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/profile/preferences", response_model=PreferencesResponse)
async def update_user_preferences(
    request: UpdatePreferencesRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Update user's preferences and background information
    """
    personalization_service = PersonalizationService()

    try:
        updated_preferences = await personalization_service.update_user_preferences(
            user_id=current_user.id,
            **request.dict(exclude_unset=True)
        )
        return updated_preferences
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))