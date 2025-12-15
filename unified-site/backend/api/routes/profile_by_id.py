from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_async_session
from services.personalization_service import PersonalizationService
from api.deps import get_current_user, require_admin_user
from models.user import User

router = APIRouter()

class ProfileResponse(BaseModel):
    id: str
    email: str
    full_name: Optional[str] = None
    experience_level: str = "intermediate"
    programming_languages: List[str] = []
    ai_ml_experience: str = "intermediate"
    hardware_experience: str = "beginner"
    learning_goals: Optional[str] = None
    gpu_access: bool = False
    robotics_kit_experience: Optional[str] = None
    preferred_topics: List[str] = []
    personalization_enabled: bool = True
    is_active: bool = True
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class ProfileUpdateRequest(BaseModel):
    full_name: Optional[str] = None
    experience_level: Optional[str] = None
    programming_languages: Optional[List[str]] = None
    ai_ml_experience: Optional[str] = None
    hardware_experience: Optional[str] = None
    learning_goals: Optional[str] = None
    gpu_access: Optional[bool] = None
    robotics_kit_experience: Optional[str] = None
    preferred_topics: Optional[List[str]] = None
    personalization_enabled: Optional[bool] = None

@router.get("/profile/{user_id}", response_model=ProfileResponse)
async def get_user_profile(
    user_id: str = Path(..., description="User ID"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Get user profile by user ID
    """
    # Users can only access their own profile, or admins can access any profile
    if current_user.id != user_id and not getattr(current_user, 'is_admin', False):
        raise HTTPException(status_code=403, detail="Not authorized to access this profile")

    personalization_service = PersonalizationService()

    try:
        preferences = await personalization_service.get_user_preferences(user_id)

        # Create a response that matches the expected profile structure
        profile_response = ProfileResponse(
            id=user_id,
            email=current_user.email,
            full_name=current_user.full_name,
            experience_level=preferences.get("experience_level", "intermediate"),
            programming_languages=preferences.get("programming_languages", []),
            ai_ml_experience=preferences.get("ai_ml_experience", "intermediate"),
            hardware_experience=preferences.get("hardware_experience", "beginner"),
            learning_goals=preferences.get("learning_goals"),
            gpu_access=preferences.get("gpu_access", False),
            robotics_kit_experience=preferences.get("robotics_kit_experience"),
            preferred_topics=preferences.get("preferred_topics", []),
            personalization_enabled=preferences.get("personalization_enabled", True),
            is_active=current_user.is_active
        )

        return profile_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/profile/{user_id}", response_model=ProfileResponse)
async def update_user_profile(
    user_id: str = Path(..., description="User ID"),
    request: ProfileUpdateRequest = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Update user profile by user ID
    """
    # Users can only update their own profile, or admins can update any profile
    if current_user.id != user_id and not getattr(current_user, 'is_admin', False):
        raise HTTPException(status_code=403, detail="Not authorized to update this profile")

    personalization_service = PersonalizationService()

    try:
        # Prepare update data from the request
        update_data = request.dict(exclude_unset=True) if request else {}

        # Remove user_id from update data as it shouldn't be changed
        update_data.pop('id', None)

        # Update user preferences using the personalization service
        updated_preferences = await personalization_service.update_user_preferences(
            user_id=user_id,
            **update_data
        )

        # Also update user information if provided
        if request and request.full_name:
            # In a real implementation, you would update the user model here
            pass

        # Return updated profile
        profile_response = ProfileResponse(
            id=user_id,
            email=current_user.email,
            full_name=request.full_name if request and request.full_name else current_user.full_name,
            experience_level=updated_preferences.get("experience_level", "intermediate"),
            programming_languages=updated_preferences.get("programming_languages", []),
            ai_ml_experience=updated_preferences.get("ai_ml_experience", "intermediate"),
            hardware_experience=updated_preferences.get("hardware_experience", "beginner"),
            learning_goals=updated_preferences.get("learning_goals"),
            gpu_access=updated_preferences.get("gpu_access", False),
            robotics_kit_experience=updated_preferences.get("robotics_kit_experience"),
            preferred_topics=updated_preferences.get("preferred_topics", []),
            personalization_enabled=updated_preferences.get("personalization_enabled", True),
            is_active=current_user.is_active
        )

        return profile_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))