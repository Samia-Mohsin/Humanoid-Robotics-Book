from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db_session
from api.deps import get_current_active_user
from models.user import User
from schemas.profile import UserProfileResponse, UpdateUserProfileRequest
from services.profile_service import get_profile_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/profile/{user_id}", response_model=UserProfileResponse)
async def get_user_profile_endpoint(
    user_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get user profile information
    """
    try:
        # Only allow users to get their own profile or admins to get any profile
        if str(current_user.user_id) != user_id and not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this profile"
            )

        profile_service = get_profile_service()
        profile = await profile_service.get_user_profile(db, user_id)
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found"
            )

        # Convert the dict response to UserProfileResponse
        return UserProfileResponse(
            user_id=profile["user_id"],
            username=profile["username"],
            email=profile["email"],
            full_name=profile["full_name"],
            is_active=profile["is_active"],
            is_admin=profile["is_admin"],
            created_at=profile["created_at"],
            preferences=profile["preferences"],
            profile_created_at=profile["profile_created_at"],
            profile_updated_at=profile["profile_updated_at"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve user profile: {str(e)}"
        )


@router.post("/profile/{user_id}", response_model=UserProfileResponse)
async def update_user_profile_endpoint(
    user_id: str,
    request: UpdateUserProfileRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Update user profile information
    """
    try:
        # Only allow users to update their own profile
        if str(current_user.user_id) != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to update this profile"
            )

        profile_service = get_profile_service()
        updated_profile = await profile_service.update_user_profile(
            db=db,
            user_id=user_id,
            preferences=request.preferences
        )

        if not updated_profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found"
            )

        # Convert the dict response to UserProfileResponse
        return UserProfileResponse(
            user_id=updated_profile["user_id"],
            username=updated_profile["username"],
            email=updated_profile["email"],
            full_name=updated_profile["full_name"],
            is_active=updated_profile["is_active"],
            is_admin=updated_profile["is_admin"],
            created_at=updated_profile["created_at"],
            preferences=updated_profile["preferences"],
            profile_created_at=updated_profile["profile_created_at"],
            profile_updated_at=updated_profile["profile_updated_at"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update user profile: {str(e)}"
        )