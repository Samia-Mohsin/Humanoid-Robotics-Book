from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from models.user import User
from models.user_profile import UserProfile
from uuid import uuid4
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ProfileService:
    """
    Service for handling user profile operations
    """

    async def get_user_profile(self, db: AsyncSession, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user profile information
        """
        try:
            # Get the user
            user_result = await db.execute(select(User).filter(User.user_id == user_id))
            user = user_result.scalar_one_or_none()

            if not user:
                return None

            # Get or create user profile
            profile_result = await db.execute(select(UserProfile).filter(UserProfile.user_id == user_id))
            profile = profile_result.scalar_one_or_none()

            if not profile:
                # Create a default profile if it doesn't exist
                profile = UserProfile(
                    user_id=user_id,
                    preferences={},
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                db.add(profile)
                await db.commit()
                await db.refresh(profile)

            return {
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "is_active": user.is_active,
                "is_admin": user.is_admin,
                "created_at": user.created_at,
                "preferences": profile.preferences,
                "profile_created_at": profile.created_at,
                "profile_updated_at": profile.updated_at
            }

        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
            return None

    async def update_user_profile(
        self,
        db: AsyncSession,
        user_id: str,
        preferences: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Update user profile information
        """
        try:
            # Get the user
            user_result = await db.execute(select(User).filter(User.user_id == user_id))
            user = user_result.scalar_one_or_none()

            if not user:
                return None

            # Get or create user profile
            profile_result = await db.execute(select(UserProfile).filter(UserProfile.user_id == user_id))
            profile = profile_result.scalar_one_or_none()

            if not profile:
                # Create a new profile if it doesn't exist
                profile = UserProfile(
                    user_id=user_id,
                    preferences=preferences or {},
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                db.add(profile)
            else:
                # Update existing profile
                profile.preferences = preferences or profile.preferences
                profile.updated_at = datetime.utcnow()

            await db.commit()
            await db.refresh(profile)

            return {
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "is_active": user.is_active,
                "is_admin": user.is_admin,
                "created_at": user.created_at,
                "preferences": profile.preferences,
                "profile_created_at": profile.created_at,
                "profile_updated_at": profile.updated_at
            }

        except Exception as e:
            logger.error(f"Error updating user profile: {e}")
            return None


# Singleton instance
_profile_service = None

def get_profile_service() -> ProfileService:
    """
    Get the singleton profile service instance
    """
    global _profile_service
    if _profile_service is None:
        _profile_service = ProfileService()
    return _profile_service