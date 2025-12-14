from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from jose import JWTError, jwt
from core.database import get_db_session
from core.config import settings
from models.user import User
import logging

logger = logging.getLogger(__name__)

# Security scheme for API
security = HTTPBearer()

async def get_current_user(
    token: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db_session)
) -> User:
    """
    Get the current user from the provided token
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token.credentials, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id: str = payload.get("sub")

        if user_id is None:
            raise credentials_exception

    except JWTError:
        raise credentials_exception

    # Fetch user from database
    from services.profile_service import get_user_by_id
    user = await get_user_by_id(db, int(user_id))
    if user is None:
        raise credentials_exception

    return user

def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Get the current active user (checks if user is active)
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user

def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """
    Require admin privileges for the current user
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user

# Rate limiting dependency (placeholder implementation)
from functools import wraps
from typing import Callable, Awaitable

def rate_limit(max_requests: int = settings.RATE_LIMIT_REQUESTS, window: int = settings.RATE_LIMIT_WINDOW):
    """
    Rate limiting decorator (placeholder implementation)
    In a real implementation, you would use a Redis or memory-based rate limiter
    """
    def rate_limit_dependency():
        # In a real implementation, you would check rate limits here
        # This is a simplified version that always allows requests
        pass
    return rate_limit_dependency

# Dependency for admin-only endpoints
async def require_admin_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Dependency to require admin privileges
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operation not allowed, admin access required"
        )
    return current_user