from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_async_session
from core.config import settings
from models.user import User
import logging

logger = logging.getLogger(__name__)

security = HTTPBearer()

def create_access_token(data: dict):
    """
    Create a new access token
    """
    to_encode = data.copy()
    # In a real implementation, you would add expiration
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_async_session)
) -> User:
    """
    Get current user from JWT token
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(credentials.credentials, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # In a real implementation, you would fetch the user from the database
    # For now, we'll return a mock user
    return User(
        id=user_id,
        email="mock@example.com",
        full_name="Mock User",
        is_active=True,
        is_verified=True
    )

def get_current_active_user(current_user: User = Depends(get_current_user)):
    """
    Get current active user
    """
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


async def require_admin_user(current_user: User = Depends(get_current_user)):
    """
    Dependency to require admin user
    """
    if not getattr(current_user, 'is_admin', False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user