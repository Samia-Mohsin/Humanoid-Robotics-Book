from datetime import datetime, timedelta
from typing import Optional, Union
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db_session
from models.user import User
from core.config import settings
import logging

logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security scheme for API
security = HTTPBearer()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against a hashed password
    """
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """
    Hash a password
    """
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create an access token with the provided data
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)  # Default 15 minutes

    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

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
    user = await db.get(User, int(user_id))
    if user is None:
        raise credentials_exception

    return user

def verify_admin(user: User = Depends(get_current_user)) -> User:
    """
    Verify that the current user is an admin
    """
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operation not allowed, admin access required"
        )
    return user

# Optional: Rate limiting decorator (would need additional dependencies like redis)
from functools import wraps
from core.config import settings

def rate_limit(max_requests: int = settings.RATE_LIMIT_REQUESTS, window: int = settings.RATE_LIMIT_WINDOW):
    """
    Rate limiting decorator (placeholder implementation)
    In a real implementation, you would use a Redis or memory-based rate limiter
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # In a real implementation, you would check rate limits here
            # This is a simplified version that always allows requests
            return await func(*args, **kwargs)
        return wrapper
    return decorator