from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_async_session
from services.content_service import ContentService
from api.deps import get_current_user
from models.user import User

router = APIRouter()

class PersonalizeContentRequest(BaseModel):
    content: str
    context: Optional[Dict[str, Any]] = None

class PersonalizeContentResponse(BaseModel):
    personalized_content: str

@router.post("/content/personalize", response_model=PersonalizeContentResponse)
async def personalize_content(
    request: PersonalizeContentRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Personalize content based on user preferences and background
    """
    content_service = ContentService()

    try:
        personalized_content = await content_service.personalize_content(
            content=request.content,
            user_id=current_user.id,
            context=request.context or {}
        )

        return PersonalizeContentResponse(
            personalized_content=personalized_content
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/content/chapters")
async def get_available_chapters(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Get list of available chapters/modules in the book
    """
    content_service = ContentService()

    try:
        chapters = await content_service.get_available_chapters()
        return chapters
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/content/chapter/{chapter_id}")
async def get_chapter_content(
    chapter_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Get content for a specific chapter, potentially personalized
    """
    content_service = ContentService()

    try:
        content = await content_service.get_chapter_content(
            chapter_id=chapter_id,
            user_id=current_user.id
        )
        return content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))