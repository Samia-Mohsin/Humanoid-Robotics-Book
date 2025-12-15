from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_async_session
from services.translation_service import TranslationService
from api.deps import get_current_user
from models.user import User

router = APIRouter()

class TranslateChapterRequest(BaseModel):
    content: str
    source_language: Optional[str] = "en"
    target_language: str = "ur"  # Default to Urdu
    user_id: Optional[str] = None

class TranslateChapterResponse(BaseModel):
    translated_content: str
    source_language: str
    target_language: str

@router.post("/translate/chapter", response_model=TranslateChapterResponse)
async def translate_chapter(
    request: TranslateChapterRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Translate book chapter content from source language to target language
    """
    translation_service = TranslationService()

    try:
        translated_text = await translation_service.translate(
            text=request.content,
            source_lang=request.source_language,
            target_lang=request.target_language,
            user_id=current_user.id if current_user else request.user_id
        )

        return TranslateChapterResponse(
            translated_content=translated_text,
            source_language=request.source_language,
            target_language=request.target_language
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))