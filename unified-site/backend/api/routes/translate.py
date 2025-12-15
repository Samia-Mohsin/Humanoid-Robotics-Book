from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_async_session
from services.translation_service import TranslationService
from api.deps import get_current_user
from models.user import User

router = APIRouter()

class TranslateRequest(BaseModel):
    text: str
    source_lang: Optional[str] = "en"
    target_lang: str = "ur"  # Default to Urdu
    user_id: Optional[str] = None

class TranslateResponse(BaseModel):
    translated_text: str
    source_lang: str
    target_lang: str

@router.post("/translate", response_model=TranslateResponse)
async def translate_text(
    request: TranslateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Translate text from source language to target language
    """
    translation_service = TranslationService()

    try:
        translated_text = await translation_service.translate(
            text=request.text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            user_id=current_user.id if current_user else request.user_id
        )

        return TranslateResponse(
            translated_text=translated_text,
            source_lang=request.source_lang,
            target_lang=request.target_lang
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/translate/detect")
async def detect_language(
    request: TranslateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Detect the language of the provided text
    """
    translation_service = TranslationService()

    try:
        detected_lang = await translation_service.detect_language(request.text)
        return {"detected_language": detected_lang}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))