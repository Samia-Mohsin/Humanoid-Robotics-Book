from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db_session
from api.deps import get_current_active_user
from models.user import User
from schemas.translate import TranslateRequest, TranslateResponse, TranslationJobCreate, TranslationJobInDB
from services.translation_service import get_translation_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/translate/chapter", response_model=TranslateResponse)
async def translate_chapter_endpoint(
    request: TranslateRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Translate a chapter to the specified target language
    """
    try:
        translation_service = get_translation_service()

        # Perform the translation
        result = await translation_service.translate_chapter(
            chapter_text=request.chapter_text,
            target_language=request.target_language,
            source_language=request.source_language,
            preserve_format=request.preserve_format
        )

        return TranslateResponse(
            chapter_id="temp-chapter-id",  # This would come from the actual chapter being translated
            translated_content=result["translated_text"],
            target_language=request.target_language,
            source_language=request.source_language,
            processing_time=result["processing_time"],
            metadata=result["metadata"]
        )
    except Exception as e:
        logger.error(f"Error translating chapter: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {str(e)}"
        )


@router.post("/translate/job", response_model=TranslationJobInDB)
async def create_translation_job_endpoint(
    request: TranslationJobCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Create a translation job for asynchronous processing
    """
    try:
        translation_service = get_translation_service()

        # Create the translation job
        job = await translation_service.create_translation_job(
            db=db,
            chapter_id=request.chapter_id,
            target_language=request.target_language,
            source_language=request.source_language,
            user_id=current_user.user_id
        )

        return job
    except Exception as e:
        logger.error(f"Error creating translation job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create translation job: {str(e)}"
        )


@router.get("/translate/job/{job_id}", response_model=TranslationJobInDB)
async def get_translation_job_endpoint(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get the status of a translation job
    """
    try:
        translation_service = get_translation_service()

        job = await translation_service.get_translation_job(db, job_id, current_user.user_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Translation job not found"
            )

        return job
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting translation job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get translation job: {str(e)}"
        )