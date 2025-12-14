from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from core.config import settings
from models.chapter import TranslationJob, ChapterContent
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from uuid import uuid4
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class TranslationService:
    """
    Service for handling translation operations
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model_name=settings.OPENAI_MODEL,
            temperature=0.3,
            openai_api_key=settings.OPENAI_API_KEY
        )

    async def translate_chapter(
        self,
        chapter_text: str,
        target_language: str,
        source_language: str = "en",
        preserve_format: bool = True
    ) -> Dict[str, Any]:
        """
        Translate a chapter to the target language
        """
        start_time = time.time()

        try:
            # Create translation prompt
            if preserve_format:
                prompt = f"""
                Translate the following text from {source_language} to {target_language}.
                Preserve the original formatting, structure, and style as much as possible.
                Maintain paragraph breaks, lists, and any special formatting.

                Text to translate:
                {chapter_text}
                """
            else:
                prompt = f"""
                Translate the following text from {source_language} to {target_language}:
                {chapter_text}
                """

            # Call the LLM for translation
            response = await self.llm.ainvoke(prompt)

            processing_time = time.time() - start_time

            return {
                "translated_text": response.content,
                "processing_time": processing_time,
                "metadata": {
                    "source_language": source_language,
                    "target_language": target_language,
                    "preserve_format": preserve_format,
                    "original_length": len(chapter_text),
                    "translated_length": len(response.content)
                }
            }

        except Exception as e:
            logger.error(f"Error translating chapter: {e}")
            processing_time = time.time() - start_time
            return {
                "translated_text": f"Translation failed: {str(e)}",
                "processing_time": processing_time,
                "metadata": {
                    "source_language": source_language,
                    "target_language": target_language,
                    "preserve_format": preserve_format,
                    "original_length": len(chapter_text),
                    "translated_length": 0
                }
            }

    async def create_translation_job(
        self,
        db: AsyncSession,
        chapter_id: str,
        target_language: str,
        source_language: str,
        user_id: Optional[str] = None
    ) -> TranslationJob:
        """
        Create a translation job in the database
        """
        job = TranslationJob(
            job_id=str(uuid4()),
            chapter_id=chapter_id,
            target_language=target_language,
            source_language=source_language,
            user_id=user_id,
            status="pending",
            created_at=datetime.utcnow()
        )

        db.add(job)
        await db.commit()
        await db.refresh(job)

        return job

    async def get_translation_job(self, db: AsyncSession, job_id: str, user_id: str) -> Optional[TranslationJob]:
        """
        Get a translation job by ID, ensuring it belongs to the user
        """
        result = await db.execute(
            select(TranslationJob)
            .filter(TranslationJob.job_id == job_id)
            .filter(TranslationJob.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def update_translation_job_status(self, db: AsyncSession, job_id: str, status: str, result: Optional[str] = None) -> Optional[TranslationJob]:
        """
        Update the status of a translation job
        """
        result_query = await db.execute(select(TranslationJob).filter(TranslationJob.job_id == job_id))
        job = result_query.scalar_one_or_none()

        if job:
            job.status = status
            if result:
                job.result = result
            job.updated_at = datetime.utcnow()

            await db.commit()
            await db.refresh(job)

        return job


# Singleton instance
_translation_service = None

def get_translation_service() -> TranslationService:
    """
    Get the singleton translation service instance
    """
    global _translation_service
    if _translation_service is None:
        _translation_service = TranslationService()
    return _translation_service