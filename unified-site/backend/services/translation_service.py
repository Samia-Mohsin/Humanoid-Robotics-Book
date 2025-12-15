from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from core.config import settings
import asyncio
import logging
import aiohttp

logger = logging.getLogger(__name__)

class TranslationService:
    def __init__(self):
        # Initialize OpenAI client for translation
        self.llm = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model_name=settings.OPENAI_MODEL
        )

    async def translate(self, text: str, source_lang: str = "en", target_lang: str = "ur", user_id: Optional[str] = None) -> str:
        """
        Translate text from source language to target language
        """
        try:
            # For Urdu translation, we'll use OpenAI to translate the text
            # In a real implementation, you might use a dedicated translation API like Google Translate

            # Create a prompt for translation
            prompt = f"""Translate the following text from {source_lang} to {target_lang}.
            Preserve the meaning, tone, and technical terminology as much as possible.

            Text to translate:
            {text}

            Translation:"""

            # Use OpenAI for translation
            response = await self.llm.ainvoke(prompt)
            translated_text = response.content if hasattr(response, 'content') else str(response)

            # Log the translation for analytics (in a real implementation)
            logger.info(f"Translation completed: {source_lang} -> {target_lang}, user_id: {user_id}")

            return translated_text

        except Exception as e:
            logger.error(f"Error translating text: {str(e)}")
            # Fallback to original text with error message
            return f"Translation failed: {str(e)}\n\nOriginal text:\n{text}"

    async def detect_language(self, text: str) -> str:
        """
        Detect the language of the provided text
        """
        try:
            # In a real implementation, this would call a language detection API
            # For now, we'll assume English if it contains Latin characters
            # and return a mock detection

            # Simple heuristic to detect language
            if any(ord(c) > 127 for c in text[:100]):  # Check first 100 characters for non-Latin
                # If we see non-Latin characters, it might be Urdu or another language
                if any('\u0600' <= c <= '\u06FF' for c in text):  # Arabic/Persian/Urdu range
                    return "ur"
                else:
                    return "en"  # Default to English
            else:
                return "en"
        except Exception as e:
            logger.error(f"Error detecting language: {str(e)}")
            return "en"

    async def get_translation_status(self, user_id: str) -> Dict[str, Any]:
        """
        Get translation statistics and status for a user
        """
        try:
            # Mock response - in real implementation, this would fetch from database
            return {
                "total_translations": 0,
                "last_translation": None,
                "supported_languages": ["ur", "es", "fr", "de"],
                "monthly_quota": 1000,
                "used_quota": 0,
                "remaining_quota": 1000
            }
        except Exception as e:
            logger.error(f"Error getting translation status: {str(e)}")
            return {
                "total_translations": 0,
                "last_translation": None,
                "supported_languages": ["ur"],
                "monthly_quota": 1000,
                "used_quota": 0,
                "remaining_quota": 1000
            }