from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from models.chapter import ChapterContent, IngestionLog
from services.ai_service import get_ai_service
from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4
import logging
import time
from datetime import datetime
import tempfile
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class IngestionService:
    """
    Service for handling content ingestion, chunking, and vector storage
    """

    def __init__(self):
        self.ai_service = get_ai_service()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    async def process_book_ingestion(self, file, user_id: str, log_id: str):
        """
        Process book ingestion: read file, chunk text, embed, and store in vector database
        """
        from core.database import get_db_session
        from sqlalchemy import create_engine
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

        # Get database session
        async with get_db_session() as db:
            try:
                # Update ingestion log status to processing
                ingestion_log = await self.update_ingestion_log_status(db, log_id, "processing")

                # Read the file content based on type
                content = await self._read_file_content(file)

                if not content:
                    await self.update_ingestion_log_status(db, log_id, "failed", error_details="Could not read file content")
                    return

                # Chunk the text
                chunks = self.text_splitter.split_text(content)

                # Update log with total chunks
                ingestion_log.total_chunks = len(chunks)
                await db.commit()

                # Process each chunk
                processed_count = 0
                for i, chunk in enumerate(chunks):
                    try:
                        # Create metadata for the chunk
                        metadata = {
                            "source_file": file.filename,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "user_id": user_id,
                            "created_at": datetime.utcnow().isoformat()
                        }

                        # Add to vector store
                        await self.ai_service.add_to_vector_store(
                            texts=[chunk],
                            metadatas=[metadata],
                            ids=[f"{log_id}-chunk-{i}"]
                        )

                        processed_count += 1

                        # Update log with progress
                        ingestion_log.processed_chunks = processed_count
                        await db.commit()

                    except Exception as e:
                        logger.error(f"Error processing chunk {i}: {e}")
                        await self.update_ingestion_log_status(
                            db,
                            log_id,
                            "failed",
                            error_details=f"Error processing chunk {i}: {str(e)}"
                        )
                        return

                # Update log status to completed
                await self.update_ingestion_log_status(db, log_id, "completed")

                logger.info(f"Successfully ingested {processed_count} chunks from {file.filename}")

            except Exception as e:
                logger.error(f"Error processing book ingestion: {e}")
                await self.update_ingestion_log_status(db, log_id, "failed", error_details=str(e))

    async def process_manual_ingestion(self, content: str, title: str, user_id: str, log_id: str):
        """
        Process manual content ingestion: chunk text, embed, and store in vector database
        """
        from core.database import get_db_session

        async with get_db_session() as db:
            try:
                # Update ingestion log status to processing
                ingestion_log = await self.update_ingestion_log_status(db, log_id, "processing")

                if not content:
                    await self.update_ingestion_log_status(db, log_id, "failed", error_details="No content provided")
                    return

                # Chunk the text
                chunks = self.text_splitter.split_text(content)

                # Update log with total chunks
                ingestion_log.total_chunks = len(chunks)
                await db.commit()

                # Process each chunk
                processed_count = 0
                for i, chunk in enumerate(chunks):
                    try:
                        # Create metadata for the chunk
                        metadata = {
                            "source_title": title,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "user_id": user_id,
                            "created_at": datetime.utcnow().isoformat()
                        }

                        # Add to vector store
                        await self.ai_service.add_to_vector_store(
                            texts=[chunk],
                            metadatas=[metadata],
                            ids=[f"{log_id}-chunk-{i}"]
                        )

                        processed_count += 1

                        # Update log with progress
                        ingestion_log.processed_chunks = processed_count
                        await db.commit()

                    except Exception as e:
                        logger.error(f"Error processing chunk {i}: {e}")
                        await self.update_ingestion_log_status(
                            db,
                            log_id,
                            "failed",
                            error_details=f"Error processing chunk {i}: {str(e)}"
                        )
                        return

                # Update log status to completed
                await self.update_ingestion_log_status(db, log_id, "completed")

                logger.info(f"Successfully ingested {processed_count} chunks from manual content")

            except Exception as e:
                logger.error(f"Error processing manual ingestion: {e}")
                await self.update_ingestion_log_status(db, log_id, "failed", error_details=str(e))

    async def _read_file_content(self, file) -> Optional[str]:
        """
        Read content from uploaded file based on its type
        """
        try:
            # Reset file pointer
            await file.seek(0)
            content = await file.read()

            # Determine file type and process accordingly
            if file.content_type == "application/pdf":
                return await self._extract_text_from_pdf(content)
            elif file.content_type == "text/plain":
                return content.decode('utf-8')
            elif file.content_type == "application/epub+zip":
                return await self._extract_text_from_epub(content)
            else:
                logger.error(f"Unsupported file type: {file.content_type}")
                return None
        except Exception as e:
            logger.error(f"Error reading file content: {e}")
            return None

    async def _extract_text_from_pdf(self, pdf_content: bytes) -> Optional[str]:
        """
        Extract text from PDF content
        """
        try:
            # This is a simplified implementation - in a real application,
            # you would add PyPDF2 or pypdf to requirements.txt
            # For now, we'll return an error message to indicate the dependency is needed
            logger.error("PDF processing requires PyPDF2 or pypdf. Add 'pypdf' to requirements.txt for PDF support.")
            return "PDF processing requires additional dependencies. Please install pypdf: pip install pypdf"
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return None

    async def _extract_text_from_epub(self, epub_content: bytes) -> Optional[str]:
        """
        Extract text from EPUB content
        """
        try:
            # This is a simplified implementation - in a real application,
            # you would add ebooklib to requirements.txt
            # For now, we'll return an error message to indicate the dependency is needed
            logger.error("EPUB processing requires ebooklib. Add 'ebooklib' to requirements.txt for EPUB support.")
            return "EPUB processing requires additional dependencies. Please install ebooklib: pip install ebooklib"
        except Exception as e:
            logger.error(f"Error extracting text from EPUB: {e}")
            return None

    async def create_ingestion_log(self, db: AsyncSession, operation_type: str, content_id: str, user_id: str) -> IngestionLog:
        """
        Create an ingestion log entry
        """
        log = IngestionLog(
            log_id=str(uuid4()),
            operation_type=operation_type,
            content_id=content_id,
            user_id=user_id,
            status="pending",
            started_at=datetime.utcnow()
        )

        db.add(log)
        await db.commit()
        await db.refresh(log)

        return log

    async def get_ingestion_log_by_id(self, db: AsyncSession, log_id: str) -> Optional[IngestionLog]:
        """
        Get ingestion log by ID
        """
        result = await db.execute(select(IngestionLog).filter(IngestionLog.log_id == log_id))
        return result.scalar_one_or_none()

    async def update_ingestion_log_status(self, db: AsyncSession, log_id: str, status: str, error_details: Optional[str] = None) -> Optional[IngestionLog]:
        """
        Update the status of an ingestion log
        """
        result = await db.execute(select(IngestionLog).filter(IngestionLog.log_id == log_id))
        log = result.scalar_one_or_none()

        if log:
            log.status = status
            if error_details:
                log.error_details = error_details
            if status == "completed":
                log.completed_at = datetime.utcnow()

            await db.commit()
            await db.refresh(log)

        return log


# Singleton instance
_ingestion_service = None

def get_ingestion_service() -> IngestionService:
    """
    Get the singleton ingestion service instance
    """
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = IngestionService()
    return _ingestion_service