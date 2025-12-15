from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_async_session
from api.deps import get_current_user
from models.user import User
from services.ingestion_service import get_ingestion_service
from services.content_service import ContentService
from typing import Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/ingest")
async def ingest_content_endpoint(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),  # Make file optional to support both scenarios
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Endpoint to ingest/re-index book content
    If file is provided: process uploaded file
    If no file: re-index existing book content
    """
    try:
        ingestion_service = get_ingestion_service()

        if file is not None:
            # Handle file upload scenario
            # Validate file type
            if file.content_type not in ["application/pdf", "text/plain", "application/epub+zip"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Unsupported file type. Supported types: PDF, TXT, EPUB"
                )

            # Validate file size (max 50MB)
            file_size = len(await file.read())
            await file.seek(0)  # Reset file pointer after reading size

            if file_size > 50 * 1024 * 1024:  # 50MB
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="File size exceeds maximum limit of 50MB"
                )

            # Create ingestion log
            ingestion_log = await ingestion_service.create_ingestion_log(
                db=db,
                operation_type="add",
                content_id=f"upload-{file.filename}",
                user_id=current_user.id
            )

            # Process the ingestion in the background
            background_tasks.add_task(
                ingestion_service.process_book_ingestion,
                file,
                current_user.id,
                ingestion_log.log_id
            )

            return {
                "success": True,
                "message": "File ingestion process started successfully",
                "job_id": ingestion_log.log_id,
                "estimated_completion": "Processing time varies based on content size"
            }
        else:
            # Handle re-indexing scenario - re-index existing book content
            content_service = ContentService()

            # Create ingestion log for re-indexing
            ingestion_log = await ingestion_service.create_ingestion_log(
                db=db,
                operation_type="reindex",
                content_id="book-reindex",
                user_id=current_user.id
            )

            # Process the re-indexing in the background
            background_tasks.add_task(
                content_service.load_book_content  # This will re-index the existing content
            )

            return {
                "success": True,
                "message": "Book re-indexing process started successfully",
                "job_id": ingestion_log.log_id,
                "estimated_completion": "Processing time varies based on content size"
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting ingestion process: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion process failed to start: {str(e)}"
        )


@router.post("/ingest/manual")
async def manual_ingest_content_endpoint(
    content: str,
    title: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Endpoint to manually ingest content as text
    """
    try:
        ingestion_service = get_ingestion_service()

        # Create ingestion log
        ingestion_log = await ingestion_service.create_ingestion_log(
            db=db,
            operation_type="add",
            content_id=f"manual-{len(content)}-chars",
            user_id=current_user.id
        )

        # Process the ingestion in the background
        background_tasks = BackgroundTasks()
        background_tasks.add_task(
            ingestion_service.process_manual_ingestion,
            content=content,
            title=title,
            user_id=current_user.id,
            log_id=ingestion_log.log_id
        )

        return {
            "success": True,
            "message": "Manual ingestion process started successfully",
            "job_id": ingestion_log.log_id,
            "estimated_completion": "Processing time varies based on content size"
        }
    except Exception as e:
        logger.error(f"Error starting manual ingestion process: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Manual ingestion process failed: {str(e)}"
        )


@router.get("/ingest/status/{log_id}")
async def get_ingestion_status_endpoint(
    log_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Get the status of an ingestion job
    """
    try:
        ingestion_service = get_ingestion_service()
        log = await ingestion_service.get_ingestion_log_by_id(db, log_id)

        if not log:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Ingestion log not found"
            )

        # Only allow the user who created it or admins
        if current_user.id != log.user_id and not getattr(current_user, 'is_admin', False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this ingestion log"
            )

        return {
            "log_id": log.log_id,
            "operation_type": log.operation_type,
            "status": log.status,
            "processed_chunks": log.processed_chunks,
            "total_chunks": log.total_chunks,
            "error_details": log.error_details,
            "started_at": log.started_at.isoformat() if log.started_at else None,
            "completed_at": log.completed_at.isoformat() if log.completed_at else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting ingestion status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get ingestion status: {str(e)}"
        )