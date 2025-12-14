from fastapi import APIRouter, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db_session
from api.deps import get_current_active_user
from models.user import User
from schemas.chat import ChatRequest, ChatResponse
from services.chat_service import get_chat_service
from services.ai_service import get_ai_service
from typing import AsyncGenerator
import asyncio
import json
from fastapi.responses import StreamingResponse
from datetime import datetime

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Chat endpoint that handles user queries with RAG capabilities using LangChain
    """
    chat_service = get_chat_service()

    # Get or create chat session
    if request.conversation_id:
        session = await chat_service.get_chat_session(db, request.conversation_id)
        if not session:
            # Create new session if provided ID doesn't exist
            session = await chat_service.create_chat_session(db, current_user.user_id, f"Chat: {request.message[:50]}...")
    else:
        session = await chat_service.create_chat_session(db, current_user.user_id, f"Chat: {request.message[:50]}...")

    # Add user message to session
    await chat_service.add_message_to_session(db, session.session_id, "user", request.message)

    # Generate AI response
    ai_response = await chat_service.generate_ai_response(request.message, request.context)

    # Add AI response to session
    await chat_service.add_message_to_session(db, session.session_id, "assistant", ai_response)

    return ChatResponse(
        conversation_id=session.session_id,
        response=ai_response,
        timestamp=datetime.utcnow()
    )


@router.post("/chat/stream")
async def chat_stream_endpoint(
    request: ChatRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Streaming chat endpoint that returns responses as Server-Sent Events
    """
    chat_service = get_chat_service()

    # Get or create chat session
    if request.conversation_id:
        session = await chat_service.get_chat_session(db, request.conversation_id)
        if not session:
            # Create new session if provided ID doesn't exist
            session = await chat_service.create_chat_session(db, current_user.user_id, f"Chat: {request.message[:50]}...")
    else:
        session = await chat_service.create_chat_session(db, current_user.user_id, f"Chat: {request.message[:50]}...")

    # Add user message to session
    await chat_service.add_message_to_session(db, session.session_id, "user", request.message)

    async def event_generator():
        try:
            # Process the chat request with streaming
            async for chunk in chat_service.stream_ai_response(
                query=request.message,
                context=request.context
            ):
                yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")