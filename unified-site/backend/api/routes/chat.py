from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_async_session
from services.chat_service import ChatService
from api.deps import get_current_user
from models.user import User

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    selected_text: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    context_used: Optional[List[Dict[str, Any]]] = None

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Main chat endpoint that handles both general questions and selected text queries
    """
    chat_service = ChatService()

    try:
        response = await chat_service.process_chat(
            message=request.message,
            user_id=current_user.id if current_user else request.user_id,
            session_id=request.session_id,
            selected_text=request.selected_text
        )

        return ChatResponse(
            response=response.get("response", ""),
            session_id=response.get("session_id", ""),
            context_used=response.get("context_used", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chat/sessions")
async def get_user_sessions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Get all chat sessions for the current user
    """
    chat_service = ChatService()
    sessions = await chat_service.get_user_sessions(current_user.id)
    return sessions

@router.get("/chat/session/{session_id}")
async def get_session_messages(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_session)
):
    """
    Get all messages in a specific session
    """
    chat_service = ChatService()
    messages = await chat_service.get_session_messages(session_id, current_user.id)
    return messages