from typing import Optional, List, Dict, Any, AsyncGenerator
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from models.chat import ChatSession, ChatMessage
from models.user import User
from services.ai_service import get_ai_service
from uuid import uuid4
import logging

logger = logging.getLogger(__name__)

class ChatService:
    """
    Service for handling chat operations including session management and message handling
    """

    def __init__(self):
        self.ai_service = get_ai_service()

    async def create_chat_session(self, db: AsyncSession, user_id: str, title: Optional[str] = None) -> ChatSession:
        """
        Create a new chat session
        """
        if not title:
            title = f"Chat Session {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"

        session = ChatSession(
            session_id=str(uuid4()),
            user_id=user_id,
            title=title,
            created_at=datetime.utcnow()
        )

        db.add(session)
        await db.commit()
        await db.refresh(session)

        return session

    async def get_chat_session(self, db: AsyncSession, session_id: str) -> Optional[ChatSession]:
        """
        Get a specific chat session by ID
        """
        result = await db.execute(select(ChatSession).filter(ChatSession.session_id == session_id))
        return result.scalar_one_or_none()

    async def get_user_sessions(self, db: AsyncSession, user_id: str) -> List[ChatSession]:
        """
        Get all chat sessions for a user
        """
        result = await db.execute(select(ChatSession).filter(ChatSession.user_id == user_id))
        return result.scalars().all()

    async def add_message_to_session(self, db: AsyncSession, session_id: str, role: str, content: str) -> ChatMessage:
        """
        Add a message to a chat session
        """
        message = ChatMessage(
            message_id=str(uuid4()),
            session_id=session_id,
            role=role,
            content=content,
            timestamp=datetime.utcnow()
        )

        db.add(message)
        await db.commit()
        await db.refresh(message)

        return message

    async def get_session_messages(self, db: AsyncSession, session_id: str) -> List[ChatMessage]:
        """
        Get all messages for a specific session
        """
        result = await db.execute(
            select(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.timestamp.asc())
        )
        return result.scalars().all()

    async def get_recent_messages(self, db: AsyncSession, session_id: str, limit: int = 10) -> List[ChatMessage]:
        """
        Get the most recent messages for a session
        """
        result = await db.execute(
            select(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.timestamp.desc())
            .limit(limit)
        )
        messages = result.scalars().all()
        # Return in chronological order
        return list(reversed(messages))

    async def generate_ai_response(self, query: str, context: Optional[str] = None) -> str:
        """
        Generate an AI response to a query
        """
        try:
            response = await self.ai_service.generate_response(query, context)
            return response
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return "I encountered an error processing your request. Please try again."

    async def stream_ai_response(self, query: str, context: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        Stream an AI response to a query
        """
        try:
            async for chunk in self.ai_service.stream_response(query, context):
                yield chunk
        except Exception as e:
            logger.error(f"Error streaming AI response: {e}")
            yield "I encountered an error processing your request."


# Singleton instance
_chat_service = None

def get_chat_service() -> ChatService:
    """
    Get the singleton chat service instance
    """
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service