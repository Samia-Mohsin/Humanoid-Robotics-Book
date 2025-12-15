from typing import Optional, List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Qdrant
import qdrant_client
from qdrant_client.http import models
import asyncio
import os
from core.config import settings

import logging
logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self):
        # Initialize Qdrant client for RAG
        try:
            self.qdrant_client = qdrant_client.QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                api_key=settings.QDRANT_API_KEY
            )

            # Create vector store for book content
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

            self.vector_store = Qdrant(
                client=self.qdrant_client,
                collection_name=settings.QDRANT_COLLECTION_NAME,
                embeddings=embeddings
            )
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant vector store: {str(e)}")
            self.vector_store = None

        # Initialize LLM if API key is available
        try:
            if settings.OPENAI_API_KEY:
                self.llm = ChatOpenAI(
                    openai_api_key=settings.OPENAI_API_KEY,
                    model_name=settings.OPENAI_MODEL
                )

                # Create prompt template for the chat
                self.chat_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are an AI assistant for the Physical AI & Humanoid Robotics educational platform. Answer questions based on the provided context from the book. Be helpful, accurate, and provide detailed explanations appropriate for the user's background level. If the context doesn't contain the answer, say so and suggest related topics from the book."),
                    ("human", "Context: {context}\n\nQuestion: {question}\n\nIf this is about selected text, please provide a detailed explanation: {selected_text}")
                ])

                self.chain = self.chat_prompt | self.llm | StrOutputParser()
            else:
                # Fallback to a mock chain if no API key
                self.llm = None
                self.chain = None
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            self.llm = None
            self.chain = None

    async def process_chat(self, message: str, user_id: str, session_id: Optional[str] = None, selected_text: Optional[str] = None):
        """
        Process a chat message with RAG capabilities
        """
        try:
            # Retrieve relevant context from vector store
            context = await self._retrieve_context(message)

            # If selected text is provided, include it in the context
            if selected_text:
                context = f"Selected text: {selected_text}\n\n{context}"

            # Generate response using the LLM if available
            if self.chain is not None:
                response = await self.chain.ainvoke({
                    "context": context,
                    "question": message,
                    "selected_text": selected_text or ""
                })
            else:
                # Fallback response when no LLM is available
                response = f"This is a demo response for: {message}. Context: {context[:200] if context else 'No context available'}"

            # In a real implementation, you would save the chat message to the database
            # For now, we'll just return the response

            return {
                "response": response,
                "session_id": session_id or f"session_{user_id}_{hash(message)}",
                "context_used": [{"content": context, "score": 0.9}] if context else []  # Mock context with score
            }
        except Exception as e:
            # In case of error, return a helpful message
            return {
                "response": f"I'm having trouble processing your request. Please try again. Error: {str(e)}",
                "session_id": session_id or f"session_{user_id}_error",
                "context_used": []
            }

    async def _retrieve_context(self, query: str) -> str:
        """
        Retrieve relevant context from the vector store
        """
        try:
            # Query the Qdrant vector store for relevant documents
            search_results = self.vector_store.similarity_search(
                query,
                k=4  # Retrieve top 4 most relevant documents
            )

            # Combine the content from retrieved documents
            context_parts = []
            for doc in search_results:
                content = doc.page_content
                if content:
                    context_parts.append(content)

            context = "\n\n".join(context_parts)
            return context if context else ""
        except Exception as e:
            # If retrieval fails, return an empty context
            return ""

    async def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all chat sessions for a user
        """
        # In a real implementation, this would fetch from the database
        # For now, we'll return a mock response
        return [
            {
                "id": "session_1",
                "title": "Introduction to Physical AI",
                "created_at": "2023-12-14T10:00:00Z",
                "message_count": 5
            },
            {
                "id": "session_2",
                "title": "ROS2 Architecture",
                "created_at": "2023-12-14T11:30:00Z",
                "message_count": 8
            }
        ]

    async def get_session_messages(self, session_id: str, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all messages in a specific session
        """
        # In a real implementation, this would fetch from the database
        # For now, we'll return a mock response
        return [
            {
                "id": "msg_1",
                "role": "user",
                "content": "What is Physical AI?",
                "timestamp": "2023-12-14T10:00:00Z"
            },
            {
                "id": "msg_2",
                "role": "assistant",
                "content": "Physical AI is an interdisciplinary field that combines artificial intelligence with physical systems, focusing on how AI agents can interact with and learn from the physical world.",
                "timestamp": "2023-12-14T10:01:00Z"
            }
        ]