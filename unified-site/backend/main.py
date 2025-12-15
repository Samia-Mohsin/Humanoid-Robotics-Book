from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import os
from datetime import datetime
import asyncio
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Qdrant
import qdrant_client
from qdrant_client.http import models
import openai
from core.config import settings
from core.database import Base
from api.deps import get_current_user
from models.user import User, UserPreferences
from models.chat import ChatSession, ChatMessage
from models.content import ContentChunk
from services.chat_service import ChatService
from services.personalization_service import PersonalizationService
from services.translation_service import TranslationService
from services.content_service import ContentService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing NeuralReader application...")

    # Initialize services
    app.state.chat_service = ChatService()
    app.state.personalization_service = PersonalizationService()
    app.state.translation_service = TranslationService()
    app.state.content_service = ContentService()

    # Load content into vector store
    await app.state.content_service.load_book_content()

    logger.info("Application initialized successfully")

    yield

    # Shutdown
    logger.info("Shutting down NeuralReader application...")

# Create FastAPI app with custom settings
app = FastAPI(
    title="NeuralReader API",
    description="AI-powered educational platform with RAG capabilities for Physical AI & Humanoid Robotics",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
from api.routes import chat, translate, translate_chapter, profile, profile_by_id, content, ingest
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(translate.router, prefix="/api", tags=["translate"])
app.include_router(translate_chapter.router, prefix="/api", tags=["translate"])
app.include_router(profile.router, prefix="/api", tags=["profile"])
app.include_router(profile_by_id.router, prefix="/api", tags=["profile"])
app.include_router(content.router, prefix="/api", tags=["content"])
app.include_router(ingest.router, prefix="/api", tags=["ingest"])

@app.get("/")
async def root():
    return {"message": "Welcome to NeuralReader API", "status": "healthy", "docs": "/docs"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "NeuralReader API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )