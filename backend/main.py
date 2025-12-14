from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from api.routes import chat, translate, profile, ingest
from core.config import settings
from core.database import engine, Base
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing NeuralReader application...")

    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database tables created successfully")

    yield

    # Shutdown
    logger.info("Shutting down NeuralReader application...")

# Create FastAPI app with custom settings
app = FastAPI(
    title="NeuralReader API",
    description="AI-powered educational platform with RAG capabilities for Physical AI & Humanoid Robotics",
    version="1.0.0",
    lifespan=lifespan,
    openapi_url="/api/openapi.json",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Additional security headers can be added here
)

# Include API routes
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(translate.router, prefix="/api", tags=["translate"])
app.include_router(profile.router, prefix="/api", tags=["profile"])
app.include_router(ingest.router, prefix="/api", tags=["ingest"])

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return {"message": "Welcome to NeuralReader API", "status": "healthy", "docs": "/api/docs"}

@app.get("/api/")
async def api_root():
    return {"message": "Welcome to NeuralReader API", "status": "healthy", "docs": "/api/docs"}

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