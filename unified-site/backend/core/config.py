from pydantic_settings import BaseSettings
from typing import Optional, List
import os

class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "NeuralReader"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://neuralreader_user:neuralreader_password@localhost:5432/neuralreader")
    DB_POOL_SIZE: int = 20
    DB_POOL_OVERFLOW: int = 10
    DB_ECHO: bool = False

    # Neon Database settings
    NEON_DATABASE_URL: str = os.getenv("NEON_DATABASE_URL", "")

    # Security settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-super-secret-jwt-signing-key-here-make-it-long-and-random")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Qdrant settings
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "neuralreader_books")

    # OpenAI settings
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

    # Rate limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 3600  # 1 hour in seconds

    # CORS settings
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000", "https://yourdomain.com"]

    # File upload settings
    MAX_FILE_SIZE: int = 52428800  # 50MB in bytes
    ALLOWED_FILE_TYPES: List[str] = ["application/pdf", "text/plain", "application/epub+zip"]

    class Config:
        env_file = ".env"

# Create a single instance of settings
settings = Settings()