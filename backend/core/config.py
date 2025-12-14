from pydantic_settings import SettingsConfigDict, BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

    # Application settings
    APP_NAME: str = "NeuralReader"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database settings
    DATABASE_URL: str = "postgresql+asyncpg://neuralreader_user:neuralreader_password@localhost:5432/neuralreader"
    DB_POOL_SIZE: int = 20
    DB_POOL_OVERFLOW: int = 10
    DB_ECHO: bool = False

    # Security settings
    SECRET_KEY: str = "your-super-secret-jwt-signing-key-here-make-it-long-and-random"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Qdrant settings
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "neuralreader_books"

    # OpenAI settings
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o"
    EMBEDDING_MODEL: str = "text-embedding-3-large"

    # Rate limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 3600  # 1 hour in seconds

    # CORS settings
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:8000,https://yourdomain.com"

    # File upload settings
    MAX_FILE_SIZE: int = 52428800  # 50MB in bytes
    ALLOWED_FILE_TYPES: str = "application/pdf,text/plain,application/epub+zip"


# Create a single instance of settings
settings = Settings()