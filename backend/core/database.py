from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import AsyncAdaptedQueuePool, StaticPool
from core.config import settings
import logging

logger = logging.getLogger(__name__)

# Create the async engine
# Use different pool settings based on database type
if settings.DATABASE_URL.startswith("sqlite"):
    # SQLite-specific settings
    engine = create_async_engine(
        settings.DATABASE_URL,
        poolclass=StaticPool,  # Use StaticPool for SQLite
        connect_args={"check_same_thread": False},  # Required for SQLite
        echo=settings.DEBUG  # Log SQL statements in debug mode
    )
else:
    # PostgreSQL-specific settings
    engine = create_async_engine(
        settings.DATABASE_URL,
        poolclass=AsyncAdaptedQueuePool,
        pool_pre_ping=True,  # Verify connections before use
        pool_recycle=300,    # Recycle connections after 5 minutes
        echo=settings.DEBUG  # Log SQL statements in debug mode
    )

# Create async session maker
async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Base class for all models
class Base(DeclarativeBase):
    pass

async def get_db_session():
    """
    Dependency to get database session
    """
    async with async_session() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()

# Async context manager for database operations
class DatabaseManager:
    def __init__(self):
        self.engine = engine
        self.session_factory = async_session

    async def create_tables(self):
        """Create all database tables"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")

    async def drop_tables(self):
        """Drop all database tables (for testing purposes)"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.info("Database tables dropped successfully")

# Create a global instance
db_manager = DatabaseManager()